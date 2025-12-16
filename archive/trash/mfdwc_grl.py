"""
MFDWC-GRL: Domain Adaptation for Acoustic Scene Classification
Using Mel-Frequency Discrete Wavelet Cepstral Coefficients with Gradient Reversal Layer
"""

import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import pandas as pd
from collections import defaultdict
import pywt

# Import the MFDWC extractor
from mfdwc_extractor import MFDWCFeatureExtractor, StatisticalAggregator

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Configuration
DATA_PATH = "./dcase"
BATCH_SIZE = 16  # Reduced for MFDWC processing
MAX_EPOCHS = 50
LEARNING_RATE = 0.0001  # Reduced for MFDWC
PRINT_EVERY_N_STEPS = 50
SAVE_MODEL_EVERY_N_EPOCHS = 10

# MFDWC Configuration (corrected implementation)
MFDWC_CONFIG = {
    'n_mels': 60,
    'n_mfdwc': 30,  # Will produce 45 features per frame (WA+ΔA+WD), 90 total (mean+std)
    'wavelet': 'haar',
    'sample_rate': 32000,
    'n_fft': 2048,
    'hop_length': 256
}

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

class AudioDataset(Dataset):
    def __init__(self, audio_files, labels, devices, label_to_idx, target_sr=32000, max_len_seconds=10, use_labels=True):
        self.audio_files = audio_files
        self.labels = labels
        self.devices = devices
        self.label_to_idx = label_to_idx
        self.target_sr = target_sr
        self.max_len_samples = target_sr * max_len_seconds
        self.use_labels = use_labels
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        label = self.labels[idx]
        device = self.devices[idx]
        
        label_idx = self.label_to_idx[label] if self.use_labels else -1
        domain = 1 if device == 'a' else 0
        
        try:
            waveform, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            
            if len(waveform) < self.max_len_samples:
                pad_len = self.max_len_samples - len(waveform)
                waveform = np.pad(waveform, (0, pad_len))
            else:
                waveform = waveform[:self.max_len_samples]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return silence if file can't be loaded
            waveform = np.zeros(self.max_len_samples)
        
        return torch.tensor(waveform, dtype=torch.float32), label_idx, domain, device

class Classifier(nn.Module):
    def __init__(self, input_size=90, num_classes=10):  # Updated for corrected MFDWC
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        
    def forward(self, h):
        return self.layer(h)

class Discriminator(nn.Module):
    def __init__(self, input_size=90, num_classes=1):  # Updated for corrected MFDWC
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, h):
        return self.layer(h)

def get_lambda(epoch, max_epoch, gamma=10.0, max_lambda=1.0):
    p = epoch / max_epoch
    return max_lambda * (2. / (1 + np.exp(-gamma * p)) - 1.)

def extract_info_from_filename(filename):
    parts = filename.replace('.wav', '').split('-')
    scene = parts[0]
    device = parts[-1]
    return scene, device

def load_audio_files(folder_path):
    files = []
    labels = []
    devices = []
    
    if not os.path.exists(folder_path):
        return files, labels, devices
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            scene, device = extract_info_from_filename(filename)
            files.append(file_path)
            labels.append(scene)
            devices.append(device)
    
    return files, labels, devices

def load_dataset(data_path):
    train_source_path = os.path.join(data_path, "train", "source")
    train_target_path = os.path.join(data_path, "train", "target")
    test_source_path = os.path.join(data_path, "test","source")
    test_target_path = os.path.join(data_path, "test","target")
    
    train_source_files, train_source_labels, train_source_devices = load_audio_files(train_source_path)
    train_target_files, train_target_labels, train_target_devices = load_audio_files(train_target_path)
    test_source_files, test_source_labels, test_source_devices = load_audio_files(test_source_path)
    test_target_files, test_target_labels, test_target_devices = load_audio_files(test_target_path)
    
    all_labels = set(train_source_labels + test_source_labels+test_target_labels)
    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"Found {len(all_labels)} classes: {sorted(all_labels)}")
    print(f"Train source (device 'a'): {len(train_source_files)} files")
    print(f"Train target (devices {sorted(set(train_target_devices))}): {len(train_target_files)} files")
    print(f"Test source (devices {sorted(set(test_source_devices))}): {len(test_source_files)} files")
    print(f"Test target (devices {sorted(set(test_target_devices))}): {len(test_target_files)} files")

    return (train_source_files, train_source_labels, train_source_devices,
            train_target_files, train_target_labels, train_target_devices,
            test_source_files, test_source_labels, test_source_devices,
            test_target_files, test_target_labels, test_target_devices,
            label_to_idx, idx_to_label)

def create_validation_split(files, labels, devices, val_ratio=0.1):
    n_val = int(len(files) * val_ratio)
    
    train_files = files[n_val:]
    train_labels = labels[n_val:]
    train_devices = devices[n_val:]
    val_files = files[:n_val]
    val_labels = labels[:n_val]
    val_devices = devices[:n_val]
    
    return train_files, train_labels, train_devices, val_files, val_labels, val_devices

def create_data_loaders(train_source_files, train_source_labels, train_source_devices,
                        train_target_files, train_target_labels, train_target_devices,
                        test_source_files, test_source_labels, test_source_devices,
                        test_target_files, test_target_labels, test_target_devices,
                        label_to_idx, batch_size=16):
    
    train_src_files, train_src_labels, train_src_devices, val_src_files, val_src_labels, val_src_devices = create_validation_split(
        train_source_files, train_source_labels, train_source_devices)
    train_tgt_files, train_tgt_labels, train_tgt_devices, val_tgt_files, val_tgt_labels, val_tgt_devices = create_validation_split(
        train_target_files, train_target_labels, train_target_devices)
    
    train_source_dataset = AudioDataset(train_src_files, train_src_labels, train_src_devices, label_to_idx, use_labels=True)
    val_source_dataset = AudioDataset(val_src_files, val_src_labels, val_src_devices, label_to_idx, use_labels=True)
    train_target_dataset = AudioDataset(train_tgt_files, train_tgt_labels, train_tgt_devices, label_to_idx, use_labels=False)
    val_target_dataset = AudioDataset(val_tgt_files, val_tgt_labels, val_tgt_devices, label_to_idx, use_labels=False)
    test_source_dataset = AudioDataset(test_source_files, test_source_labels, test_source_devices, label_to_idx, use_labels=True)
    test_target_dataset = AudioDataset(test_target_files, test_target_labels, test_target_devices, label_to_idx, use_labels=True)   
    
    train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_source_loader = DataLoader(val_source_dataset, batch_size=batch_size, shuffle=False)
    val_target_loader = DataLoader(val_target_dataset, batch_size=batch_size, shuffle=False)
    test_source_loader = DataLoader(test_source_dataset, batch_size=batch_size, shuffle=False)
    test_target_loader = DataLoader(test_target_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training - Source: {len(train_src_files)}, Target: {len(train_tgt_files)}")
    print(f"Validation - Source: {len(val_src_files)}, Target: {len(val_tgt_files)}")
    print(f"Test -Source: {len(test_source_files)}, Target: {len(test_target_files)}")
    
    return train_source_loader, train_target_loader, val_source_loader, val_target_loader, test_source_loader,test_target_loader

def load_audio_files(folder_path):
    """Load audio files from a folder"""
    files = []
    labels = []
    devices = []
    
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist")
        return files, labels, devices
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            scene, device = extract_info_from_filename(filename)
            files.append(file_path)
            labels.append(scene)
            devices.append(device)
    
    return files, labels, devices

def load_dataset(data_path):
    """Load the complete dataset"""
    train_source_path = os.path.join(data_path, "train", "source")
    train_target_path = os.path.join(data_path, "train", "target")
    test_source_path = os.path.join(data_path, "test", "source")
    test_target_path = os.path.join(data_path, "test", "target")
    
    train_source_files, train_source_labels, train_source_devices = load_audio_files(train_source_path)
    train_target_files, train_target_labels, train_target_devices = load_audio_files(train_target_path)
    test_source_files, test_source_labels, test_source_devices = load_audio_files(test_source_path)
    test_target_files, test_target_labels, test_target_devices = load_audio_files(test_target_path)
    
    all_labels = set(train_source_labels + test_source_labels + test_target_labels)
    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    print(f"Found {len(all_labels)} classes: {sorted(all_labels)}")
    print(f"Train source (device 'a'): {len(train_source_files)} files")
    print(f"Train target (devices {sorted(set(train_target_devices))}): {len(train_target_files)} files")
    print(f"Test source (devices {sorted(set(test_source_devices))}): {len(test_source_files)} files")
    print(f"Test target (devices {sorted(set(test_target_devices))}): {len(test_target_files)} files")

    return (train_source_files, train_source_labels, train_source_devices,
            train_target_files, train_target_labels, train_target_devices,
            test_source_files, test_source_labels, test_source_devices,
            test_target_files, test_target_labels, test_target_devices,
            label_to_idx, idx_to_label)

def create_validation_split(files, labels, devices, val_ratio=0.1):
    """Create validation split from training data"""
    n_val = int(len(files) * val_ratio)
    
    train_files = files[n_val:]
    train_labels = labels[n_val:]
    train_devices = devices[n_val:]
    val_files = files[:n_val]
    val_labels = labels[:n_val]
    val_devices = devices[:n_val]
    
    return train_files, train_labels, train_devices, val_files, val_labels, val_devices

def create_data_loaders(train_source_files, train_source_labels, train_source_devices,
                        train_target_files, train_target_labels, train_target_devices,
                        test_source_files, test_source_labels, test_source_devices,
                        test_target_files, test_target_labels, test_target_devices,
                        label_to_idx, batch_size=16):
    """Create data loaders for training and testing"""
    
    train_src_files, train_src_labels, train_src_devices, val_src_files, val_src_labels, val_src_devices = create_validation_split(
        train_source_files, train_source_labels, train_source_devices)
    train_tgt_files, train_tgt_labels, train_tgt_devices, val_tgt_files, val_tgt_labels, val_tgt_devices = create_validation_split(
        train_target_files, train_target_labels, train_target_devices)
    
    train_source_dataset = AudioDataset(train_src_files, train_src_labels, train_src_devices, label_to_idx, use_labels=True)
    val_source_dataset = AudioDataset(val_src_files, val_src_labels, val_src_devices, label_to_idx, use_labels=True)
    train_target_dataset = AudioDataset(train_tgt_files, train_tgt_labels, train_tgt_devices, label_to_idx, use_labels=False)
    val_target_dataset = AudioDataset(val_tgt_files, val_tgt_labels, val_tgt_devices, label_to_idx, use_labels=False)
    test_source_dataset = AudioDataset(test_source_files, test_source_labels, test_source_devices, label_to_idx, use_labels=True)
    test_target_dataset = AudioDataset(test_target_files, test_target_labels, test_target_devices, label_to_idx, use_labels=True)   
    
    train_source_loader = DataLoader(train_source_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    train_target_loader = DataLoader(train_target_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_source_loader = DataLoader(val_source_dataset, batch_size=batch_size, shuffle=False)
    val_target_loader = DataLoader(val_target_dataset, batch_size=batch_size, shuffle=False)
    test_source_loader = DataLoader(test_source_dataset, batch_size=batch_size, shuffle=False)
    test_target_loader = DataLoader(test_target_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training - Source: {len(train_src_files)}, Target: {len(train_tgt_files)}")
    print(f"Validation - Source: {len(val_src_files)}, Target: {len(val_tgt_files)}")
    print(f"Test - Source: {len(test_source_files)}, Target: {len(test_target_files)}")
    
    return train_source_loader, train_target_loader, val_source_loader, val_target_loader, test_source_loader, test_target_loader

def evaluate_model(feature_extractor, classifier, data_loader, device, return_predictions=False):
    """Evaluate the model on a dataset"""
    feature_extractor.eval()
    classifier.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (waveform, labels, domains, devices) in enumerate(data_loader):
            waveform = waveform.to(device)
            labels = labels.to(device)
            
            # Skip batches with invalid labels
            if torch.any(labels < 0):
                continue
            
            # Extract features and classify
            features = feature_extractor(waveform)
            class_outputs = classifier(features)
            
            loss = criterion(class_outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            _, predictions = torch.max(class_outputs, 1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    if return_predictions:
        return accuracy, avg_loss, all_predictions, all_labels
    return accuracy, avg_loss

def train_dann_mfdwc(feature_extractor, classifier, discriminator, grl, 
                     train_source_loader, train_target_loader,
                     val_source_loader, val_target_loader,
                     test_source_loader, test_target_loader,
                     num_epochs=50, device=DEVICE):
    """Train the DANN model with MFDWC features"""
    
    # Optimizers (MFDWC extractor has no trainable parameters)
    extractor_params = list(feature_extractor.parameters())
    if len(extractor_params) > 0:
        optimizer_F = optim.Adam(feature_extractor.parameters(), lr=LEARNING_RATE)
    else:
        optimizer_F = None  # No trainable parameters in MFDWC
        print("Note: MFDWC feature extractor has no trainable parameters (uses signal processing)")
    
    optimizer_C = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    
    # Loss functions
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCELoss()
    
    # Training history
    history = {
        'train_class_loss': [],
        'train_domain_loss': [],
        'val_source_acc': [],
        'val_target_acc': [],
        'test_source_acc': [],
        'test_target_acc': [],
        'lambda_values': []
    }
    
    print("Starting DANN training with MFDWC features...")
    print(f"Feature dimension: {MFDWC_CONFIG['n_mfdwc']} coefficients * 1.5 * 2 = {int(MFDWC_CONFIG['n_mfdwc'] * 1.5 * 2)} features")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Set lambda for GRL
        lambda_p = get_lambda(epoch, num_epochs)
        grl.set_lambda(lambda_p)
        history['lambda_values'].append(lambda_p)
        
        # Training
        feature_extractor.train()
        classifier.train()
        discriminator.train()
        
        epoch_class_loss = 0
        epoch_domain_loss = 0
        num_batches = 0
        
        # Create iterators
        source_iter = iter(train_source_loader)
        target_iter = iter(train_target_loader)
        
        max_batches = min(len(train_source_loader), len(train_target_loader))
        
        for batch_idx in range(max_batches):
            try:
                # Get source batch
                source_waveform, source_labels, source_domains, _ = next(source_iter)
                source_waveform = source_waveform.to(device)
                source_labels = source_labels.to(device)
                source_domains = source_domains.float().to(device)
                
                # Get target batch
                target_waveform, _, target_domains, _ = next(target_iter)
                target_waveform = target_waveform.to(device)
                target_domains = target_domains.float().to(device)
                
                # Forward pass - Source
                source_features = feature_extractor(source_waveform)
                source_class_outputs = classifier(source_features)
                source_domain_features = grl(source_features)
                source_domain_outputs = discriminator(source_domain_features).squeeze()
                
                # Forward pass - Target
                target_features = feature_extractor(target_waveform)
                target_domain_features = grl(target_features)
                target_domain_outputs = discriminator(target_domain_features).squeeze()
                
                # Calculate losses
                class_loss = class_criterion(source_class_outputs, source_labels)
                
                # Domain loss (source = 1, target = 0)
                domain_loss_source = domain_criterion(source_domain_outputs, source_domains)
                domain_loss_target = domain_criterion(target_domain_outputs, target_domains)
                domain_loss = domain_loss_source + domain_loss_target
                
                total_loss = class_loss + domain_loss
                
                # Backward pass
                if optimizer_F is not None:
                    optimizer_F.zero_grad()
                optimizer_C.zero_grad()
                optimizer_D.zero_grad()
                
                total_loss.backward()
                
                if optimizer_F is not None:
                    optimizer_F.step()
                optimizer_C.step()
                optimizer_D.step()
                
                epoch_class_loss += class_loss.item()
                epoch_domain_loss += domain_loss.item()
                num_batches += 1
                
                if batch_idx % PRINT_EVERY_N_STEPS == 0:
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{max_batches}, "
                          f"Class Loss: {class_loss.item():.4f}, Domain Loss: {domain_loss.item():.4f}, "
                          f"Lambda: {lambda_p:.4f}")
                
            except StopIteration:
                break
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Average losses for the epoch
        avg_class_loss = epoch_class_loss / num_batches if num_batches > 0 else 0
        avg_domain_loss = epoch_domain_loss / num_batches if num_batches > 0 else 0
        
        history['train_class_loss'].append(avg_class_loss)
        history['train_domain_loss'].append(avg_domain_loss)
        
        # Validation
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            val_source_acc, _ = evaluate_model(feature_extractor, classifier, val_source_loader, device)
            val_target_acc, _ = evaluate_model(feature_extractor, classifier, val_target_loader, device)
            test_source_acc, _ = evaluate_model(feature_extractor, classifier, test_source_loader, device)
            test_target_acc, _ = evaluate_model(feature_extractor, classifier, test_target_loader, device)
            
            history['val_source_acc'].append(val_source_acc)
            history['val_target_acc'].append(val_target_acc)
            history['test_source_acc'].append(test_source_acc)
            history['test_target_acc'].append(test_target_acc)
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.2f}s")
            print(f"  Train - Class Loss: {avg_class_loss:.4f}, Domain Loss: {avg_domain_loss:.4f}")
            print(f"  Val - Source Acc: {val_source_acc:.4f}, Target Acc: {val_target_acc:.4f}")
            print(f"  Test - Source Acc: {test_source_acc:.4f}, Target Acc: {test_target_acc:.4f}")
            print(f"  Lambda: {lambda_p:.4f}")
            print("-" * 60)
        
        # Save model periodically
        if (epoch + 1) % SAVE_MODEL_EVERY_N_EPOCHS == 0:
            save_path = f"mfdwc_dann_epoch_{epoch+1}.pth"
            save_data = {
                'epoch': epoch,
                'feature_extractor_state_dict': feature_extractor.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_C_state_dict': optimizer_C.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'history': history,
                'mfdwc_config': MFDWC_CONFIG
            }
            if optimizer_F is not None:
                save_data['optimizer_F_state_dict'] = optimizer_F.state_dict()
            
            torch.save(save_data, save_path)
            print(f"Model saved to {save_path}")
    
    return history

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training losses
    axes[0, 0].plot(history['train_class_loss'], label='Classification Loss')
    axes[0, 0].plot(history['train_domain_loss'], label='Domain Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation accuracies
    eval_epochs = list(range(0, len(history['val_source_acc'])))
    axes[0, 1].plot(eval_epochs, history['val_source_acc'], label='Source')
    axes[0, 1].plot(eval_epochs, history['val_target_acc'], label='Target')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Evaluation Step')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Test accuracies
    axes[1, 0].plot(eval_epochs, history['test_source_acc'], label='Source')
    axes[1, 0].plot(eval_epochs, history['test_target_acc'], label='Target')
    axes[1, 0].set_title('Test Accuracy')
    axes[1, 0].set_xlabel('Evaluation Step')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Lambda schedule
    axes[1, 1].plot(history['lambda_values'])
    axes[1, 1].set_title('GRL Lambda Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Lambda')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('mfdwc_dann_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function"""
    print("🚀 Starting MFDWC-DANN Training")
    print(f"MFDWC Configuration: {MFDWC_CONFIG}")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Epochs: {MAX_EPOCHS}")
    print("-" * 60)
    
    # Load dataset
    try:
        (train_src_files, train_src_labels, train_src_devices,
         train_tgt_files, train_tgt_labels, train_tgt_devices,
         test_source_files, test_source_labels, test_source_devices,
         test_target_files, test_target_labels, test_target_devices,
         label_to_idx, idx_to_label) = load_dataset(DATA_PATH)
        
        num_classes = len(label_to_idx)
        print(f"Number of classes: {num_classes}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create data loaders
    try:
        train_source_loader, train_target_loader, val_source_loader, val_target_loader, test_source_loader, test_target_loader = create_data_loaders(
            train_src_files, train_src_labels, train_src_devices,
            train_tgt_files, train_tgt_labels, train_tgt_devices,
            test_source_files, test_source_labels, test_source_devices,
            test_target_files, test_target_labels, test_target_devices,
            label_to_idx, BATCH_SIZE
        )
    except Exception as e:
        print(f"❌ Error creating data loaders: {e}")
        return
    
    # Initialize models
    try:
        feature_extractor = MFDWCFeatureExtractor(**MFDWC_CONFIG).to(DEVICE)
        
        # Calculate feature dimension
        test_audio = torch.randn(1, MFDWC_CONFIG['sample_rate'] * 10).to(DEVICE)
        with torch.no_grad():
            test_features = feature_extractor(test_audio)
            feature_dim = test_features.shape[1]
        
        print(f"✅ Feature dimension: {feature_dim}")
        
        classifier = Classifier(input_size=feature_dim, num_classes=num_classes).to(DEVICE)
        discriminator = Discriminator(input_size=feature_dim).to(DEVICE)
        grl = GradientReversalLayer().to(DEVICE)
        
        print("✅ Models initialized successfully!")
        
        # Count parameters
        total_params = sum(p.numel() for p in feature_extractor.parameters()) + \
                      sum(p.numel() for p in classifier.parameters()) + \
                      sum(p.numel() for p in discriminator.parameters())
        print(f"Total parameters: {total_params:,}")
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        return
    
    # Start training
    try:
        history = train_dann_mfdwc(
            feature_extractor, classifier, discriminator, grl,
            train_source_loader, train_target_loader,
            val_source_loader, val_target_loader,
            test_source_loader, test_target_loader,
            num_epochs=MAX_EPOCHS, device=DEVICE
        )
        
        # Plot results
        plot_training_history(history)
        
        # Final evaluation
        print("\n🎯 Final Evaluation:")
        test_source_acc, _ = evaluate_model(feature_extractor, classifier, test_source_loader, DEVICE)
        test_target_acc, _ = evaluate_model(feature_extractor, classifier, test_target_loader, DEVICE)
        
        print(f"Final Test Source Accuracy: {test_source_acc:.4f}")
        print(f"Final Test Target Accuracy: {test_target_acc:.4f}")
        print(f"Domain Gap: {abs(test_source_acc - test_target_acc):.4f}")
        
        # Save final model
        final_save_path = "mfdwc_dann_final.pth"
        torch.save({
            'feature_extractor_state_dict': feature_extractor.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'history': history,
            'mfdwc_config': MFDWC_CONFIG,
            'label_to_idx': label_to_idx,
            'idx_to_label': idx_to_label,
            'final_test_source_acc': test_source_acc,
            'final_test_target_acc': test_target_acc
        }, final_save_path)
        print(f"Final model saved to {final_save_path}")
        
        print("🎉 Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()