import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import pandas as pd
import librosa
import ipdb
import warnings
warnings.filterwarnings('ignore')
import sys
print(sys.argv)

import data
from data import data_split, label_to_numerical, SimpleAudioDataset, create_combined_loader

import models
from models import Feature_extractor as FeatureExtractor
from models import Classifier_no_weights as Classifier
from models import Discriminator_no_weights as AdversarialNetwork
from mfdwc_extractor_with_flag import MFDWCFeatureExtractor

# --- Configuration updated for GRL method with MFDWC ---
METHOD = 'GRL'
# Path to the directory where the audio subfolders are located
PATH = '/DATA/G3/Datasets/archive/Original_split/TAU-urban-acoustic-scenes-2020-mobile-development'
# Define the specific source and target devices to be used from the datasets
src_device = 'a'
tgt_device = sys.argv[1] if len(sys.argv) > 1 else 'b'
print(f"Source Device: {src_device}, Target Device: {tgt_device}")
TARGET_SAMPLE_RATE = 44100  # Changed to 44100 for MFDWC
NUM_CLASS = 10
USE_GPU = True
BATCH_SIZE = 128  # Reduced for MFDWC processing
NUM_EPOCHS = 200
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = './results/'
os.makedirs(save_dir, exist_ok=True)

# MFDWC Configuration
MFDWC_N_MELS = 60
MFDWC_WAVELET = 'haar'


# --- Utility and Loss Functions ---
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    """Calculates the coefficient for GRL."""
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


# --- Gradient Reversal Layer ---
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


# --- Train and Test Functions ---
def test(mfdwc_extractor, feature_extractor, classifier, dataloader, device):
    """Evaluation loop."""
    mfdwc_extractor.eval()
    feature_extractor.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            # Skip batches where data loading failed
            valid_indices = [i for i, label in enumerate(target) if label != "error"]
            if not valid_indices:
                continue
            
            data = data[valid_indices]
            target = [target[i] for i in valid_indices]

            target = label_to_numerical(target)
            data, target = data.to(device), target.to(device)
            
            mfdwc_features = mfdwc_extractor(data)  # (batch, 1, 90, n_frames)
            features = feature_extractor(mfdwc_features)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    if total == 0:
        return 0.0
    accuracy = 100 * correct / total
    return accuracy


def check_gradient_norm(model):
    """Calculates the total L2 norm of gradients for a given model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def train():
    """Main training and evaluation function."""
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data loading ---
    src_devices_mixup, target_devices_mixup, label_mix_up, train_src_df, train_tgt_df, test_src_df, test_tgt_df = data_split(src_device, tgt_device, data_path=PATH)

    print(f"Number of classes detected: {NUM_CLASS}")
    
    src_dataset = SimpleAudioDataset(file_df=train_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    tgt_dataset = SimpleAudioDataset(file_df=train_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_loader = DataLoader(tgt_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    src_test_dataset = SimpleAudioDataset(file_df=test_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_test_loader = DataLoader(src_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    tgt_test_dataset = SimpleAudioDataset(file_df=test_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize MFDWC feature extractor
    mfdwc_extractor = MFDWCFeatureExtractor(
        n_mels=MFDWC_N_MELS,
        n_fft=2048,
        hop_length=256,
        wavelet=MFDWC_WAVELET,
        sample_rate=TARGET_SAMPLE_RATE,
        return_temporal=True  # Return (batch, 1, 90, n_frames) for CNN
    ).to(device)
    
    # Calculate correct flattened size
    with torch.no_grad():
        # Create dummy input: 10 seconds of audio
        dummy_audio = torch.randn(1, TARGET_SAMPLE_RATE * 10).to(device)
        dummy_mfdwc = mfdwc_extractor(dummy_audio)  # (1, 1, 90, n_frames)
        feature_extractor_temp = FeatureExtractor().to(device)
        dummy_features = feature_extractor_temp(dummy_mfdwc)
        flattened_size = dummy_features.shape[1]
        del feature_extractor_temp, dummy_audio, dummy_mfdwc, dummy_features
    
    print(f"Calculated flattened size: {flattened_size}")

    # Models for CNN backbone
    feature_extractor = FeatureExtractor().to(device)
    classifier = Classifier(flattened_size=flattened_size).to(device)
    discriminator = AdversarialNetwork(flattened_size=flattened_size).to(device)

    # Optimizers
    F_opt = optim.Adam(feature_extractor.parameters(), lr=LR)
    C_opt = optim.Adam(classifier.parameters(), lr=LR)
    D_opt = optim.Adam(discriminator.parameters(), lr=LR)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_adv = nn.BCEWithLogitsLoss()
    
    # Setup for logging results
    results_log = []

    # Track best models
    best_target_acc = 0.0
    best_epoch = 0

    # Create subdirectory for this experiment
    experiment_dir = os.path.join(save_dir, f"grl-mfdwc-{src_device}-{tgt_device}")
    os.makedirs(experiment_dir, exist_ok=True)

    max_iter = NUM_EPOCHS * min(len(src_loader), len(tgt_loader))
    iter_num = 0

    print("Starting training with GRL method and MFDWC features...")
    for epoch in range(1, NUM_EPOCHS + 1):
        mfdwc_extractor.eval()  # MFDWC is frozen
        feature_extractor.train()
        classifier.train()
        discriminator.train()

        total_cls_loss, total_adv_loss = 0, 0
        total_f_grad_norm, total_c_grad_norm, total_d_grad_norm = 0, 0, 0
        
        # Count actual batches processed in this epoch
        num_batches = 0
        
        for batch_idx, (src_data, tgt_data) in enumerate(create_combined_loader(src_loader, tgt_loader)):
            
            src_wave_form, src_label = src_data
            tgt_wave_form, _ = tgt_data  # Target labels are not used in unsupervised DA

            if "error" in src_label:
                print(f"Skipping batch {batch_idx+1} due to data loading error in source.")
                continue

            src_label = label_to_numerical(src_label)
            src = src_wave_form.to(DEVICE)
            labels = src_label.to(DEVICE)
            tgt = tgt_wave_form.to(DEVICE)
    
            current_batch_size = min(len(src), len(tgt))
            if current_batch_size == 0:
                continue

            src, labels, tgt = src[:current_batch_size], labels[:current_batch_size], tgt[:current_batch_size]
            
            print(f"\rEpoch {epoch}, Batch {batch_idx+1}", end="", flush=True)
            
            F_opt.zero_grad()
            C_opt.zero_grad()
            D_opt.zero_grad()

            # Extract MFDWC features (no gradients needed)
            with torch.no_grad():
                src_mfdwc = mfdwc_extractor(src)  # (batch, 1, 90, n_frames)
                tgt_mfdwc = mfdwc_extractor(tgt)  # (batch, 1, 90, n_frames)

            # --- FORWARD PASS ---
            feat_source = feature_extractor(src_mfdwc)
            pred_source = classifier(feat_source)
            feat_target = feature_extractor(tgt_mfdwc)

            # 1. Classification Loss (on source data)
            cls_loss = criterion_cls(pred_source, labels)

            # --- GRL Adversarial Loss Calculation ---
            # Combine features and apply Gradient Reversal Layer
            features_combined = torch.cat((feat_source, feat_target), dim=0)
            coeff = calc_coeff(iter_num, max_iter=max_iter)
            reversed_features = GradReverse.apply(features_combined, coeff)
            
            # Get domain predictions from the discriminator
            domain_preds = discriminator(reversed_features)

            # Create domain labels: 1 for source, 0 for target
            domain_labels_src = torch.ones(current_batch_size, 1, device=device)
            domain_labels_tgt = torch.zeros(current_batch_size, 1, device=device)
            domain_labels = torch.cat((domain_labels_src, domain_labels_tgt), dim=0)

            # 2. Adversarial Loss (to fool the discriminator)
            adv_loss = criterion_adv(domain_preds, domain_labels)
            
            # --- Total Loss and Backward Pass ---
            total_loss = cls_loss + adv_loss
            total_loss.backward()

            # Gradient norm checking
            total_f_grad_norm += check_gradient_norm(feature_extractor)
            total_c_grad_norm += check_gradient_norm(classifier)
            total_d_grad_norm += check_gradient_norm(discriminator)

            # Update weights
            F_opt.step()
            C_opt.step()
            D_opt.step()

            iter_num += 1
            total_cls_loss += cls_loss.item()
            total_adv_loss += adv_loss.item()
            num_batches += 1
        
        # --- Epoch-end logging and evaluation ---
        if num_batches == 0:
            print(f"\nEpoch [{epoch}/{NUM_EPOCHS}] - No valid batches processed!")
            continue
            
        avg_cls_loss = total_cls_loss / num_batches
        avg_adv_loss = total_adv_loss / num_batches
        
        avg_f_grad = total_f_grad_norm / num_batches
        avg_c_grad = total_c_grad_norm / num_batches
        avg_d_grad = total_d_grad_norm / num_batches

        source_acc = test(mfdwc_extractor, feature_extractor, classifier, src_test_loader, device)
        target_acc = test(mfdwc_extractor, feature_extractor, classifier, tgt_test_loader, device)

        print(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}] ({num_batches} batches) | "
            f"Cls Loss: {avg_cls_loss:.4f}, Adv Loss: {avg_adv_loss:.4f} | "
            f"Src Acc: {source_acc:.2f}%, Tgt Acc: {target_acc:.2f}%"
        )
        print(
            f"  -> Avg Grad Norms | Feature Extractor: {avg_f_grad:.4f}, "
            f"Classifier: {avg_c_grad:.4f}, Discriminator: {avg_d_grad:.4f}\n"
        )

        # Append results to log
        epoch_results = {
            'epoch': epoch,
            'classification_loss': avg_cls_loss,
            'adversarial_loss': avg_adv_loss,
            'source_accuracy': source_acc,
            'target_accuracy': target_acc,
            'feature_extractor_grad_norm': avg_f_grad,
            'classifier_grad_norm': avg_c_grad,
            'discriminator_grad_norm': avg_d_grad
        }
        results_log.append(epoch_results)

        # Save best model based on target accuracy
        if target_acc > best_target_acc:
            best_target_acc = target_acc
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': feature_extractor.state_dict(),
                'target_accuracy': target_acc,
                'source_accuracy': source_acc
            }, os.path.join(experiment_dir, f"best_FE_{src_device}-{tgt_device}.pth"))
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'target_accuracy': target_acc,
                'source_accuracy': source_acc
            }, os.path.join(experiment_dir, f"best_CL_{src_device}-{tgt_device}.pth"))
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'target_accuracy': target_acc,
                'source_accuracy': source_acc
            }, os.path.join(experiment_dir, f"best_D_{src_device}-{tgt_device}.pth"))
            
            print(f"  -> Saved best model with target accuracy: {target_acc:.2f}%")
        
        # Always save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': feature_extractor.state_dict(),
            'target_accuracy': target_acc,
            'source_accuracy': source_acc
        }, os.path.join(experiment_dir, f"latest_FE_{src_device}-{tgt_device}.pth"))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'target_accuracy': target_acc,
            'source_accuracy': source_acc
        }, os.path.join(experiment_dir, f"latest_CL_{src_device}-{tgt_device}.pth"))

        # Save CSV every epoch
        csv_path = os.path.join(experiment_dir, f"training_results_grl_mfdwc_{src_device}-{tgt_device}.csv")
        pd.DataFrame(results_log).to_csv(csv_path, index=False)

    print(f"\nTraining finished!")
    print(f"Best target accuracy: {best_target_acc:.2f}% at epoch {best_epoch}")
    print(f"Results saved to {experiment_dir}")


if __name__ == '__main__':
    train()