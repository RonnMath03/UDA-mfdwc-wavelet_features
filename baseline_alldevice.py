import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import pandas as pd
import librosa
import data 
from data import data_split , label_to_numerical, SimpleAudioDataset , create_combined_loader
from datetime import datetime

import models
from models import Feature_extractor as FeatureExtractor 
from models import Classifier_no_weights as Classifier 
from mfdwc_extractor_with_flag import MFDWCFeatureExtractor

# --- Configuration and Hyperparameters ---
METHOD = 'cnn'
PATH = '/DATA/G3/Datasets/archive/Original_split/TAU-urban-acoustic-scenes-2020-mobile-development'
output_csv_path = 'training_results_dcase_cnn.csv'

src_device = 'a'

# All target devices for evaluation
ALL_TARGET_DEVICES = ['b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']

TARGET_SAMPLE_RATE = 44100
NUM_CLASS = 10
USE_GPU = True
BATCH_SIZE = 128
NUM_EPOCHS = 200
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = './results/'
os.makedirs(save_dir, exist_ok=True)
import warnings
warnings.filterwarnings("ignore")

MFDWC_N_MELS = 60
MFDWC_WAVELET = 'haar'


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
            valid_indices = [i for i, label in enumerate(target) if label != "error"]
            if not valid_indices:
                continue
            
            data = data[valid_indices]
            target = [target[i] for i in valid_indices]

            target = label_to_numerical(target)
            data, target = data.to(device), target.to(device)

            mfdwc_features = mfdwc_extractor(data)
            
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


def collect_target_test_loaders(tgt_devices, src_device, data_path, target_sr, batch_size):
    """
    Collect test loaders for all target devices (for evaluation only).
    Source-only baseline doesn't use target data during training.
    """
    test_tgt_loaders = {}
    
    for tgt_dev in tgt_devices:
        _, _, _, _, _, _, test_tgt_df = data_split(
            src_device, tgt_dev, data_path=data_path
        )
        
        tgt_test_ds = SimpleAudioDataset(
            file_df=test_tgt_df[tgt_dev], root=data_path, target_sr=target_sr
        )
        test_tgt_loaders[tgt_dev] = DataLoader(
            tgt_test_ds, batch_size=batch_size, shuffle=False, num_workers=4
        )
        print(f"  Target device '{tgt_dev}' test set size: {len(tgt_test_ds)}")
    
    return test_tgt_loaders


def train():
    """Main training and evaluation function (source-only baseline)."""
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Source Device: {src_device}")
    print(f"Target Devices (eval only): {ALL_TARGET_DEVICES}")

    # --- Data loading (source only for training) ---
    _, _, _, train_src_df, _, test_src_df, _ = data_split(
        src_device, ALL_TARGET_DEVICES[0], data_path=PATH
    )

    print(f"Number of classes detected: {NUM_CLASS}")
    
    src_dataset = SimpleAudioDataset(
        file_df=train_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE
    )
    src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print(f"Source training set size: {len(src_dataset)}")

    src_test_dataset = SimpleAudioDataset(
        file_df=test_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE
    )
    src_test_loader = DataLoader(src_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Collect test loaders for ALL target devices (evaluation only)
    print("Loading target device test sets...")
    per_device_test_loaders = collect_target_test_loaders(
        ALL_TARGET_DEVICES, src_device, PATH, TARGET_SAMPLE_RATE, BATCH_SIZE
    )

    # Initialize MFDWC feature extractor
    mfdwc_extractor = MFDWCFeatureExtractor(
        n_mels=MFDWC_N_MELS,
        n_fft=2048,
        hop_length=256,
        wavelet=MFDWC_WAVELET,
        sample_rate=TARGET_SAMPLE_RATE,
        return_temporal=True
    ).to(device)
    
    # Calculate correct flattened size
    with torch.no_grad():
        dummy_audio = torch.randn(1, TARGET_SAMPLE_RATE * 10).to(device)
        dummy_mfdwc = mfdwc_extractor(dummy_audio)
        feature_extractor_temp = FeatureExtractor().to(device)
        dummy_features = feature_extractor_temp(dummy_mfdwc)
        flattened_size = dummy_features.shape[1]
        del feature_extractor_temp, dummy_audio, dummy_mfdwc, dummy_features
    
    print(f"Calculated flattened size: {flattened_size}")

    # Models
    feature_extractor = FeatureExtractor().to(device)
    classifier = Classifier(flattened_size=flattened_size).to(device)

    # Optimizers (no discriminator for source-only baseline)
    F_opt = optim.Adam(feature_extractor.parameters(), lr=LR)
    C_opt = torch.optim.Adam(classifier.parameters(), lr=LR)

    criterion_cls = nn.CrossEntropyLoss()
    
    results_log = []
    best_avg_target_acc = 0.0
    best_epoch = 0

    tgt_label = '+'.join(ALL_TARGET_DEVICES)
    experiment_dir = os.path.join(save_dir, f"source-only-{src_device}-{tgt_label}")
    os.makedirs(experiment_dir, exist_ok=True)

    iter_num = 0

    print(f"\nStarting source-only baseline training...")
    print(f"Training on source '{src_device}', evaluating on {ALL_TARGET_DEVICES}\n")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        mfdwc_extractor.eval()
        feature_extractor.train()
        classifier.train()

        total_cls_loss = 0
        total_f_grad_norm, total_c_grad_norm = 0, 0
        num_batches = 0
        
        # Source-only training: iterate over source data only
        for batch_idx, (src_wave_form, src_label) in enumerate(src_loader):
            
            # Skip batch if data loading failed
            valid_indices = [i for i, label in enumerate(src_label) if label != "error"]
            if not valid_indices:
                print(f"Skipping batch {batch_idx+1} due to data loading error in source.")
                continue

            src_wave_form = src_wave_form[valid_indices]
            src_label = [src_label[i] for i in valid_indices]

            src_label = label_to_numerical(src_label)
            src = src_wave_form.to(DEVICE)
            labels = src_label.to(DEVICE)
            
            if len(src) == 0:
                continue

            print(f"\rEpoch {epoch}, Batch {batch_idx+1}/{len(src_loader)}", end="", flush=True)
            
            F_opt.zero_grad()
            C_opt.zero_grad()

            with torch.no_grad():
                src_mfdwc = mfdwc_extractor(src)

            feat_source = feature_extractor(src_mfdwc)
            pred_source = classifier(feat_source)

            cls_loss = criterion_cls(pred_source, labels)
            cls_loss.backward()

            total_f_grad_norm += check_gradient_norm(feature_extractor)
            total_c_grad_norm += check_gradient_norm(classifier)

            F_opt.step()
            C_opt.step()

            iter_num += 1
            total_cls_loss += cls_loss.item()
            num_batches += 1

        # --- Epoch-end logging and evaluation ---
        if num_batches == 0:
            print(f"\nEpoch [{epoch}/{NUM_EPOCHS}] - No valid batches processed!")
            continue
            
        avg_cls_loss = total_cls_loss / num_batches
        avg_f_grad = total_f_grad_norm / num_batches
        avg_c_grad = total_c_grad_norm / num_batches

        source_acc = test(mfdwc_extractor, feature_extractor, classifier, src_test_loader, device)
        
        # Evaluate on EACH target device individually
        target_accs = {}
        for tgt_dev in ALL_TARGET_DEVICES:
            tgt_acc = test(
                mfdwc_extractor, feature_extractor, classifier,
                per_device_test_loaders[tgt_dev], device
            )
            target_accs[tgt_dev] = tgt_acc
        
        avg_target_acc = np.mean(list(target_accs.values()))

        # Print per-device results
        tgt_acc_str = ", ".join([f"{dev}: {acc:.2f}%" for dev, acc in target_accs.items()])
        print(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}] ({num_batches} batches) | "
            f"Cls Loss: {avg_cls_loss:.4f} | "
            f"Src Acc: {source_acc:.2f}%"
        )
        print(f"  -> Target Accs | {tgt_acc_str}")
        print(f"  -> Avg Target Acc: {avg_target_acc:.2f}%")
        print(
            f"  -> Avg Grad Norms | Feature Extractor: {avg_f_grad:.4f}, "
            f"Classifier: {avg_c_grad:.4f}\n"
        )

        # Build epoch results dict with per-device accuracies
        epoch_results = {
            'epoch': epoch,
            'classification_loss': avg_cls_loss,
            'source_accuracy': source_acc,
            'avg_target_accuracy': avg_target_acc,
            'feature_extractor_grad_norm': avg_f_grad,
            'classifier_grad_norm': avg_c_grad,
        }
        for tgt_dev in ALL_TARGET_DEVICES:
            epoch_results[f'target_accuracy_{tgt_dev}'] = target_accs[tgt_dev]
        
        results_log.append(epoch_results)
        
        # Save best model based on average target accuracy
        if avg_target_acc > best_avg_target_acc:
            best_avg_target_acc = avg_target_acc
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': feature_extractor.state_dict(),
                'avg_target_accuracy': avg_target_acc,
                'per_device_accuracy': target_accs,
                'source_accuracy': source_acc
            }, os.path.join(experiment_dir, f"best_FE_{src_device}-{tgt_label}.pth"))
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'avg_target_accuracy': avg_target_acc,
                'per_device_accuracy': target_accs,
                'source_accuracy': source_acc
            }, os.path.join(experiment_dir, f"best_CL_{src_device}-{tgt_label}.pth"))
            
            print(f"  -> Saved best model with avg target accuracy: {avg_target_acc:.2f}%")
        
        # Always save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': feature_extractor.state_dict(),
            'avg_target_accuracy': avg_target_acc,
            'per_device_accuracy': target_accs,
            'source_accuracy': source_acc
        }, os.path.join(experiment_dir, f"latest_FE_{src_device}-{tgt_label}.pth"))
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': classifier.state_dict(),
            'avg_target_accuracy': avg_target_acc,
            'per_device_accuracy': target_accs,
            'source_accuracy': source_acc
        }, os.path.join(experiment_dir, f"latest_CL_{src_device}-{tgt_label}.pth"))

        # Save CSV every epoch
        csv_path = os.path.join(experiment_dir, f"training_results_mfdwc-dcase-cnn_{src_device}-{tgt_label}.csv")
        pd.DataFrame(results_log).to_csv(csv_path, index=False)

    print(f"\nTraining finished!")
    print(f"Best avg target accuracy: {best_avg_target_acc:.2f}% at epoch {best_epoch}")
    print(f"Results saved to {experiment_dir}")

if __name__ == '__main__':
    train()