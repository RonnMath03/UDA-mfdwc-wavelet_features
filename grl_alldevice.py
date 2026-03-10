import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.autograd import Function
import pandas as pd
import librosa
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
from training_timer import TrainingTimer

# --- Configuration updated for GRL method with MFDWC ---
METHOD = 'GRL'
PATH = '/DATA/G3/Datasets/archive/Original_split/TAU-urban-acoustic-scenes-2020-mobile-development'
src_device = 'a'

# All target devices combined
ALL_TARGET_DEVICES = ['b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']
# Allow override from command line: pass 'all' or specific devices comma-separated
if len(sys.argv) > 1:
    if sys.argv[1] == 'all':
        tgt_devices = ALL_TARGET_DEVICES
    else:
        tgt_devices = sys.argv[1].split(',')
else:
    tgt_devices = ALL_TARGET_DEVICES

print(f"Source Device: {src_device}, Target Devices: {tgt_devices}")
TARGET_SAMPLE_RATE = 44100
NUM_CLASS = 10
USE_GPU = True
BATCH_SIZE = 128
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


def collect_target_data(tgt_devices, src_device, data_path, target_sr, batch_size):
    """
    Collect train and test datasets/loaders for all target devices.
    Returns a single combined train loader and per-device test loaders.
    """
    train_tgt_datasets = []
    test_tgt_loaders = {}
    
    for tgt_dev in tgt_devices:
        _, _, _, _, train_tgt_df, _, test_tgt_df = data_split(
            src_device, tgt_dev, data_path=data_path
        )
        
        # Accumulate training target datasets
        tgt_train_ds = SimpleAudioDataset(
            file_df=train_tgt_df[tgt_dev], root=data_path, target_sr=target_sr
        )
        train_tgt_datasets.append(tgt_train_ds)
        
        # Per-device test loaders for individual evaluation
        tgt_test_ds = SimpleAudioDataset(
            file_df=test_tgt_df[tgt_dev], root=data_path, target_sr=target_sr
        )
        test_tgt_loaders[tgt_dev] = DataLoader(
            tgt_test_ds, batch_size=batch_size, shuffle=False, num_workers=4
        )
    
    # Combine all target training data into one dataset
    combined_tgt_train_dataset = ConcatDataset(train_tgt_datasets)
    combined_tgt_train_loader = DataLoader(
        combined_tgt_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    print(f"Combined target training set size: {len(combined_tgt_train_dataset)} "
          f"(from {len(tgt_devices)} devices)")
    
    return combined_tgt_train_loader, test_tgt_loaders


def train():
    """Main training and evaluation function."""
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data loading ---
    # We still need source data from one call to data_split
    _, _, _, train_src_df, _, test_src_df, _ = data_split(
        src_device, tgt_devices[0], data_path=PATH
    )

    print(f"Number of classes detected: {NUM_CLASS}")
    
    src_dataset = SimpleAudioDataset(
        file_df=train_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE
    )
    src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    src_test_dataset = SimpleAudioDataset(
        file_df=test_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE
    )
    src_test_loader = DataLoader(src_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Collect ALL target devices into one combined loader + per-device test loaders
    combined_tgt_loader, per_device_test_loaders = collect_target_data(
        tgt_devices, src_device, PATH, TARGET_SAMPLE_RATE, BATCH_SIZE
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
    discriminator = AdversarialNetwork(flattened_size=flattened_size).to(device)

    # Optimizers
    F_opt = optim.Adam(feature_extractor.parameters(), lr=LR)
    C_opt = optim.Adam(classifier.parameters(), lr=LR)
    D_opt = optim.Adam(discriminator.parameters(), lr=LR)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_adv = nn.BCEWithLogitsLoss()
    
    results_log = []
    best_avg_target_acc = 0.0
    best_epoch = 0

    tgt_label = '+'.join(tgt_devices)
    experiment_dir = os.path.join(save_dir, f"grl-mfdwc-{src_device}-{tgt_label}")
    os.makedirs(experiment_dir, exist_ok=True)

    max_iter = NUM_EPOCHS * min(len(src_loader), len(combined_tgt_loader))
    iter_num = 0

    timer = TrainingTimer(NUM_EPOCHS)
    print(f"Starting training with GRL method and MFDWC features...")
    print(f"Adapting {src_device} -> [{tgt_label}] (all targets combined)")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        timer.start_epoch()
        mfdwc_extractor.eval()
        feature_extractor.train()
        classifier.train()
        discriminator.train()

        total_cls_loss, total_adv_loss = 0, 0
        total_f_grad_norm, total_c_grad_norm, total_d_grad_norm = 0, 0, 0
        num_batches = 0
        
        for batch_idx, (src_data, tgt_data) in enumerate(
            create_combined_loader(src_loader, combined_tgt_loader)
        ):
            src_wave_form, src_label = src_data
            tgt_wave_form, _ = tgt_data

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

            with torch.no_grad():
                src_mfdwc = mfdwc_extractor(src)
                tgt_mfdwc = mfdwc_extractor(tgt)

            feat_source = feature_extractor(src_mfdwc)
            pred_source = classifier(feat_source)
            feat_target = feature_extractor(tgt_mfdwc)

            cls_loss = criterion_cls(pred_source, labels)

            features_combined = torch.cat((feat_source, feat_target), dim=0)
            coeff = calc_coeff(iter_num, max_iter=max_iter)
            reversed_features = GradReverse.apply(features_combined, coeff)
            
            domain_preds = discriminator(reversed_features)

            domain_labels_src = torch.ones(current_batch_size, 1, device=device)
            domain_labels_tgt = torch.zeros(current_batch_size, 1, device=device)
            domain_labels = torch.cat((domain_labels_src, domain_labels_tgt), dim=0)

            adv_loss = criterion_adv(domain_preds, domain_labels)
            
            total_loss = cls_loss + adv_loss
            total_loss.backward()

            total_f_grad_norm += check_gradient_norm(feature_extractor)
            total_c_grad_norm += check_gradient_norm(classifier)
            total_d_grad_norm += check_gradient_norm(discriminator)

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
        
        # Evaluate on EACH target device individually
        target_accs = {}
        for tgt_dev in tgt_devices:
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
            f"Cls Loss: {avg_cls_loss:.4f}, Adv Loss: {avg_adv_loss:.4f} | "
            f"Src Acc: {source_acc:.2f}%"
        )
        print(f"  -> Target Accs | {tgt_acc_str}")
        print(f"  -> Avg Target Acc: {avg_target_acc:.2f}%")
        print(
            f"  -> Avg Grad Norms | Feature Extractor: {avg_f_grad:.4f}, "
            f"Classifier: {avg_c_grad:.4f}, Discriminator: {avg_d_grad:.4f}\n"
        )

        # Build epoch results dict with per-device accuracies
        epoch_results = {
            'epoch': epoch,
            'classification_loss': avg_cls_loss,
            'adversarial_loss': avg_adv_loss,
            'source_accuracy': source_acc,
            'avg_target_accuracy': avg_target_acc,
            'feature_extractor_grad_norm': avg_f_grad,
            'classifier_grad_norm': avg_c_grad,
            'discriminator_grad_norm': avg_d_grad
        }
        for tgt_dev in tgt_devices:
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
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': discriminator.state_dict(),
                'avg_target_accuracy': avg_target_acc,
                'per_device_accuracy': target_accs,
                'source_accuracy': source_acc
            }, os.path.join(experiment_dir, f"best_D_{src_device}-{tgt_label}.pth"))
            
            print(f"  -> Saved best model with avg target accuracy: {avg_target_acc:.2f}%")
        
        # Always save latest
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

        csv_path = os.path.join(experiment_dir, f"training_results_grl_mfdwc_{src_device}-{tgt_label}.csv")
        pd.DataFrame(results_log).to_csv(csv_path, index=False)

        timer.end_epoch(epoch, NUM_EPOCHS)

    timer.summary()
    print(f"\nTraining finished!")
    print(f"Best avg target accuracy: {best_avg_target_acc:.2f}% at epoch {best_epoch}")
    print(f"Results saved to {experiment_dir}")


if __name__ == '__main__':
    train()