import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import sys
print(sys.argv)

import data
from data import data_split, label_to_numerical, SimpleAudioDataset, create_combined_loader

import models
from models import Feature_extractor as FeatureExtractor
from models import CST_Classifier
from mfdwc_extractor_with_flag import MFDWCFeatureExtractor


# ==============================================================================
# Configuration
# ==============================================================================
METHOD = 'CST'
PATH = '/DATA/G3/Datasets/archive/Original_split/TAU-urban-acoustic-scenes-2020-mobile-development'
src_device = 'a'
tgt_device = sys.argv[1] if len(sys.argv) > 1 else 'b'
print(f"Source Device: {src_device}, Target Device: {tgt_device}")

TARGET_SAMPLE_RATE = 44100
NUM_CLASS = 10
USE_GPU = True
BATCH_SIZE = 64  # Smaller than GRL (128) due to matrix inversion in CST
NUM_EPOCHS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = './results/'
os.makedirs(save_dir, exist_ok=True)

# MFDWC Configuration
MFDWC_N_MELS = 60
MFDWC_WAVELET = 'haar'

# CST-specific hyperparameters
TEMPERATURE = 2.0
ALPHA = 1.9
TRADE_OFF = 0.08        # Weight for Tsallis entropy transfer loss
TRADE_OFF1 = 0.5        # Weight for CST reverse (kernel regression) loss
TRADE_OFF3 = 0.5        # Weight for FixMatch pseudo-label loss
THRESHOLD = 0.97        # Confidence threshold for pseudo-labels
BOTTLENECK_DIM = 256

# SAM + SGD optimizer settings
LR = 0.001
LR_GAMMA = 0.001
LR_DECAY = 0.75
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-3
RHO = 0.5              # SAM perturbation radius

# SpecAugment settings for strong augmentation on MFDWC features
FREQ_MASK_WIDTH = 10    # Max frequency bands to mask (out of 90)
TIME_MASK_WIDTH = 20    # Max time frames to mask


# ==============================================================================
# SAM Optimizer (Sharpness-Aware Minimization)
# ==============================================================================
class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer.
    
    Two-step optimization: first_step perturbs weights to local maximum,
    second_step restores original weights and applies the sharpness-aware update.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# ==============================================================================
# Tsallis Entropy Loss
# ==============================================================================
def entropy(predictions, reduction='none'):
    """Shannon entropy of softmax predictions."""
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    return H


class TsallisEntropy(nn.Module):
    """Entropy-weighted Tsallis divergence for domain transfer loss."""
    def __init__(self, temperature, alpha):
        super(TsallisEntropy, self).__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(self, logits):
        N, C = logits.shape
        pred = F.softmax(logits / self.temperature, dim=1)
        entropy_weight = entropy(pred).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (N * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)
        sum_dim = torch.sum(pred * entropy_weight, dim=0).unsqueeze(dim=0)
        return 1 / (self.alpha - 1) * torch.sum(
            (1 / torch.mean(sum_dim) - torch.sum(pred ** self.alpha / sum_dim * entropy_weight, dim=-1))
        )


# ==============================================================================
# SpecAugment — Strong augmentation on MFDWC feature maps
# ==============================================================================
def spec_augment(features, freq_mask_width=FREQ_MASK_WIDTH, time_mask_width=TIME_MASK_WIDTH):
    """Apply SpecAugment-style masking to MFDWC feature maps.
    
    Args:
        features: (batch, 1, n_features, n_frames) MFDWC tensor
        freq_mask_width: Max number of contiguous frequency bands to zero out
        time_mask_width: Max number of contiguous time frames to zero out
    
    Returns:
        Masked features (same shape), not in-place
    """
    augmented = features.clone()
    _, _, n_freq, n_time = augmented.shape

    # Frequency masking
    if freq_mask_width > 0 and n_freq > freq_mask_width:
        f = torch.randint(0, freq_mask_width + 1, (1,)).item()
        f0 = torch.randint(0, n_freq - f + 1, (1,)).item()
        augmented[:, :, f0:f0 + f, :] = 0.0

    # Time masking
    if time_mask_width > 0 and n_time > time_mask_width:
        t = torch.randint(0, time_mask_width + 1, (1,)).item()
        t0 = torch.randint(0, n_time - t + 1, (1,)).item()
        augmented[:, :, :, t0:t0 + t] = 0.0

    return augmented


# ==============================================================================
# Utility Functions
# ==============================================================================
def check_gradient_norm(model):
    """Calculates the total L2 norm of gradients for a given model."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


# ==============================================================================
# Test / Evaluation
# ==============================================================================
def test(mfdwc_extractor, feature_extractor, cst_classifier, dataloader, device):
    """Evaluation loop for CST model."""
    mfdwc_extractor.eval()
    feature_extractor.eval()
    cst_classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for audio, target in dataloader:
            valid_indices = [i for i, label in enumerate(target) if label != "error"]
            if not valid_indices:
                continue

            audio = audio[valid_indices]
            target = [target[i] for i in valid_indices]

            target = label_to_numerical(target)
            audio, target = audio.to(device), target.to(device)

            mfdwc_features = mfdwc_extractor(audio)
            features = feature_extractor(mfdwc_features)
            logits, _ = cst_classifier(features)
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    if total == 0:
        return 0.0
    return 100 * correct / total


# ==============================================================================
# CST Forward Pass + Loss Computation
# ==============================================================================
def compute_cst_losses(feature_extractor, cst_classifier, src_mfdwc, tgt_mfdwc,
                       tgt_mfdwc_strong, labels, ts_loss, batch_size, device):
    """Compute all CST loss components in a single forward pass.
    
    Returns:
        total_loss, cls_loss, transfer_loss, reverse_loss, Lu (all scalar tensors)
    """
    # Forward: source + weak target through shared backbone
    x = torch.cat((src_mfdwc, tgt_mfdwc), dim=0)
    feat = feature_extractor(x)
    y, f = cst_classifier(feat)

    # Forward: strong-augmented target
    feat_strong = feature_extractor(tgt_mfdwc_strong)
    y_t_u, _ = cst_classifier(feat_strong)

    # Split source / target outputs
    y_s, y_t = y.chunk(2, dim=0)
    f_s, f_t = f.chunk(2, dim=0)

    # 1. Classification loss (source only)
    cls_loss = F.cross_entropy(y_s, labels)

    # 2. Transfer loss: Tsallis entropy on target logits
    transfer_loss = ts_loss(y_t)

    # 3. FixMatch pseudo-label loss
    max_prob, pred_u = torch.max(F.softmax(y_t, dim=-1), dim=-1)
    Lu = (F.cross_entropy(y_t_u, pred_u, reduction='none') *
          max_prob.ge(THRESHOLD).float().detach()).mean()

    # 4. CST kernel regression (cycle self-training reverse loss)
    # Normalize embeddings
    f_t_norm = f_t / (torch.norm(f_t, dim=-1, keepdim=True) + 1e-8)
    f_s_norm = f_s / (torch.norm(f_s, dim=-1, keepdim=True) + 1e-8)

    # Gram matrices
    target_kernel = torch.clamp(f_t_norm.mm(f_t_norm.t()), -0.99999999, 0.99999999)
    test_kernel = torch.clamp(f_s_norm.mm(f_t_norm.t()), -0.99999999, 0.99999999)

    # Kernel regression: predict source labels from target pseudo-labels
    target_labels_oh = F.one_hot(pred_u, NUM_CLASS).float() - 1.0 / float(NUM_CLASS)
    source_labels_oh = F.one_hot(labels, NUM_CLASS).float() - 1.0 / float(NUM_CLASS)
    regularizer = 0.001 * torch.eye(batch_size, device=device)
    predicted_source = test_kernel.mm(torch.inverse(target_kernel + regularizer)).mm(target_labels_oh)
    reverse_loss = F.mse_loss(predicted_source, source_labels_oh)

    # Total loss
    if Lu.item() != 0:
        total_loss = cls_loss + TRADE_OFF * transfer_loss + TRADE_OFF1 * reverse_loss + TRADE_OFF3 * Lu
    else:
        total_loss = cls_loss + TRADE_OFF * transfer_loss + TRADE_OFF1 * reverse_loss

    return total_loss, cls_loss, transfer_loss, reverse_loss, Lu


# ==============================================================================
# Training Loop
# ==============================================================================
def train():
    """Main training and evaluation function for CST with MFDWC features."""
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data loading ---
    src_devices_mixup, target_devices_mixup, label_mix_up, train_src_df, train_tgt_df, test_src_df, test_tgt_df = data_split(src_device, tgt_device, data_path=PATH)
    print(f"Number of classes: {NUM_CLASS}")

    src_dataset = SimpleAudioDataset(file_df=train_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    tgt_dataset = SimpleAudioDataset(file_df=train_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_loader = DataLoader(tgt_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    src_test_dataset = SimpleAudioDataset(file_df=test_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_test_loader = DataLoader(src_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    tgt_test_dataset = SimpleAudioDataset(file_df=test_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- MFDWC feature extractor (frozen) ---
    mfdwc_extractor = MFDWCFeatureExtractor(
        n_mels=MFDWC_N_MELS,
        n_fft=2048,
        hop_length=256,
        wavelet=MFDWC_WAVELET,
        sample_rate=TARGET_SAMPLE_RATE,
        return_temporal=True
    ).to(device)

    # --- Calculate flattened size via dummy forward pass ---
    with torch.no_grad():
        dummy_audio = torch.randn(1, TARGET_SAMPLE_RATE * 10).to(device)
        dummy_mfdwc = mfdwc_extractor(dummy_audio)
        feature_extractor_temp = FeatureExtractor().to(device)
        dummy_features = feature_extractor_temp(dummy_mfdwc)
        flattened_size = dummy_features.shape[1]
        del feature_extractor_temp, dummy_audio, dummy_mfdwc, dummy_features

    print(f"Calculated flattened size: {flattened_size}")
    print(f"MFDWC output shape: (batch, 1, 90, n_frames)")

    # --- Models ---
    feature_extractor = FeatureExtractor().to(device)
    cst_classifier = CST_Classifier(
        num_classes=NUM_CLASS,
        flattened_size=flattened_size,
        bottleneck_dim=BOTTLENECK_DIM
    ).to(device)

    # --- SAM optimizer wrapping SGD ---
    all_params = [
        {"params": feature_extractor.parameters(), "lr": LR},
        {"params": cst_classifier.parameters(), "lr": LR},
    ]
    optimizer = SAM(all_params, optim.SGD, lr=LR,
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY,
                    adaptive=True, rho=RHO)
    lr_scheduler = LambdaLR(optimizer, lambda x: LR * (1.0 + LR_GAMMA * float(x)) ** (-LR_DECAY))

    # --- Tsallis entropy loss ---
    ts_loss = TsallisEntropy(temperature=TEMPERATURE, alpha=ALPHA)

    # --- Logging ---
    results_log = []
    best_target_acc = 0.0
    best_epoch = 0

    experiment_dir = os.path.join(save_dir, f"cst-mfdwc-{src_device}-{tgt_device}")
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Starting CST training with MFDWC features...")
    print(f"  Optimizer: SAM(SGD), LR={LR}, Momentum={MOMENTUM}, Rho={RHO}")
    print(f"  Tsallis: temperature={TEMPERATURE}, alpha={ALPHA}")
    print(f"  Loss weights: transfer={TRADE_OFF}, reverse={TRADE_OFF1}, fixmatch={TRADE_OFF3}")
    print(f"  Pseudo-label threshold: {THRESHOLD}")
    print(f"  SpecAugment: freq_mask={FREQ_MASK_WIDTH}, time_mask={TIME_MASK_WIDTH}")

    for epoch in range(1, NUM_EPOCHS + 1):
        mfdwc_extractor.eval()
        feature_extractor.train()
        cst_classifier.train()

        total_cls_loss = 0
        total_transfer_loss = 0
        total_reverse_loss = 0
        total_fixmatch_loss = 0
        total_fe_grad_norm = 0
        total_cl_grad_norm = 0
        num_batches = 0

        for batch_idx, (src_data, tgt_data) in enumerate(create_combined_loader(src_loader, tgt_loader)):
            src_waveform, src_label = src_data
            tgt_waveform, _ = tgt_data

            if "error" in src_label:
                print(f"Skipping batch {batch_idx+1} due to data loading error in source.")
                continue

            src_label = label_to_numerical(src_label)
            src = src_waveform.to(device)
            labels = src_label.to(device)
            tgt = tgt_waveform.to(device)

            current_batch_size = min(len(src), len(tgt))
            if current_batch_size == 0:
                continue
            src, labels, tgt = src[:current_batch_size], labels[:current_batch_size], tgt[:current_batch_size]

            print(f"\rEpoch {epoch}, Batch {batch_idx+1}", end="", flush=True)

            # --- Extract MFDWC features (frozen, no gradients) ---
            with torch.no_grad():
                src_mfdwc = mfdwc_extractor(src)     # (B, 1, 90, n_frames)
                tgt_mfdwc = mfdwc_extractor(tgt)     # (B, 1, 90, n_frames) — weak

            # Strong augmentation via SpecAugment on target features
            tgt_mfdwc_strong = spec_augment(tgt_mfdwc)

            # --- SAM Step 1: Forward + backward with current weights ---
            optimizer.zero_grad()
            loss1, cls_loss, transfer_loss, reverse_loss, Lu = compute_cst_losses(
                feature_extractor, cst_classifier,
                src_mfdwc, tgt_mfdwc, tgt_mfdwc_strong,
                labels, ts_loss, current_batch_size, device
            )
            loss1.backward()
            optimizer.first_step(zero_grad=True)
            lr_scheduler.step()

            # --- SAM Step 2: Forward + backward with perturbed weights ---
            loss2, _, _, _, _ = compute_cst_losses(
                feature_extractor, cst_classifier,
                src_mfdwc, tgt_mfdwc, tgt_mfdwc_strong,
                labels, ts_loss, current_batch_size, device
            )
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            # Track metrics from SAM step 1
            total_cls_loss += cls_loss.item()
            total_transfer_loss += transfer_loss.item()
            total_reverse_loss += reverse_loss.item()
            total_fixmatch_loss += Lu.item()
            total_fe_grad_norm += check_gradient_norm(feature_extractor)
            total_cl_grad_norm += check_gradient_norm(cst_classifier)
            num_batches += 1

        # --- Epoch-end logging and evaluation ---
        if num_batches == 0:
            print(f"\nEpoch [{epoch}/{NUM_EPOCHS}] - No valid batches processed!")
            continue

        avg_cls = total_cls_loss / num_batches
        avg_transfer = total_transfer_loss / num_batches
        avg_reverse = total_reverse_loss / num_batches
        avg_fixmatch = total_fixmatch_loss / num_batches
        avg_fe_grad = total_fe_grad_norm / num_batches
        avg_cl_grad = total_cl_grad_norm / num_batches

        source_acc = test(mfdwc_extractor, feature_extractor, cst_classifier, src_test_loader, device)
        target_acc = test(mfdwc_extractor, feature_extractor, cst_classifier, tgt_test_loader, device)

        print(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}] ({num_batches} batches) | "
            f"Cls: {avg_cls:.4f}, Transfer: {avg_transfer:.4f}, "
            f"Reverse: {avg_reverse:.4f}, FixMatch: {avg_fixmatch:.4f} | "
            f"Src Acc: {source_acc:.2f}%, Tgt Acc: {target_acc:.2f}%"
        )
        print(
            f"  -> Avg Grad Norms | Feature Extractor: {avg_fe_grad:.4f}, "
            f"CST Classifier: {avg_cl_grad:.4f}\n"
        )

        epoch_results = {
            'epoch': epoch,
            'classification_loss': avg_cls,
            'transfer_loss': avg_transfer,
            'reverse_loss': avg_reverse,
            'fixmatch_loss': avg_fixmatch,
            'source_accuracy': source_acc,
            'target_accuracy': target_acc,
            'feature_extractor_grad_norm': avg_fe_grad,
            'classifier_grad_norm': avg_cl_grad
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
                'model_state_dict': cst_classifier.state_dict(),
                'target_accuracy': target_acc,
                'source_accuracy': source_acc
            }, os.path.join(experiment_dir, f"best_CL_{src_device}-{tgt_device}.pth"))

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
            'model_state_dict': cst_classifier.state_dict(),
            'target_accuracy': target_acc,
            'source_accuracy': source_acc
        }, os.path.join(experiment_dir, f"latest_CL_{src_device}-{tgt_device}.pth"))

        # Save CSV every epoch
        csv_path = os.path.join(experiment_dir, f"training_results_cst_mfdwc_{src_device}-{tgt_device}.csv")
        pd.DataFrame(results_log).to_csv(csv_path, index=False)

    print(f"\nTraining finished!")
    print(f"Best target accuracy: {best_target_acc:.2f}% at epoch {best_epoch}")
    print(f"Results saved to {experiment_dir}")


if __name__ == '__main__':
    train()
