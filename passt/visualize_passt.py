"""
Post-training visualization and evaluation for PaSST experiments.

Supports three model types via --model-type:
  - dann       : DANN+GRL (PASSETFeatureExtractor + Classifier + Discriminator)
  - source_only: Source-only baseline (PaSST + simple classifier, no adaptation)
  - cst        : Cycle Self-Training (AudioClassifier with bottleneck)

Generates: t-SNE plots, confusion matrices, per-class accuracy bars,
classification reports, and training curve plots from JSON history.
Plot styling matches visualize.py exactly for side-by-side comparison
with MFDWC-based models.

Data loading uses data.py (from UDA-mfdwc-wavelet_features) which reads
the dataset via meta.csv / fold1_evaluate.csv splits, so the dataset no
longer needs to be manually organised into source/target folders.

Label mapping: all PaSST-based models use **alphabetically sorted** scene
labels (airport=0, bus=1, metro=2, ..., tram=9).  This differs from
data.py's custom label_keys ordering.  The script uses DEFAULT_LABEL_TO_IDX
for correct alignment; source-only checkpoints that embed their own mapping
are honoured automatically.

Usage examples:
    # DANN+GRL evaluation (default)
    python visualize_passt.py --model-type dann --checkpoint passt_grl.pth \
        --data-path /DATA/G3/Datasets/.../TAU-urban-acoustic-scenes-2020-mobile-development \
        --data-script-dir /path/to/UDA-mfdwc-wavelet_features \
        --devices b,c,s1,s2,s3

    # Source-only baseline evaluation
    python visualize_passt.py --model-type source_only --checkpoint passt_baseline.pth \
        --data-path /DATA/G3/Datasets/.../TAU-urban-acoustic-scenes-2020-mobile-development \
        --data-script-dir /path/to/UDA-mfdwc-wavelet_features \
        --devices b,c,s1,s2,s3

    # Cycle Self-Training evaluation
    python visualize_passt.py --model-type cst --checkpoint passt_cst.pth \
        --data-path /DATA/G3/Datasets/.../TAU-urban-acoustic-scenes-2020-mobile-development \
        --data-script-dir /path/to/UDA-mfdwc-wavelet_features \
        --devices b,c,s1,s2,s3

    # Plot training curves only (DANN, no model loading needed)
    python visualize_passt.py --plot-curves-only --history-json history.json

    # Custom output directory
    python visualize_passt.py --model-type dann --checkpoint passt_grl.pth \
        --data-path /DATA/G3/Datasets/.../TAU-urban-acoustic-scenes-2020-mobile-development \
        --data-script-dir /path/to/UDA-mfdwc-wavelet_features \
        --devices b,c,s1,s2,s3 --output-dir my_plots/
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from hear21passt.base import get_basic_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGET_SAMPLE_RATE = 32000
MAX_LEN_SECONDS = 10

# All PaSST-based models (DANN, source-only, CST) use alphabetically sorted labels.
# This differs from data.py's label_keys which has a custom (non-alphabetical) order.
SCENE_LABELS = sorted([
    'airport', 'bus', 'metro', 'metro_station', 'park',
    'public_square', 'shopping_mall', 'street_pedestrian',
    'street_traffic', 'tram',
])
DEFAULT_LABEL_TO_IDX = {label: i for i, label in enumerate(SCENE_LABELS)}


# ==============================================================================
# Model definitions (from dann_dcase_w_grl.ipynb)
# ==============================================================================
class PASSETFeatureExtractor(nn.Module):
    def __init__(self, freeze_passt=True):
        super(PASSETFeatureExtractor, self).__init__()
        self.passt_model = get_basic_model(mode="embed_only")
        self.passt_model.eval()

        if freeze_passt:
            for param in self.passt_model.parameters():
                param.requires_grad = False

        self.feature_dim = 768
        self.adaptation_layers = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        with torch.no_grad():
            passt_features = self.passt_model(x)
        return self.adaptation_layers(passt_features)


class Classifier(nn.Module):
    def __init__(self, input_size=256, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, h):
        return self.layer(h)


class Discriminator(nn.Module):
    def __init__(self, input_size=256, num_classes=1):
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
            nn.Linear(64, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, h):
        return self.layer(h)


# ==============================================================================
# Model definitions — Source-only baseline (from source_only_dcase_w_ds.ipynb)
# ==============================================================================
class SourceOnlyFeatureExtractor(nn.Module):
    """PaSST feature extractor for source-only baseline (768-dim output, no adaptation)."""
    def __init__(self):
        super(SourceOnlyFeatureExtractor, self).__init__()
        self.model = get_basic_model(mode="embed_only")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            features = self.model(x)
        return features


class SourceOnlyClassifier(nn.Module):
    """Classifier for source-only baseline (768 -> 128 -> num_classes)."""
    def __init__(self, input_size=768, num_classes=10):
        super(SourceOnlyClassifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, h):
        return self.layer(h)


# ==============================================================================
# Model definitions — CST / Cycle Self-Training (from cst_dcase.ipynb)
# ==============================================================================
class CSTAudioClassifier(nn.Module):
    """AudioClassifier for CST: backbone + bottleneck + head.

    forward() returns (outputs, embeddings) where embeddings are 256-dim.
    """
    def __init__(self, num_classes=10, bottleneck_dim=256):
        super(CSTAudioClassifier, self).__init__()
        self.backbone = get_basic_model(mode="embed_only")
        self.bottleneck = nn.Sequential(
            nn.Linear(768, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.bottleneck(features)
        outputs = self.head(embeddings)
        return outputs, embeddings


# ==============================================================================
# Dataset (adapted for DataFrame-based loading via data.py)
# ==============================================================================
class AudioDataset(Dataset):
    def __init__(self, root, file_df, label_to_idx, device_name,
                 target_sr=TARGET_SAMPLE_RATE, max_len_seconds=MAX_LEN_SECONDS):
        self.label_to_idx = label_to_idx
        self.device_name = device_name
        self.target_sr = target_sr
        self.max_len_samples = target_sr * max_len_seconds
        self.paths = []
        self.labels = []
        for _, row in file_df.iterrows():
            self.paths.append(os.path.join(root, row['filename']))
            self.labels.append(row['scene_label'])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        file_path = self.paths[idx]
        label_idx = self.label_to_idx[self.labels[idx]]

        waveform, _ = librosa.load(file_path, sr=self.target_sr, mono=True)
        if len(waveform) < self.max_len_samples:
            waveform = np.pad(waveform, (0, self.max_len_samples - len(waveform)))
        else:
            waveform = waveform[:self.max_len_samples]

        return torch.tensor(waveform, dtype=torch.float32), label_idx, self.device_name


# ==============================================================================
# Model loading
# ==============================================================================
def load_models(checkpoint_path, num_classes, device):
    """Load PaSST feature extractor, classifier, and discriminator from checkpoint.

    Auto-detects the checkpoint architecture:
      - Keys starting with 'passt_model.' + 'adaptation_layers.' → PASSETFeatureExtractor (256-dim)
      - Keys starting with 'model.' without adaptation_layers → SourceOnlyFeatureExtractor (768-dim)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    fe_sd = ckpt['feature_extractor_state_dict']

    has_adaptation = any(k.startswith('adaptation_layers.') for k in fe_sd)
    has_passt_model = any(k.startswith('passt_model.') for k in fe_sd)

    if has_adaptation and has_passt_model:
        # Full DANN feature extractor with adaptation layers (768 → 256)
        feature_extractor = PASSETFeatureExtractor().to(device)
        feat_dim = 256
    else:
        # Plain PaSST wrapper (768-dim output, no adaptation layers)
        feature_extractor = SourceOnlyFeatureExtractor().to(device)
        feat_dim = 768

    classifier = Classifier(input_size=feat_dim, num_classes=num_classes).to(device)
    discriminator = Discriminator(input_size=feat_dim).to(device)

    feature_extractor.load_state_dict(fe_sd)
    classifier.load_state_dict(ckpt['classifier_state_dict'])
    discriminator.load_state_dict(ckpt['discriminator_state_dict'])

    feature_extractor.eval()
    classifier.eval()
    discriminator.eval()

    print(f"Loaded DANN checkpoint from {checkpoint_path} (feat_dim={feat_dim})")
    return feature_extractor, classifier, discriminator


def load_source_only_models(checkpoint_path, num_classes, device):
    """Load source-only baseline feature extractor and classifier from checkpoint."""
    feature_extractor = SourceOnlyFeatureExtractor().to(device)
    classifier = SourceOnlyClassifier(input_size=768, num_classes=num_classes).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    feature_extractor.load_state_dict(ckpt['feature_extractor_state_dict'])
    classifier.load_state_dict(ckpt['classifier_state_dict'])

    feature_extractor.eval()
    classifier.eval()

    print(f"Loaded source-only checkpoint from {checkpoint_path}")
    return feature_extractor, classifier


def load_cst_model(checkpoint_path, num_classes, device):
    """Load CST AudioClassifier from checkpoint (flat state_dict)."""
    model = CSTAudioClassifier(num_classes=num_classes).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Handle both bare state_dict and wrapped formats
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)

    model.eval()

    print(f"Loaded CST checkpoint from {checkpoint_path}")
    return model


# ==============================================================================
# Data loading (uses DataFrames from data.py's data_split())
# ==============================================================================
def build_test_loaders(data_path, test_src_df, test_tgt_df, src_device, tgt_devices,
                       label_to_idx, batch_size, num_workers=4):
    """Build per-device test DataLoaders from pre-split DataFrames."""
    pin = torch.cuda.is_available()
    loaders = {}

    # Source
    if src_device in test_src_df and len(test_src_df[src_device]) > 0:
        ds = AudioDataset(data_path, test_src_df[src_device], label_to_idx, src_device)
        loaders[src_device] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=pin)

    # Targets
    for dev in tgt_devices:
        if dev in test_tgt_df and len(test_tgt_df[dev]) > 0:
            ds = AudioDataset(data_path, test_tgt_df[dev], label_to_idx, dev)
            loaders[dev] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, pin_memory=pin)

    return loaders


# ==============================================================================
# Feature & prediction extraction
# ==============================================================================
def extract_features_and_predictions(feature_extractor, classifier, dataloader, device):
    """Run inference, collect adapted features (256-d), predictions, and labels.

    Returns dict matching visualize.py format:
        features, predictions, true_labels, embeddings (always None)
    """
    all_features = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, labels, _ in dataloader:
            audio = audio.to(device)
            labels_t = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels

            features = feature_extractor(audio)
            logits = classifier(features)
            preds = logits.argmax(dim=1)

            all_features.append(features.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_t.numpy())

    return {
        'features': np.concatenate(all_features, axis=0),
        'predictions': np.concatenate(all_preds, axis=0),
        'true_labels': np.concatenate(all_labels, axis=0),
        'embeddings': None,
    }


def extract_features_source_only(feature_extractor, classifier, dataloader, device):
    """Run inference for source-only baseline (768-dim PaSST features)."""
    all_features = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, labels, _ in dataloader:
            audio = audio.to(device)
            labels_t = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels

            features = feature_extractor(audio)
            logits = classifier(features)
            preds = logits.argmax(dim=1)

            all_features.append(features.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_t.numpy())

    return {
        'features': np.concatenate(all_features, axis=0),
        'predictions': np.concatenate(all_preds, axis=0),
        'true_labels': np.concatenate(all_labels, axis=0),
        'embeddings': None,
    }


def extract_features_cst(model, dataloader, device):
    """Run inference for CST model (256-dim bottleneck embeddings as features)."""
    all_features = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, labels, _ in dataloader:
            audio = audio.to(device)
            labels_t = torch.tensor(labels, dtype=torch.long) if not isinstance(labels, torch.Tensor) else labels

            outputs, embeddings = model(audio)
            preds = outputs.argmax(dim=1)

            all_features.append(embeddings.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels_t.numpy())

    return {
        'features': np.concatenate(all_features, axis=0),
        'predictions': np.concatenate(all_preds, axis=0),
        'true_labels': np.concatenate(all_labels, axis=0),
        'embeddings': None,
    }


# ==============================================================================
# t-SNE plots  (matches visualize.py styling exactly)
# ==============================================================================
def plot_tsne(features_dict, class_names, output_dir, tag=''):
    from sklearn.manifold import TSNE

    all_feats, all_labels, all_domains = [], [], []
    for domain_name, data in features_dict.items():
        all_feats.append(data['features'])
        all_labels.append(data['true_labels'])
        all_domains.extend([domain_name] * len(data['true_labels']))

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domains = np.array(all_domains)

    max_points = 5000
    if len(all_feats) > max_points:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_feats), max_points, replace=False)
        all_feats = all_feats[idx]
        all_labels = all_labels[idx]
        all_domains = all_domains[idx]

    print(f"  Running t-SNE on {len(all_feats)} samples...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    coords = tsne.fit_transform(all_feats)

    suffix = f'_{tag}' if tag else ''

    # --- Plot by class ---
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab10', len(class_names))
    for i, name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(i)],
                   label=name, s=8, alpha=0.6)
    ax.legend(fontsize=8, markerscale=2, loc='best')
    ax.set_title(f't-SNE by Class{" (" + tag + ")" if tag else ""}')
    ax.set_xticks([])
    ax.set_yticks([])
    path = os.path.join(output_dir, f'tsne_by_class{suffix}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")

    # --- Plot by domain ---
    unique_domains = sorted(set(all_domains))
    fig, ax = plt.subplots(figsize=(10, 8))
    domain_cmap = plt.cm.get_cmap('Set1', len(unique_domains))
    for i, dom in enumerate(unique_domains):
        mask = all_domains == dom
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[domain_cmap(i)],
                   label=f'Device {dom}', s=8, alpha=0.6)
    ax.legend(fontsize=8, markerscale=2, loc='best')
    ax.set_title(f't-SNE by Domain{" (" + tag + ")" if tag else ""}')
    ax.set_xticks([])
    ax.set_yticks([])
    path = os.path.join(output_dir, f'tsne_by_domain{suffix}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ==============================================================================
# Confusion matrix  (matches visualize.py styling exactly)
# ==============================================================================
def plot_confusion_matrix(true_labels, predictions, class_names, device_name, output_dir):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(true_labels, predictions, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=45)
    ax.set_title(f'Confusion Matrix \u2014 Device {device_name}')
    path = os.path.join(output_dir, f'confusion_matrix_{device_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ==============================================================================
# Per-class accuracy bar chart  (matches visualize.py styling exactly)
# ==============================================================================
def plot_per_class_accuracy(results_by_device, class_names, output_dir):
    device_names = list(results_by_device.keys())
    n_classes = len(class_names)
    n_devices = len(device_names)

    accuracies = np.zeros((n_devices, n_classes))
    for d_idx, dev in enumerate(device_names):
        true = results_by_device[dev]['true_labels']
        pred = results_by_device[dev]['predictions']
        for c in range(n_classes):
            mask = true == c
            if mask.sum() > 0:
                accuracies[d_idx, c] = 100.0 * (pred[mask] == c).sum() / mask.sum()

    x = np.arange(n_classes)
    width = 0.8 / n_devices
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.get_cmap('Set2', n_devices)

    for d_idx, dev in enumerate(device_names):
        offset = (d_idx - n_devices / 2 + 0.5) * width
        ax.bar(x + offset, accuracies[d_idx], width, label=f'Device {dev}',
               color=cmap(d_idx), edgecolor='grey', linewidth=0.5)

    ax.set_xlabel('Scene Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy by Device')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    path = os.path.join(output_dir, 'per_class_accuracy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ==============================================================================
# Classification report  (matches visualize.py styling exactly)
# ==============================================================================
def save_classification_report(true_labels, predictions, class_names, device_name, output_dir):
    from sklearn.metrics import classification_report

    report_str = classification_report(
        true_labels, predictions,
        target_names=class_names, digits=4, zero_division=0
    )
    print(f"\n  Classification Report \u2014 Device {device_name}:")
    print(report_str)

    report_dict = classification_report(
        true_labels, predictions,
        target_names=class_names, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report_dict).T
    path = os.path.join(output_dir, f'classification_report_{device_name}.csv')
    df.to_csv(path)
    print(f"  Saved {path}")


# ==============================================================================
# Training curves from JSON history  (matches visualize.py curve styling)
# ==============================================================================
def plot_training_curves(history_path, output_dir):
    """Plot training loss, domain accuracy, and source accuracy from a JSON history file.

    The JSON should contain the dict returned by train_dann_improved():
        cls_loss, disc_loss, domain_acc, lambda_values  (per epoch)
        src_acc  (every 5 epochs)
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = list(range(1, len(history['cls_loss']) + 1))

    # --- Loss curves ---
    fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
    ax_loss.plot(epochs, history['cls_loss'], label='Classification loss', linewidth=1)
    ax_loss.plot(epochs, history['disc_loss'], label='Discriminator loss',
                 linewidth=1, linestyle='--')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Training Loss Curves')
    ax_loss.legend(fontsize=7, loc='best')
    ax_loss.grid(alpha=0.3)
    path = os.path.join(output_dir, 'training_loss_curves.png')
    fig_loss.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig_loss)
    print(f"  Saved {path}")

    # --- Accuracy / domain curves ---
    fig_acc, ax_acc = plt.subplots(figsize=(12, 6))
    ax_acc.plot(epochs, [a * 100 for a in history['domain_acc']],
                label='Domain accuracy', linewidth=1)
    if history.get('src_acc'):
        # src_acc recorded every 5 epochs
        src_epochs = list(range(5, 5 * len(history['src_acc']) + 1, 5))
        ax_acc.plot(src_epochs, [a * 100 for a in history['src_acc']],
                    label='Source val accuracy', linewidth=1, linestyle='--', marker='o',
                    markersize=3)
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Training Accuracy Curves')
    ax_acc.legend(fontsize=7, loc='best')
    ax_acc.grid(alpha=0.3)
    path = os.path.join(output_dir, 'training_accuracy_curves.png')
    fig_acc.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig_acc)
    print(f"  Saved {path}")

    # --- Lambda schedule ---
    fig_lam, ax_lam = plt.subplots(figsize=(12, 6))
    ax_lam.plot(epochs, history['lambda_values'], label='GRL \u03bb', linewidth=1, color='tab:red')
    ax_lam.set_xlabel('Epoch')
    ax_lam.set_ylabel('\u03bb')
    ax_lam.set_title('GRL Lambda Schedule')
    ax_lam.legend(fontsize=7, loc='best')
    ax_lam.grid(alpha=0.3)
    path = os.path.join(output_dir, 'lambda_schedule.png')
    fig_lam.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig_lam)
    print(f"  Saved {path}")


# ==============================================================================
# CLI
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Post-training visualization for PaSST experiments (DANN, source-only, CST)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model-type', type=str, default='dann',
                        choices=['dann', 'source_only', 'cst'],
                        help='Model type: dann, source_only, or cst (default: dann)')
    parser.add_argument('--checkpoint', type=str, default='dann_audio_dcase.pth',
                        help='Path to .pth checkpoint (default: dann_audio_dcase.pth)')
    parser.add_argument('--data-path', type=str,
                        help='Path to TAU dataset root (where meta.csv and audio/ live)')
    parser.add_argument('--data-script-dir', type=str,
                        help='Path to directory containing data.py (UDA-mfdwc-wavelet_features)')
    parser.add_argument('--src-device', type=str, default='a',
                        help='Source device (default: a)')
    parser.add_argument('--devices', type=str, default='b,c,s1,s2,s3',
                        help='Comma-separated target devices to evaluate (default: b,c,s1,s2,s3)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader worker processes (default: 4, 0 to disable)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: plots_passt/ next to checkpoint)')
    parser.add_argument('--history-json', type=str, default=None,
                        help='Path to training history JSON file (for training curve plots)')
    parser.add_argument('--plot-curves-only', action='store_true',
                        help='Only plot training curves from JSON (no model loading)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.checkpoint:
        output_dir = os.path.join(os.path.dirname(args.checkpoint) or '.', 'plots_passt')
    else:
        output_dir = './plots_passt'
    os.makedirs(output_dir, exist_ok=True)

    # --- Curves-only mode ---
    if args.plot_curves_only:
        if not args.history_json:
            print("Error: --history-json is required with --plot-curves-only")
            sys.exit(1)
        print(f"Plotting training curves from {args.history_json}...")
        plot_training_curves(args.history_json, output_dir)
        print(f"Done. Outputs in {output_dir}")
        return

    # --- Full evaluation mode ---
    if not args.data_path or not args.data_script_dir:
        print("Error: --data-path and --data-script-dir are required for full evaluation mode.")
        sys.exit(1)

    # Import data.py from the specified directory (only for data_split file lists)
    sys.path.insert(0, args.data_script_dir)
    from data import data_split

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tgt_devices = [d.strip() for d in args.devices.split(',')]

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Source: {args.src_device}, Targets: {tgt_devices}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Label mapping — all PaSST models use alphabetically sorted labels.
    # For source_only, the checkpoint stores the mapping; use it if available.
    label_to_idx = DEFAULT_LABEL_TO_IDX
    if args.model_type == 'source_only':
        ckpt_peek = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt_peek, dict) and 'label_to_idx' in ckpt_peek:
            label_to_idx = ckpt_peek['label_to_idx']
            print(f"Using label_to_idx from checkpoint: {label_to_idx}")
        del ckpt_peek

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Load models based on model type (auto-detect if not explicitly set)
    model_type = args.model_type
    if model_type == 'dann':
        # Auto-detect: peek at checkpoint to see if it's really DANN, source_only, or CST
        ckpt_peek = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt_peek, dict) and 'feature_extractor_state_dict' in ckpt_peek:
            if 'discriminator_state_dict' in ckpt_peek:
                model_type = 'dann'
            else:
                model_type = 'source_only'
        else:
            # Bare state_dict or dict without feature_extractor_state_dict → CST
            sd = ckpt_peek.get('state_dict', ckpt_peek) if isinstance(ckpt_peek, dict) else ckpt_peek
            if any(k.startswith('backbone.') for k in sd):
                model_type = 'cst'
        del ckpt_peek
        if model_type != args.model_type:
            print(f"Auto-detected model type: {model_type} (override with --model-type)")
    print(f"Model type: {model_type}")

    if model_type == 'dann':
        feat_ext, cls_head, _ = load_models(args.checkpoint, num_classes, device)
    elif model_type == 'source_only':
        feat_ext, cls_head = load_source_only_models(args.checkpoint, num_classes, device)
    elif model_type == 'cst':
        cst_model = load_cst_model(args.checkpoint, num_classes, device)

    # Split dataset via data.py (test splits are device-independent)
    _, _, _, _, _, test_src_df, test_tgt_df = data_split(
        args.src_device, tgt_devices[0], data_path=args.data_path
    )

    # Build test loaders (one per device)
    loaders = build_test_loaders(
        args.data_path, test_src_df, test_tgt_df, args.src_device, tgt_devices,
        label_to_idx, args.batch_size, num_workers=args.num_workers
    )
    print(f"Test loaders: {list(loaders.keys())}")

    # Extract features and predictions
    print("\nExtracting features and predictions...")
    results = {}
    for dev_name, loader in loaders.items():
        print(f"  Processing device '{dev_name}'...")
        if model_type == 'cst':
            results[dev_name] = extract_features_cst(cst_model, loader, device)
        elif model_type == 'source_only':
            results[dev_name] = extract_features_source_only(feat_ext, cls_head, loader, device)
        else:
            results[dev_name] = extract_features_and_predictions(feat_ext, cls_head, loader, device)
        n = len(results[dev_name]['true_labels'])
        acc = 100.0 * (results[dev_name]['predictions'] == results[dev_name]['true_labels']).mean()
        print(f"    {n} samples, accuracy: {acc:.2f}%")

    # Overall accuracy across all devices
    all_true = np.concatenate([r['true_labels'] for r in results.values()])
    all_pred = np.concatenate([r['predictions'] for r in results.values()])
    overall_acc = 100.0 * (all_pred == all_true).mean()
    print(f"\n  Overall accuracy ({len(all_true)} samples): {overall_acc:.2f}%")

    # Source-only / target-only aggregated accuracy
    src_devs = {args.src_device}
    tgt_results = {k: v for k, v in results.items() if k not in src_devs}
    if tgt_results:
        tgt_true = np.concatenate([r['true_labels'] for r in tgt_results.values()])
        tgt_pred = np.concatenate([r['predictions'] for r in tgt_results.values()])
        tgt_acc = 100.0 * (tgt_pred == tgt_true).mean()
        print(f"  Target-only accuracy ({len(tgt_true)} samples): {tgt_acc:.2f}%")
    src_results = {k: v for k, v in results.items() if k in src_devs}
    if src_results:
        src_true = np.concatenate([r['true_labels'] for r in src_results.values()])
        src_pred = np.concatenate([r['predictions'] for r in src_results.values()])
        src_acc = 100.0 * (src_pred == src_true).mean()
        print(f"  Source-only accuracy ({len(src_true)} samples): {src_acc:.2f}%")

    # --- Generate plots ---
    print("\nGenerating t-SNE plots...")
    plot_tsne(results, class_names, output_dir)

    print("\nGenerating confusion matrices...")
    for dev_name, res in results.items():
        plot_confusion_matrix(res['true_labels'], res['predictions'],
                              class_names, dev_name, output_dir)

    print("\nGenerating per-class accuracy chart...")
    plot_per_class_accuracy(results, class_names, output_dir)

    print("\nGenerating classification reports...")
    for dev_name, res in results.items():
        save_classification_report(res['true_labels'], res['predictions'],
                                   class_names, dev_name, output_dir)

    # Training curves from JSON
    history_path = args.history_json
    if not history_path:
        # Auto-detect in checkpoint directory
        candidate = os.path.join(os.path.dirname(args.checkpoint) or '.', 'history.json')
        if os.path.isfile(candidate):
            history_path = candidate
    if history_path and os.path.isfile(history_path):
        print(f"\nPlotting training curves from {history_path}...")
        plot_training_curves(history_path, output_dir)

    print(f"\nAll outputs saved to {output_dir}")
    print("Done.")


if __name__ == '__main__':
    main()