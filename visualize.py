"""
Post-training visualization and evaluation for UDA experiments.

Supports baseline (source-only CNN), GRL, and CST methods.
Generates: t-SNE plots, confusion matrices, per-class accuracy bars,
classification reports, and training curve plots from CSV logs.

Usage examples:
    # Evaluate a baseline single-device checkpoint
    python visualize.py --method baseline --checkpoint-dir results/mfdwc-cnn/a-b \
        --data-path /DATA/G3/Datasets/... --devices b

    # Evaluate a GRL all-device checkpoint
    python visualize.py --method grl --checkpoint-dir results/grl-mfdwc-a-b+c+s1+s2+s3 \
        --data-path /DATA/G3/Datasets/... --devices b,c,s1,s2,s3

    # Evaluate CST with bottleneck embedding t-SNE
    python visualize.py --method cst --checkpoint-dir results/cst-mfdwc-a-b \
        --data-path /DATA/G3/Datasets/... --devices b

    # Plot training curves only (no model loading needed)
    python visualize.py --csv-dir results/mfdwc-grl/csv_logs --plot-curves-only

    # Source device defaults to 'a', override with --src-device
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data import data_split, label_to_numerical, SimpleAudioDataset, label_keys
from models import Feature_extractor as FeatureExtractor
from models import Classifier_no_weights as Classifier
from models import CST_Classifier
from mfdwc_extractor_with_flag import MFDWCFeatureExtractor

# Reverse label_keys: index -> name
INDEX_TO_LABEL = {v: k for k, v in label_keys.items()}
CLASS_NAMES = [INDEX_TO_LABEL[i] for i in range(len(INDEX_TO_LABEL))]

TARGET_SAMPLE_RATE = 44100
MFDWC_N_MELS = 60
MFDWC_WAVELET = 'haar'

ALL_TARGET_DEVICES = ['b', 'c', 's1', 's2', 's3']


# ==============================================================================
# Model loading
# ==============================================================================
def load_models(method, checkpoint_dir, device):
    """Load MFDWC extractor, Feature_extractor, and classifier from checkpoints.

    Returns:
        (mfdwc_extractor, feature_extractor, classifier, flattened_size)
        For CST, classifier is CST_Classifier (returns logits, embeddings).
    """
    mfdwc_extractor = MFDWCFeatureExtractor(
        n_mels=MFDWC_N_MELS, n_fft=2048, hop_length=256,
        wavelet=MFDWC_WAVELET, sample_rate=TARGET_SAMPLE_RATE,
        return_temporal=True
    ).to(device)

    # Infer flattened size via dummy forward
    with torch.no_grad():
        dummy = torch.randn(1, TARGET_SAMPLE_RATE * 10).to(device)
        dummy_mfdwc = mfdwc_extractor(dummy)
        fe_temp = FeatureExtractor().to(device)
        dummy_feat = fe_temp(dummy_mfdwc)
        flattened_size = dummy_feat.shape[1]
        del fe_temp, dummy, dummy_mfdwc, dummy_feat

    # Find checkpoint files
    fe_path = _find_checkpoint(checkpoint_dir, 'best_FE_')
    cl_path = _find_checkpoint(checkpoint_dir, 'best_CL_')

    feature_extractor = FeatureExtractor().to(device)
    fe_ckpt = torch.load(fe_path, map_location=device, weights_only=False)
    fe_state = fe_ckpt['model_state_dict'] if isinstance(fe_ckpt, dict) and 'model_state_dict' in fe_ckpt else fe_ckpt
    feature_extractor.load_state_dict(fe_state)

    if method == 'cst':
        classifier = CST_Classifier(
            num_classes=len(CLASS_NAMES),
            flattened_size=flattened_size,
            bottleneck_dim=256
        ).to(device)
    else:
        classifier = Classifier(
            num_classes=len(CLASS_NAMES),
            flattened_size=flattened_size
        ).to(device)

    cl_ckpt = torch.load(cl_path, map_location=device, weights_only=False)
    cl_state = cl_ckpt['model_state_dict'] if isinstance(cl_ckpt, dict) and 'model_state_dict' in cl_ckpt else cl_ckpt
    classifier.load_state_dict(cl_state)

    mfdwc_extractor.eval()
    feature_extractor.eval()
    classifier.eval()

    print(f"Loaded checkpoints from {checkpoint_dir}")
    print(f"  FE: {os.path.basename(fe_path)} | CL: {os.path.basename(cl_path)}")
    if 'epoch' in fe_ckpt:
        print(f"  Best epoch: {fe_ckpt['epoch']}")

    return mfdwc_extractor, feature_extractor, classifier, flattened_size


def _find_checkpoint(directory, prefix):
    """Find a checkpoint file by prefix in a directory."""
    matches = glob.glob(os.path.join(directory, f"{prefix}*.pth"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching '{prefix}*.pth' found in {directory}"
        )
    return matches[0]


# ==============================================================================
# Feature & prediction extraction
# ==============================================================================
def extract_features_and_predictions(mfdwc_extractor, feature_extractor, classifier,
                                     dataloader, device, method='baseline'):
    """Run inference and collect features, predictions, and labels.

    Returns dict with keys:
        features: np.ndarray (N, feat_dim) — post Feature_extractor
        embeddings: np.ndarray (N, bottleneck_dim) — CST only, else None
        predictions: np.ndarray (N,)
        true_labels: np.ndarray (N,)
    """
    all_features = []
    all_embeddings = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for audio, target in dataloader:
            valid_indices = [i for i, lbl in enumerate(target) if lbl != "error"]
            if not valid_indices:
                continue

            audio = audio[valid_indices]
            target = [target[i] for i in valid_indices]
            labels = label_to_numerical(target)

            audio = audio.to(device)
            mfdwc = mfdwc_extractor(audio)
            features = feature_extractor(mfdwc)

            if method == 'cst':
                logits, embeddings = classifier(features)
                all_embeddings.append(embeddings.cpu().numpy())
                preds = logits.argmax(dim=1)
            else:
                logits = classifier(features)
                preds = logits.argmax(dim=1)

            all_features.append(features.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    result = {
        'features': np.concatenate(all_features, axis=0),
        'predictions': np.concatenate(all_preds, axis=0),
        'true_labels': np.concatenate(all_labels, axis=0),
        'embeddings': np.concatenate(all_embeddings, axis=0) if all_embeddings else None,
    }
    return result


# ==============================================================================
# t-SNE plots
# ==============================================================================
def plot_tsne(features_dict, output_dir, tag=''):
    """Generate t-SNE plots colored by class and by domain.

    Args:
        features_dict: dict mapping domain_name -> extract_features_and_predictions result
        output_dir: directory to save PNGs
        tag: optional filename suffix (e.g. 'embeddings' for CST bottleneck)
    """
    from sklearn.manifold import TSNE

    # Gather all features, labels, domains
    all_feats = []
    all_labels = []
    all_domains = []
    for domain_name, data in features_dict.items():
        feat_key = 'embeddings' if (tag == 'embeddings' and data['embeddings'] is not None) else 'features'
        all_feats.append(data[feat_key])
        all_labels.append(data['true_labels'])
        all_domains.extend([domain_name] * len(data['true_labels']))

    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_domains = np.array(all_domains)

    # Subsample if too many points for reasonable t-SNE performance
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
    cmap = plt.cm.get_cmap('tab10', len(CLASS_NAMES))
    for i, name in enumerate(CLASS_NAMES):
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
# Confusion matrix
# ==============================================================================
def plot_confusion_matrix(true_labels, predictions, device_name, output_dir):
    """Plot and save a confusion matrix for a single device."""
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(true_labels, predictions, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=45)
    ax.set_title(f'Confusion Matrix — Device {device_name}')
    path = os.path.join(output_dir, f'confusion_matrix_{device_name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ==============================================================================
# Per-class accuracy bar chart
# ==============================================================================
def plot_per_class_accuracy(results_by_device, output_dir):
    """Grouped bar chart of per-class accuracy across devices.

    Args:
        results_by_device: dict mapping device_name -> {true_labels, predictions}
    """
    device_names = list(results_by_device.keys())
    n_classes = len(CLASS_NAMES)
    n_devices = len(device_names)

    # Compute per-class accuracy for each device
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
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    path = os.path.join(output_dir, 'per_class_accuracy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ==============================================================================
# Classification report
# ==============================================================================
def save_classification_report(true_labels, predictions, device_name, output_dir):
    """Print and save sklearn classification report as CSV."""
    from sklearn.metrics import classification_report

    report_str = classification_report(
        true_labels, predictions,
        target_names=CLASS_NAMES, digits=4, zero_division=0
    )
    print(f"\n  Classification Report — Device {device_name}:")
    print(report_str)

    report_dict = classification_report(
        true_labels, predictions,
        target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    df = pd.DataFrame(report_dict).T
    path = os.path.join(output_dir, f'classification_report_{device_name}.csv')
    df.to_csv(path)
    print(f"  Saved {path}")


# ==============================================================================
# Training curves from CSV logs
# ==============================================================================
def plot_training_curves(csv_dir, output_dir, method=None):
    """Plot training loss and accuracy curves from CSV log files.

    Finds all CSV files in csv_dir and overlays them.
    """
    csv_files = sorted(glob.glob(os.path.join(csv_dir, '*.csv')))
    if not csv_files:
        print(f"  No CSV files found in {csv_dir}")
        return

    print(f"  Found {len(csv_files)} CSV log(s) in {csv_dir}")

    # --- Loss curves ---
    fig_loss, ax_loss = plt.subplots(figsize=(12, 6))
    fig_acc, ax_acc = plt.subplots(figsize=(12, 6))

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        fname = os.path.splitext(os.path.basename(csv_path))[0]
        # Shorten label: extract device pair from filename
        label = fname.split('_')[-1] if '_' in fname else fname

        epochs = df['epoch']

        # Plot classification loss (always present)
        if 'classification_loss' in df.columns:
            ax_loss.plot(epochs, df['classification_loss'], label=f'{label} cls', linewidth=1)

        # Method-specific losses
        if 'adversarial_loss' in df.columns:
            ax_loss.plot(epochs, df['adversarial_loss'], label=f'{label} adv',
                         linewidth=1, linestyle='--')
        if 'transfer_loss' in df.columns:
            ax_loss.plot(epochs, df['transfer_loss'], label=f'{label} transfer',
                         linewidth=1, linestyle='--')
        if 'reverse_loss' in df.columns:
            ax_loss.plot(epochs, df['reverse_loss'], label=f'{label} reverse',
                         linewidth=1, linestyle=':')
        if 'fixmatch_loss' in df.columns:
            ax_loss.plot(epochs, df['fixmatch_loss'], label=f'{label} fixmatch',
                         linewidth=1, linestyle=':')

        # Accuracy curves
        if 'source_accuracy' in df.columns:
            ax_acc.plot(epochs, df['source_accuracy'], label=f'{label} src', linewidth=1)
        if 'target_accuracy' in df.columns:
            ax_acc.plot(epochs, df['target_accuracy'], label=f'{label} tgt',
                        linewidth=1, linestyle='--')
        if 'avg_target_accuracy' in df.columns:
            ax_acc.plot(epochs, df['avg_target_accuracy'], label=f'{label} avg_tgt',
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

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Accuracy (%)')
    ax_acc.set_title('Training Accuracy Curves')
    ax_acc.legend(fontsize=7, loc='best')
    ax_acc.grid(alpha=0.3)
    path = os.path.join(output_dir, 'training_accuracy_curves.png')
    fig_acc.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig_acc)
    print(f"  Saved {path}")


# ==============================================================================
# Build test loaders for all devices
# ==============================================================================
def build_all_test_loaders(data_path, src_device='a', target_sr=TARGET_SAMPLE_RATE,
                           batch_size=64):
    """Build test DataLoaders for source + all target devices.

    Calls data_split once (test_tgt_df already contains all devices) and
    returns a dict mapping device_name -> DataLoader for the source and
    each device in ALL_TARGET_DEVICES.
    """
    _, _, _, _, _, test_src_df, test_tgt_df = data_split(
        src_device, ALL_TARGET_DEVICES[0], data_path=data_path
    )
    loaders = {}

    # Source
    src_test_ds = SimpleAudioDataset(
        file_df=test_src_df[src_device], root=data_path, target_sr=target_sr
    )
    loaders[src_device] = DataLoader(
        src_test_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    # Targets
    for tgt_dev in ALL_TARGET_DEVICES:
        tgt_test_ds = SimpleAudioDataset(
            file_df=test_tgt_df[tgt_dev], root=data_path, target_sr=target_sr
        )
        loaders[tgt_dev] = DataLoader(
            tgt_test_ds, batch_size=batch_size, shuffle=False, num_workers=4
        )

    return loaders


# ==============================================================================
# Programmatic API — called from training scripts after training
# ==============================================================================
def run_visualization(method, mfdwc_extractor, feature_extractor, classifier,
                      test_loaders, experiment_dir, csv_path=None, data_path=None,
                      src_device='a', batch_size=64):
    """Generate all visualizations using already-loaded models and data loaders.

    Called automatically at the end of training scripts — no checkpoint reloading.

    Args:
        method: 'baseline', 'grl', or 'cst'
        mfdwc_extractor: MFDWCFeatureExtractor (already on device)
        feature_extractor: Feature_extractor (already on device)
        classifier: Classifier_no_weights or CST_Classifier (already on device)
        test_loaders: dict mapping device_name -> DataLoader
                      e.g. {'a': src_test_loader, 'b': tgt_test_loader}
        experiment_dir: path to experiment output directory
        csv_path: path to training CSV log file (optional, for training curves)
        data_path: path to TAU dataset root (optional). When provided and
                   test_loaders does not cover all devices, missing device
                   loaders are built automatically so that plots contain
                   all devices.
        src_device: source device name (default: 'a')
        batch_size: batch size for any auto-built loaders (default: 64)
    """
    device = next(feature_extractor.parameters()).device
    output_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    # Auto-build loaders for missing devices when data_path is provided
    if data_path:
        all_devices = [src_device] + ALL_TARGET_DEVICES
        missing = [d for d in all_devices if d not in test_loaders]
        if missing:
            print(f"Building test loaders for missing devices: {missing}")
            all_loaders = build_all_test_loaders(
                data_path, src_device=src_device,
                batch_size=batch_size
            )
            for d in missing:
                test_loaders[d] = all_loaders[d]

    print(f"\n{'='*60}")
    print(f"Generating post-training visualizations...")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # Extract features and predictions for all devices
    print("\nExtracting features and predictions...")
    results = {}
    for dev_name, loader in test_loaders.items():
        print(f"  Processing device '{dev_name}'...")
        results[dev_name] = extract_features_and_predictions(
            mfdwc_extractor, feature_extractor, classifier, loader, device, method=method
        )
        n = len(results[dev_name]['true_labels'])
        acc = 100.0 * (results[dev_name]['predictions'] == results[dev_name]['true_labels']).mean()
        print(f"    {n} samples, accuracy: {acc:.2f}%")

    # Overall accuracy summary
    all_true = np.concatenate([r['true_labels'] for r in results.values()])
    all_pred = np.concatenate([r['predictions'] for r in results.values()])
    print(f"\n  Overall accuracy ({len(all_true)} samples): "
          f"{100.0 * (all_pred == all_true).mean():.2f}%")

    tgt_results = {k: v for k, v in results.items() if k != src_device}
    if tgt_results:
        tgt_true = np.concatenate([r['true_labels'] for r in tgt_results.values()])
        tgt_pred = np.concatenate([r['predictions'] for r in tgt_results.values()])
        print(f"  Target-only accuracy ({len(tgt_true)} samples): "
              f"{100.0 * (tgt_pred == tgt_true).mean():.2f}%")

    # t-SNE (CNN features)
    print("\nGenerating t-SNE plots (CNN features)...")
    plot_tsne(results, output_dir, tag='')

    # CST: extra t-SNE on bottleneck embeddings
    if method == 'cst' and any(r['embeddings'] is not None for r in results.values()):
        print("Generating t-SNE plots (CST bottleneck embeddings)...")
        plot_tsne(results, output_dir, tag='embeddings')

    # Confusion matrices
    print("\nGenerating confusion matrices...")
    for dev_name, res in results.items():
        plot_confusion_matrix(res['true_labels'], res['predictions'], dev_name, output_dir)

    # Per-class accuracy bar chart
    print("\nGenerating per-class accuracy chart...")
    plot_per_class_accuracy(results, output_dir)

    # Classification reports
    print("\nGenerating classification reports...")
    for dev_name, res in results.items():
        save_classification_report(res['true_labels'], res['predictions'], dev_name, output_dir)

    # Training curves from CSV
    if csv_path and os.path.isfile(csv_path):
        csv_dir = os.path.dirname(csv_path)
        print(f"\nPlotting training curves from {csv_dir}...")
        plot_training_curves(csv_dir, output_dir, method=method)

    print(f"\nAll visualizations saved to {output_dir}")


# ==============================================================================
# Main CLI
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Post-training visualization for UDA experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--method', type=str, choices=['baseline', 'grl', 'cst'],
                        help='Training method (determines classifier type)')
    parser.add_argument('--checkpoint-dir', type=str,
                        help='Path to experiment directory with best_FE_*.pth and best_CL_*.pth')
    parser.add_argument('--data-path', type=str,
                        help='Path to TAU dataset root')
    parser.add_argument('--src-device', type=str, default='a',
                        help='Source device (default: a)')
    parser.add_argument('--devices', type=str, default='b,c,s1,s2,s3',
                        help='Comma-separated target devices to evaluate (default: b,c,s1,s2,s3)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference (default: 64)')
    parser.add_argument('--csv-dir', type=str, default=None,
                        help='Directory with CSV training logs (for --plot-curves-only or auto-detected)')
    parser.add_argument('--plot-curves-only', action='store_true',
                        help='Only plot training curves from CSVs (no model loading)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: {checkpoint-dir}/plots/)')
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.checkpoint_dir:
        output_dir = os.path.join(args.checkpoint_dir, 'plots')
    elif args.csv_dir:
        output_dir = os.path.join(args.csv_dir, 'plots')
    else:
        output_dir = './plots'
    os.makedirs(output_dir, exist_ok=True)

    # --- Curves-only mode ---
    if args.plot_curves_only:
        csv_dir = args.csv_dir
        if not csv_dir:
            print("Error: --csv-dir is required with --plot-curves-only")
            sys.exit(1)
        print(f"Plotting training curves from {csv_dir}...")
        plot_training_curves(csv_dir, output_dir, method=args.method)
        print("Done.")
        return

    # --- Full evaluation mode ---
    if not all([args.method, args.checkpoint_dir, args.data_path]):
        print("Error: --method, --checkpoint-dir, and --data-path are required "
              "for full evaluation mode.")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tgt_devices = [d.strip() for d in args.devices.split(',')]
    print(f"Method: {args.method}")
    print(f"Source: {args.src_device}, Targets: {tgt_devices}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    # Load models
    mfdwc_ext, feat_ext, classifier, flattened_size = load_models(
        args.method, args.checkpoint_dir, device
    )

    # Build test dataloaders for source + each target
    _, _, _, _, _, test_src_df, test_tgt_df = data_split(
        args.src_device, tgt_devices[0], data_path=args.data_path
    )
    loaders = {}

    # Source test loader
    src_test_ds = SimpleAudioDataset(
        file_df=test_src_df[args.src_device],
        root=args.data_path, target_sr=TARGET_SAMPLE_RATE
    )
    loaders[args.src_device] = DataLoader(
        src_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Target test loaders (test_tgt_df already has all devices)
    for tgt_dev in tgt_devices:
        tgt_test_ds = SimpleAudioDataset(
            file_df=test_tgt_df[tgt_dev],
            root=args.data_path, target_sr=TARGET_SAMPLE_RATE
        )
        loaders[tgt_dev] = DataLoader(
            tgt_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

    # Extract features and predictions for all devices
    print("\nExtracting features and predictions...")
    results = {}
    for dev_name, loader in loaders.items():
        print(f"  Processing device '{dev_name}'...")
        results[dev_name] = extract_features_and_predictions(
            mfdwc_ext, feat_ext, classifier, loader, device, method=args.method
        )
        n = len(results[dev_name]['true_labels'])
        acc = 100.0 * (results[dev_name]['predictions'] == results[dev_name]['true_labels']).mean()
        print(f"    {n} samples, accuracy: {acc:.2f}%")

    # Overall accuracy summary
    all_true = np.concatenate([r['true_labels'] for r in results.values()])
    all_pred = np.concatenate([r['predictions'] for r in results.values()])
    overall_acc = 100.0 * (all_pred == all_true).mean()
    print(f"\n  Overall accuracy ({len(all_true)} samples): {overall_acc:.2f}%")

    tgt_results = {k: v for k, v in results.items() if k != args.src_device}
    if tgt_results:
        tgt_true = np.concatenate([r['true_labels'] for r in tgt_results.values()])
        tgt_pred = np.concatenate([r['predictions'] for r in tgt_results.values()])
        tgt_acc = 100.0 * (tgt_pred == tgt_true).mean()
        print(f"  Target-only accuracy ({len(tgt_true)} samples): {tgt_acc:.2f}%")

    # --- Generate plots ---
    print("\nGenerating t-SNE plots (CNN features)...")
    plot_tsne(results, output_dir, tag='')

    # CST: extra t-SNE on bottleneck embeddings
    if args.method == 'cst' and any(r['embeddings'] is not None for r in results.values()):
        print("Generating t-SNE plots (CST bottleneck embeddings)...")
        plot_tsne(results, output_dir, tag='embeddings')

    # Confusion matrices (one per device)
    print("\nGenerating confusion matrices...")
    for dev_name, res in results.items():
        plot_confusion_matrix(res['true_labels'], res['predictions'], dev_name, output_dir)

    # Per-class accuracy bar chart
    print("\nGenerating per-class accuracy chart...")
    plot_per_class_accuracy(results, output_dir)

    # Classification reports
    print("\nGenerating classification reports...")
    for dev_name, res in results.items():
        save_classification_report(res['true_labels'], res['predictions'], dev_name, output_dir)

    # Training curves from CSV logs
    csv_dir = args.csv_dir
    if not csv_dir:
        # Try to auto-detect CSVs in checkpoint dir
        csv_candidates = glob.glob(os.path.join(args.checkpoint_dir, '*.csv'))
        if csv_candidates:
            csv_dir = args.checkpoint_dir
        else:
            # Check parent for csv_logs subdir
            parent = os.path.dirname(args.checkpoint_dir)
            csv_logs_dir = os.path.join(parent, 'csv_logs')
            if os.path.isdir(csv_logs_dir):
                csv_dir = csv_logs_dir

    if csv_dir:
        print(f"\nPlotting training curves from {csv_dir}...")
        plot_training_curves(csv_dir, output_dir, method=args.method)

    print(f"\nAll outputs saved to {output_dir}")
    print("Done.")


if __name__ == '__main__':
    main()
