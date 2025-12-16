import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import pandas as pd
import librosa
from hear21passt.base import get_basic_model
import ipdb
import data 
from data import Passt_with_mix , label_to_numerical, SimpleAudioDataset , create_combined_loader
from datetime import datetime

import models
from models import Feature_extractor as FeatureExtractor 
from models import Classifier_no_weights as Classifier 

# --- CHANGED: Configuration and Hyperparameters updated for new data loading logic ---
METHOD = 'cnn' # 'CDAN' or 'CDAN-E'
# Path to the directory where the audio subfolders are located
PATH = './TAU-urban-acoustic-scenes-2020-mobile-development'
output_csv_path = 'training_results_dcase_cnn.csv'

# Define the specific source and target devices to be used from the datasets
src_device = 'a'
tgt_device = 'b'
TARGET_SAMPLE_RATE = 32000
NUM_CLASS = 10 # Will be updated dynamically by the data loader
USE_GPU = True
BATCH_SIZE = 128
NUM_EPOCHS = 200
LR = 0.001
RANDOM = True # Use random layer in CDAN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = './results/'
os.makedirs(save_dir , exist_ok=True)
import warnings
warnings.filterwarnings("ignore")



# --- Train and Test Functions ---
def test(feature_extractor, classifier, dataloader, device):
    """Evaluation loop."""
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
            features = feature_extractor(data)
            outputs = classifier(features)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    if total == 0:
        return 0.0
    accuracy = 100 * correct / total
    return accuracy


# --- NEW: Function to check for exploding gradients ---
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
    src_devices_mixup , target_devices_mixup, label_mix_up , train_src_df, train_tgt_df, test_src_df, test_tgt_df  = Passt_with_mix(src_device , tgt_device)

    print(f"Number of classes detected: {NUM_CLASS}")
    
    src_dataset = SimpleAudioDataset(file_df=train_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    tgt_dataset = SimpleAudioDataset(file_df=train_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_loader = DataLoader(tgt_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    src_test_dataset = SimpleAudioDataset(file_df=test_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_test_loader = DataLoader(src_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    tgt_test_dataset = SimpleAudioDataset(file_df=test_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Models for CNN backbone 
    

    feature_dim = 384
    feature_extractor = FeatureExtractor().to(device)
    classifier = Classifier().to(device)


    # Optimizers
    F_opt = optim.Adam(feature_extractor.parameters(), lr=LR)
    C_opt = torch.optim.Adam(classifier.parameters(), lr=LR)

    criterion_cls = nn.CrossEntropyLoss()
    
    # --- NEW: Setup for logging results ---
    results_log = []


    max_iter = NUM_EPOCHS * min(len(src_loader), len(tgt_loader))
    iter_num = 0
    num_batches = min(len(src_loader), len(tgt_loader))

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        feature_extractor.train()
        classifier.train()
        # discriminator.train()

        # --- NEW: Reset epoch-level trackers ---
        total_cls_loss, total_adv_loss, total_entropy_loss = 0, 0, 0
        total_f_grad_norm, total_c_grad_norm, total_d_grad_norm = 0, 0, 0
        
        for batch_idx, (src_data, tgt_data) in enumerate(create_combined_loader(src_loader, tgt_loader)):
            
            src_wave_form, src_label = src_data
            tgt_wave_form, _ = tgt_data # Target labels are not used in unsupervised DA
            print(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches}" ,end=",")
            # --- Skip batch if data loading failed ---
            if "error" in src_label:
                print(f"Skipping batch {batch_idx+1} due to data loading error in source.")
                continue

            src_label = label_to_numerical(src_label )
            src = src_wave_form.to(DEVICE)
            labels = src_label.to(DEVICE)
            tgt = tgt_wave_form.to(DEVICE)
    
            current_batch_size = min(len(src), len(tgt))
            if current_batch_size == 0: continue

            src, labels, tgt = src[:current_batch_size], labels[:current_batch_size], tgt[:current_batch_size]
            
            F_opt.zero_grad()
            C_opt.zero_grad()
            # D_opt.zero_grad()

            feat_source = feature_extractor(src)
            pred_source = classifier(feat_source)
            feat_target = feature_extractor(tgt)
            pred_target = classifier(feat_target)

            cls_loss = criterion_cls(pred_source, labels)

        
            
            total_loss = cls_loss 
            total_loss.backward()

            # --- NEW: Check gradient norms after backward pass ---
            total_f_grad_norm += check_gradient_norm(feature_extractor)
            total_c_grad_norm += check_gradient_norm(classifier)

            F_opt.step()
            C_opt.step()
            # D_opt.step()

            iter_num += 1
            total_cls_loss += cls_loss.item()
        timestamp = datetime.now()
        
        # --- NEW: Epoch-end logging and evaluation ---
        avg_cls_loss = total_cls_loss / num_batches
        avg_adv_loss = total_adv_loss / num_batches
        avg_ent_loss = total_entropy_loss / num_batches if METHOD == 'CDAN-E' else 0.0

        avg_f_grad = total_f_grad_norm / num_batches
        avg_c_grad = total_c_grad_norm / num_batches
        avg_d_grad = total_d_grad_norm / num_batches

        source_acc = test(feature_extractor, classifier, src_test_loader, device)
        target_acc = test(feature_extractor, classifier, tgt_test_loader, device)

        print(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}] | "
            f"Cls Loss: {avg_cls_loss:.4f}"
            f"Src Acc: {source_acc:.2f}%, Tgt Acc: {target_acc:.2f}%"
        )
        print(
            f"  -> Avg Grad Norms | Feature Extractor: {avg_f_grad:.4f}, "
            f"Classifier: {avg_c_grad:.4f}, Discriminator: {avg_d_grad:.4f}\n"
        )
        feature_extractor.state_dict()
        # Append results to log
        epoch_results = {
            'epoch': epoch,
            'classification_loss': avg_cls_loss,
            'adversarial_loss': avg_adv_loss,
            'entropy_loss': avg_ent_loss,
            'source_accuracy': source_acc,
            'target_accuracy': target_acc,
            'feature_extractor_grad_norm': avg_f_grad,
            'classifier_grad_norm': avg_c_grad,
            'discriminator_grad_norm': avg_d_grad
        }
        if epoch % 10 == 0 : 
            results_log.append(epoch_results)
            torch.save(feature_extractor.state_dict(), f"{save_dir}/FE_{timestamp}.pth")
            torch.save(classifier.state_dict() , f"{save_dir}/CL_{timestamp}.pth")
            # Save to CSV after each epoch
            pd.DataFrame(results_log).to_csv(output_csv_path, index=False)


    print(f"Training finished. Results saved to {output_csv_path}")


if __name__ == '__main__':
    train()
