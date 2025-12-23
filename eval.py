### Unmodified and not updated for current use


import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

import models
from models import Feature_extractor as FeatureExtractor 
from models import Classifier_no_weights as Classifier 
from models import Discriminator_no_weights as AdversarialNetwork 
import data

from data import Passt_with_mix, SimpleAudioDataset , label_to_numerical 

# --- CHANGED: Configuration and Hyperparameters updated for new data loading logic ---
PATH = './TAU-urban-acoustic-scenes-2020-mobile-development'
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
save_dir = './results'
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




def eval():
    tgt_devices = ['b' , 'c' , 's1' , 's2' ,  's3']

    """Main training and evaluation function."""
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data loading ---
    for i in tgt_devices:
        _ , _, _ , _, _, test_src_df, test_tgt_df  = Passt_with_mix(src_device = src_device , tgt_device=i)

        print(f"Number of classes detected: {NUM_CLASS}")
     

        src_test_dataset = SimpleAudioDataset(file_df=test_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
        src_test_loader = DataLoader(src_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        tgt_test_dataset = SimpleAudioDataset(file_df=test_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
        tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        feature_extractor = FeatureExtractor().to(device)
        classifier = Classifier().to(device)

        feature_extractor.load_state_dict(torch.load(f"{save_dir}/best_fe.pth"))
        classifier.load_state_dict(torch.load(f"{save_dir}/best_classifier.pth"))

        source_acc = test(feature_extractor, classifier, src_test_loader, device)
        target_acc = test(feature_extractor, classifier, tgt_test_loader, device)
        
        print(
            f"Src Acc: {source_acc:.2f}%, Tgt Acc: {target_acc:.2f}%"
        )
if __name__ == "__main__":
    eval()