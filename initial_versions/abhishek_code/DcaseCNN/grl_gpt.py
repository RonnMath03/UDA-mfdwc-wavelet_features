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
import warnings
warnings.filterwarnings('ignore')
import sys
print (sys.argv)

# --- CHANGED: Configuration updated for GRL method ---
METHOD = 'GRL' # Changed from 'CDAN-E' to 'GRL'
# Path to the directory where the audio subfolders are located
PATH = './TAU-urban-acoustic-scenes-2020-mobile-development'
# Define the specific source and target devices to be used from the datasets
src_device = 'a'
tgt_device = sys.argv[1] if len(sys.argv) > 1 else 'b'  # Target device can be specified via command line argument
print(f"Source Device: {src_device}, Target Device: {tgt_device}")
TARGET_SAMPLE_RATE = 32000
NUM_CLASS = 10 # Will be updated dynamically by the data loader
USE_GPU = True
BATCH_SIZE = 256
NUM_EPOCHS = 200
LR = 0.001
# RANDOM = True # No longer needed for GRL
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Data Preparation Function ---
def Passt_with_mix(data_path=PATH):
    #######################
    ### Load CSV files ####
    #######################
    global src_device , tgt_device
    df = pd.read_csv(os.path.join(data_path, 'meta.csv'),sep='\t')
    evaluate_df = pd.read_csv(os.path.join(data_path, 'fold1_evaluate.csv'), sep='\t')
    train_df = pd.read_csv(os.path.join(data_path, 'fold1_train.csv'), sep='\t')
    # test_df = pd.read_csv(os.path.join(data_path, 'fold1_test.csv') , sep='\t')
    df['file_number'] = df['filename'].apply(lambda x: x.split('-')[2]  +  x.split('-')[3])


    ##############################################
    ### Split according to Dcase train-test ######
    ##############################################

    train_with_meta = pd.merge(train_df, df[['filename', 'scene_label', 'source_label' , 'file_number']], on=['filename', 'scene_label'], how='left')
    evaluate_with_meta = pd.merge(evaluate_df, df[['filename', 'scene_label', 'source_label' , 'file_number']], on=['filename', 'scene_label'], how='left')

    train_with_meta = pd.merge(train_df, df[['filename', 'scene_label', 'source_label' , 'file_number']], on=['filename', 'scene_label'], how='left')
    evaluate_with_meta = pd.merge(evaluate_df, df[['filename', 'scene_label', 'source_label' , 'file_number']], on=['filename', 'scene_label'], how='left')
    #########################
    # Define domain groups ##
    #########################
    train_src = ['a']
    train_tgt = ['b', 'c', 's1', 's2', 's3']
    test_src = ['a']
    test_tgt = ['b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']

    print("Source Length" , len(train_with_meta[train_with_meta['source_label'] == src_device ]))
    print("Target Length" , len(train_with_meta[train_with_meta['source_label'] == tgt_device ]))

    ###########################################
    ### Create Source and Target Dataframes ###
    ###########################################


    train_src_df = {d: train_with_meta[train_with_meta['source_label'] == d].copy() for d in train_src}
    train_tgt_df = {d: train_with_meta[train_with_meta['source_label'] == d].copy() for d in train_tgt}

    test_src_df = {d: evaluate_with_meta[evaluate_with_meta['source_label'] == d].copy() for d in test_src}
    test_tgt_df = {d: evaluate_with_meta[evaluate_with_meta['source_label'] == d].copy() for d in test_tgt}

    ##############################################
    ## Mixed up devices that are paried segment ##
    ##############################################
    src_devices_mixup = []
    target_devices_mixup = []
    filtered_df = train_with_meta[train_with_meta['source_label'].isin([src_device, tgt_device])]
    label_list = []
    for i in filtered_df.groupby('file_number'):
        if len(i[1]) > 1 :
            label = i[1]['scene_label'].iloc[0]

            label_list.append(label)
            for j in i[1].iterrows() :
                device = j[1]['source_label']

                if device == src_device : ### Device that is source [use device in [src] or [tgt] ]
                    src_devices_mixup.append(j[1]['filename'])
                elif device == tgt_device:
                    target_devices_mixup.append(j[1]['filename'])
    print("Paired devices found: ", len(src_devices_mixup))

    return src_devices_mixup , target_devices_mixup , label_list , train_src_df, train_tgt_df, test_src_df, test_tgt_df

label_keys = { 'airport' : 0 , 'shopping_mall' : 1 , 'park' : 2, 'street_pedestrian' : 3,
       'street_traffic' : 4 , 'metro_station' : 5 , 'public_square' : 6 , 'metro' : 7, 'bus' : 8 ,
       'tram' : 9}

def label_to_numerical(label_list , label_keys=label_keys):
    label_list_new = []
    for i in label_list:
        label_list_new.append(label_keys[i])
    label_list_new = np.array(label_list_new)
    label_list_new = torch.tensor(label_list_new)
    return label_list_new

# --- Utility and Loss Functions ---
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    """Calculates the coefficient for GRL."""
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

# --- REMOVED: CDAN, Entropy, and RandomLayer are no longer needed for GRL ---

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

# --- Model Definitions ---
# NOTE: The FeatureExtractor, Classifier, and AdversarialNetwork classes
# are assumed to be correctly defined in their respective modules.
# The local definitions below are for reference and might be overridden by imports.

class FeatureExtractor(nn.Module):
    """Feature Extractor using PaSST."""
    def __init__(self, device):
        super(FeatureExtractor, self).__init__()
        if get_basic_model is None:
            raise ImportError("hear21passt is required for FeatureExtractor.")
        self.device = device
        self.model = get_basic_model(mode="embed_only")
        self.model.to(self.device)
        self.device = next(self.model.parameters()).device

    def forward(self, x):
        return self.model(x)

class Classifier(nn.Module):
    """Classifier head."""
    def __init__(self, input_size=768, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512 , 256 ),
            nn.ReLU(inplace=True),
            nn.Linear(256 , 128 ),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.layer(x)

class AdversarialNetwork(nn.Module):
    """Discriminator for GRL, predicts domain from features."""
    def __init__(self, input_size=384): # Input is now the feature dimension
        super(AdversarialNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1), # Outputs a single logit for domain classification
        )

    def forward(self, h):
        return self.layer(h)

# --- Dataset class remains the same ---
class SimpleAudioDataset(Dataset):
    def __init__(self, root , file_df, target_sr=32000):
        self.file_paths = file_df
        self.target_sr = target_sr
        self.scene_labels = []
        self.device_labels = []
        self.path = []
        for i in file_df.iterrows():
            idx = i[0]
            pd_dict = i[1]
            # Path of the audio file , joined to the root
            path = os.path.join(root , pd_dict['filename'])
            self.path.append(path)
            # scene Label
            self.scene_labels.append(pd_dict['scene_label'])
            # Device label
            self.device_labels.append(pd_dict['source_label'])

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        audio_path = self.path[idx]
        try:
            waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            tensor = torch.tensor(waveform, dtype=torch.float32)
            return tensor , self.scene_labels[idx]
        except Exception as e:
            # Return a zero tensor and a placeholder label if a file is corrupt
            print(f"Warning: Could not load file {audio_path}. Returning zeros. Error: {e}")
            return torch.zeros(self.target_sr * 10, dtype=torch.float32), "error"


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

def create_combined_loader(source_loader , target_loader):
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for _ in range(len(source_loader)):   # run for all source batches
        src_data = next(source_iter)
        try:
            tgt_data = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)  # reset target
            tgt_data = next(target_iter)
        yield src_data, tgt_data

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
    src_devices_mixup , target_devices_mixup, label_mix_up , train_src_df, train_tgt_df, test_src_df, test_tgt_df  = Passt_with_mix()

    print(f"Number of classes detected: {NUM_CLASS}")
    from data import SimpleAudioDataset
    
    src_dataset = SimpleAudioDataset(file_df=train_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    tgt_dataset = SimpleAudioDataset(file_df=train_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_loader = DataLoader(tgt_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    src_test_dataset = SimpleAudioDataset(file_df=test_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_test_loader = DataLoader(src_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    tgt_test_dataset = SimpleAudioDataset(file_df=test_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Models for CNN backbone 
    import models
    from models import Feature_extractor as FeatureExtractor 
    from models import Classifier_no_weights as Classifier 
    from models import Discriminator_no_weights as AdversarialNetwork 

    feature_dim = 384
    feature_extractor = FeatureExtractor().to(device)
    classifier = Classifier().to(device)
    # --- CHANGED: Discriminator for GRL takes feature_dim as input ---
    discriminator = AdversarialNetwork().to(device)


    # Optimizers
    F_opt = optim.Adam(feature_extractor.parameters(), lr=LR)
    C_opt = torch.optim.Adam(classifier.parameters(), lr=LR)
    D_opt = optim.Adam(discriminator.parameters(), lr=LR)

    criterion_cls = nn.CrossEntropyLoss()
    # --- NEW: Loss for domain discriminator ---
    criterion_adv = nn.BCEWithLogitsLoss()
    
    # --- CHANGED: Setup for logging results for GRL ---
    results_log = []
    output_csv_path = f'training_results_grl_dcase_cnn-{tgt_device}.csv'


    max_iter = NUM_EPOCHS * min(len(src_loader), len(tgt_loader))
    iter_num = 0
    num_batches = min(len(src_loader), len(tgt_loader))

    print("Starting training with GRL method...")
    for epoch in range(1, NUM_EPOCHS + 1):
        feature_extractor.train()
        classifier.train()
        discriminator.train()

        total_cls_loss, total_adv_loss = 0, 0
        total_f_grad_norm, total_c_grad_norm, total_d_grad_norm = 0, 0, 0
        
        for batch_idx, (src_data, tgt_data) in enumerate(create_combined_loader(src_loader, tgt_loader)):
            
            src_wave_form, src_label = src_data
            tgt_wave_form, _ = tgt_data # Target labels are not used in unsupervised DA
            print(f"Epoch {epoch}, Batch {batch_idx+1}/{num_batches}", end=",")

            if "error" in src_label:
                print(f"Skipping batch {batch_idx+1} due to data loading error in source.")
                continue

            src_label = label_to_numerical(src_label , label_keys)
            src = src_wave_form.to(DEVICE)
            labels = src_label.to(DEVICE)
            tgt = tgt_wave_form.to(DEVICE)
    
            current_batch_size = min(len(src), len(tgt))
            if current_batch_size == 0: continue

            src, labels, tgt = src[:current_batch_size], labels[:current_batch_size], tgt[:current_batch_size]
            
            F_opt.zero_grad()
            C_opt.zero_grad()
            D_opt.zero_grad()

            # --- FORWARD PASS ---
            feat_source = feature_extractor(src)
            pred_source = classifier(feat_source)
            feat_target = feature_extractor(tgt)

            # 1. Classification Loss (on source data)
            cls_loss = criterion_cls(pred_source, labels)

            # --- CHANGED: GRL Adversarial Loss Calculation ---
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
        
        # --- Epoch-end logging and evaluation ---
        avg_cls_loss = total_cls_loss / num_batches
        avg_adv_loss = total_adv_loss / num_batches
        
        avg_f_grad = total_f_grad_norm / num_batches
        avg_c_grad = total_c_grad_norm / num_batches
        avg_d_grad = total_d_grad_norm / num_batches

        source_acc = test(feature_extractor, classifier, src_test_loader, device)
        target_acc = test(feature_extractor, classifier, tgt_test_loader, device)

        print(
            f"\nEpoch [{epoch}/{NUM_EPOCHS}] | "
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

        # Save to CSV after each epoch
        pd.DataFrame(results_log).to_csv(output_csv_path, index=False)


    print(f"Training finished. Results saved to {output_csv_path}")


if __name__ == '__main__':
    train()