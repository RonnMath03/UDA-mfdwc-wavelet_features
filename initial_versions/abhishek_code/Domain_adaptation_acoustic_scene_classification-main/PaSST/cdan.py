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
import time # --- MODIFIED: Added for timestamping results directory

# --- CHANGED: Configuration and Hyperparameters updated for new data loading logic ---
METHOD = 'CDAN-E' # 'CDAN' or 'CDAN-E'
# Path to the directory where the audio subfolders are located
PATH = '/DATA/s23103/TAU-urban-acoustic-scenes-2020-mobile-development'
# Define the specific source and target devices to be used from the datasets
src_device = 'a'
tgt_device = 'b'
TARGET_SAMPLE_RATE = 32000
NUM_CLASS = 10 # Will be updated dynamically by the data loader
USE_GPU = True
BATCH_SIZE = 4
NUM_EPOCHS = 11
LR = 0.001
RANDOM = True # Use random layer in CDAN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODIFIED: Added configuration for output directory ---
RESULTS_DIR = f'./results/{METHOD}_{src_device}_to_{tgt_device}_{time.strftime("%Y%m%d-%H%M%S")}'


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

    #########################
    # Define domain groups ##
    #########################
    train_src_devices = ['a']
    train_tgt_devices = ['b', 'c', 's1', 's2', 's3']
    test_src_devices = ['a']
    test_tgt_devices = ['b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']

    print("Source Length" , len(train_with_meta[train_with_meta['source_label'] == src_device ]))
    print("Target Length" , len(train_with_meta[train_with_meta['source_label'] == tgt_device ]))

    ###########################################
    ### Create Source and Target Dataframes ###
    ###########################################

    train_src_df = {d: train_with_meta[train_with_meta['source_label'] == d].copy() for d in train_src_devices}
    train_tgt_df = {d: train_with_meta[train_with_meta['source_label'] == d].copy() for d in train_tgt_devices}

    test_src_df = {d: evaluate_with_meta[evaluate_with_meta['source_label'] == d].copy() for d in test_src_devices}
    test_tgt_df = {d: evaluate_with_meta[evaluate_with_meta['source_label'] == d].copy() for d in test_tgt_devices}

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
    # --- MODIFIED: Ensure tensor is created on the correct device later in the loop ---
    label_list_new = torch.tensor(label_list_new, dtype=torch.long)
    return label_list_new

# --- Utility and Loss Functions ---
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    """Calculates the coefficient for GRL."""
    return float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

def Entropy(input_):
    """Calculates entropy of the input."""
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def CDAN(input_list, ad_net, random_layer=None):
    """Simplified CDAN loss function without entropy hook."""
    softmax_output = input_list[1].detach()
    feature = input_list[0]

    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))

    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()

    if next(ad_net.parameters()).is_cuda:
        dc_target = dc_target.to(next(ad_net.parameters()).device)

    return nn.BCEWithLogitsLoss()(ad_out, dc_target)

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
class FeatureExtractor(nn.Module):
    """Feature Extractor using PaSST."""
    def __init__(self, device):
        super(FeatureExtractor, self).__init__()
        if get_basic_model is None:
            raise ImportError("hear21passt is required for FeatureExtractor.")
        self.model = get_basic_model(mode="embed_only")
        self.model.to(device)

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
    """Simplified Discriminator"""
    def __init__(self, input_size=768 , num_classes=1):
        super(AdversarialNetwork, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512 , 256 ),
            nn.ReLU(inplace=True),
            nn.Linear(256 , 128 ),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, h):
        return self.layer(h)
        
class RandomLayer(nn.Module):
    """Random Layer for CDAN with random features."""
    def __init__(self, input_dim_list, output_dim):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / torch.pow(torch.tensor(float(self.output_dim)), 1.0/len(return_list))
        for i in range(1, len(return_list)):
            return_tensor = torch.mul(return_tensor, return_list[i])
        return return_tensor

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.random_matrix = [matrix.to(device) for matrix in self.random_matrix]
        return self

# --- CHANGED: Dataset class updated to use the new loading logic ---
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
            # --- MODIFIED: Return a dummy tensor and a placeholder label on error ---
            print(f"Warning: Could not load {audio_path}. Error: {e}. Returning zeros.")
            return torch.zeros(self.target_sr * 10, dtype=torch.float32), "error_label"


# --- Train and Test Functions ---
def test(feature_extractor, classifier, dataloader, device):
    """Evaluation loop."""
    feature_extractor.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target_labels in dataloader:
            # --- MODIFIED: Filter out any samples that failed to load ---
            valid_indices = [i for i, label in enumerate(target_labels) if label != "error_label"]
            if not valid_indices:
                continue
            data = data[valid_indices]
            target_labels = [target_labels[i] for i in valid_indices]

            target = label_to_numerical(target_labels)
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

    # --- MODIFIED: Make combined loader length of the shorter loader to avoid incomplete batches ---
    num_batches = min(len(source_loader), len(target_loader))
    
    for _ in range(num_batches):
        src_data = next(source_iter)
        try:
            tgt_data = next(target_iter)
        except StopIteration:
            # This logic ensures the target loader resets if it's shorter than the source
            target_iter = iter(target_loader)
            tgt_data = next(target_iter)
        yield src_data, tgt_data

def train():
    """Main training and evaluation function."""
    device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- MODIFIED: Create results directory ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file_path = os.path.join(RESULTS_DIR, 'training_log.csv')
    print(f"Results and models will be saved in: {RESULTS_DIR}")

    # --- CHANGED: Data loading now uses the Passt_with_mix function ---
    src_devices_mixup , target_devices_mixup, label_mix_up , train_src_df, train_tgt_df, test_src_df, test_tgt_df  = Passt_with_mix()

    print(f"Number of classes: {NUM_CLASS}")

    src_dataset = SimpleAudioDataset(file_df=train_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_loader = DataLoader(src_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    tgt_dataset = SimpleAudioDataset(file_df=train_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_loader = DataLoader(tgt_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    src_test_dataset = SimpleAudioDataset(file_df=test_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_test_loader = DataLoader(src_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    tgt_test_dataset = SimpleAudioDataset(file_df=test_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Models
    feature_dim = 768
    feature_extractor = FeatureExtractor(device)
    classifier = Classifier(input_size=feature_dim, num_classes=NUM_CLASS).to(device)

    ad_net_input_dim = feature_dim * NUM_CLASS
    # ipdb.set_trace()
    if RANDOM:
        random_layer = RandomLayer([feature_dim, NUM_CLASS], 768).to(device)
        ad_net_input_dim = 768
    else:
        random_layer = None

    discriminator = AdversarialNetwork(input_size=ad_net_input_dim).to(device)

    # Optimizers
    optimizer_F_C = optim.Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=1e-5)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR)

    criterion_cls = nn.CrossEntropyLoss()

    max_iter = NUM_EPOCHS * min(len(src_loader), len(tgt_loader))
    iter_num = 0

    # --- MODIFIED: Setup for logging and saving best model ---
    best_target_acc = 0.0
    with open(log_file_path, 'w') as f:
        log_header = 'epoch,cls_loss,adv_loss'
        if METHOD == 'CDAN-E':
            log_header += ',entropy_loss'
        log_header += ',source_acc,target_acc\n'
        f.write(log_header)

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        feature_extractor.train()
        classifier.train()
        discriminator.train()

        total_cls_loss, total_adv_loss, total_entropy_loss = 0, 0, 0
        num_batches = min(len(src_loader), len(tgt_loader))

        combined_loader = create_combined_loader(src_loader, tgt_loader)

        for batch_idx, (src_batch, tgt_batch) in enumerate(combined_loader):
            src_wave_form, src_label_str = src_batch
            tgt_wave_form, _ = tgt_batch

            # --- MODIFIED: Filter out error samples from batch ---
            valid_src_indices = [i for i, label in enumerate(src_label_str) if label != "error_label"]
            if not valid_src_indices:
                continue
            src_wave_form = src_wave_form[valid_src_indices]
            src_label_str = [src_label_str[i] for i in valid_src_indices]

            if (batch_idx + 1) % 100 == 0 :
                print(f"Epoch: {epoch}/{NUM_EPOCHS}, Batch: {batch_idx+1}/{num_batches}")

            labels = label_to_numerical(src_label_str , label_keys)
            
            src = src_wave_form.to(DEVICE)
            labels = labels.to(DEVICE)
            tgt = tgt_wave_form.to(DEVICE)
    
            current_batch_size = min(tgt.size(0), src.size(0))
            if current_batch_size == 0: continue # Skip if batch is empty after filtering

            tgt = tgt[:current_batch_size]
            src = src[:current_batch_size]
            labels = labels[:current_batch_size]
            
            optimizer_F_C.zero_grad()
            optimizer_D.zero_grad()

            feat_source = feature_extractor(src)
            pred_source = classifier(feat_source)
            feat_target = feature_extractor(tgt)
            pred_target = classifier(feat_target)

            cls_loss = criterion_cls(pred_source, labels)

            features_combined = torch.cat((feat_source, feat_target), dim=0)
            pred_combined = torch.cat((pred_source, pred_target), dim=0)
            softmax_combined = nn.Softmax(dim=1)(pred_combined)

            coeff = calc_coeff(iter_num, max_iter=max_iter)
            reversed_features = GradReverse.apply(features_combined, coeff)

            # --- Inlined CDAN loss calculation ---
            softmax_output = softmax_combined.detach()
            feature = reversed_features
            ad_net = discriminator
            if random_layer is None:
                op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
                ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
            else:
                random_out = random_layer.forward([feature, softmax_output])
                ad_out = ad_net(random_out.view(-1, random_out.size(1)))

            # --- MODIFIED: Ensure batch size for target is correct after filtering ---
            effective_batch_size = labels.size(0) # Based on valid source samples
            dc_target_src = torch.ones(effective_batch_size, 1, device=DEVICE)
            dc_target_tgt = torch.zeros(feat_target.size(0), 1, device=DEVICE)
            dc_target = torch.cat((dc_target_src, dc_target_tgt), 0)

            adv_loss =  nn.BCEWithLogitsLoss()(ad_out, dc_target)

            total_loss = cls_loss + adv_loss
            if METHOD == 'CDAN-E':
                softmax_target = nn.Softmax(dim=1)(pred_target)
                entropy_loss = torch.mean(Entropy(softmax_target))
                total_loss += (entropy_loss * coeff)
                total_entropy_loss += entropy_loss.item()

            total_loss.backward()

            optimizer_F_C.step()
            optimizer_D.step()

            iter_num += 1
            total_cls_loss += cls_loss.item()
            total_adv_loss += adv_loss.item()
        
        # --- MODIFIED: Evaluation and logging at the end of each epoch ---
        avg_cls_loss = total_cls_loss / num_batches
        avg_adv_loss = total_adv_loss / num_batches
        
        log_str = f'Epoch [{epoch}/{NUM_EPOCHS}] -> Cls Loss: {avg_cls_loss:.4f}, Adv Loss: {avg_adv_loss:.4f}'
        
        log_data = f'{epoch},{avg_cls_loss:.4f},{avg_adv_loss:.4f}'
        
        if METHOD == 'CDAN-E':
            avg_ent_loss = total_entropy_loss / num_batches
            log_str += f', Entropy Loss: {avg_ent_loss:.4f}'
            log_data += f',{avg_ent_loss:.4f}'

        source_acc = test(feature_extractor, classifier, src_test_loader, device)
        target_acc = test(feature_extractor, classifier, tgt_test_loader, device)
        
        log_str += f' || Source Acc: {source_acc:.2f}%, Target Acc: {target_acc:.2f}%'
        log_data += f',{source_acc:.2f},{target_acc:.2f}\n'
        
        print(log_str)
        
        with open(log_file_path, 'a') as f:
            f.write(log_data)
        
        # Save the best model based on target accuracy
        if target_acc > best_target_acc:
            best_target_acc = target_acc
            print(f'   -> New best target accuracy: {best_target_acc:.2f}%. Saving model...')
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_F_C': optimizer_F_C.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'epoch': epoch,
            }, os.path.join(RESULTS_DIR, 'best_model.pth'))


    # --- MODIFIED: Save the final model ---
    print("Training finished. Saving final model.")
    torch.save({
        'feature_extractor': feature_extractor.state_dict(),
        'classifier': classifier.state_dict(),
        'discriminator': discriminator.state_dict(),
        'optimizer_F_C': optimizer_F_C.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
        'epoch': NUM_EPOCHS,
    }, os.path.join(RESULTS_DIR, 'final_model.pth'))


if __name__ == '__main__':
    train()
