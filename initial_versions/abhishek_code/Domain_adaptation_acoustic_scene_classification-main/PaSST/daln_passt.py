import torch
import torch.nn as nn
import numpy as np
import datetime
import os
import librosa
import pandas as pd
from hear21passt.base import get_basic_model
from torch.utils.data import Dataset, DataLoader
from DALN.nwd import NuclearWassersteinDiscrepancy
from torch.utils.data import Dataset, DataLoader
import librosa
import gc
import ipdb


MODEL_NAME = 'DANN'
print("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PaSSTFeatureExtractor(torch.nn.Module):
    def __init__(self, device=None):
        super(PaSSTFeatureExtractor, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_basic_model(mode="embed_only") 
        self.model.to(self.device)

    def forward(self, audio_waveform, sample_rate=44000):
        if audio_waveform.dim() == 1:
            audio_waveform = audio_waveform.unsqueeze(0)  

        audio_waveform = audio_waveform.to(self.device)
        
        # Allow gradients to flow through PaSST for domain adaptation
        features = self.model(audio_waveform)
             
        return features      

class Classifier(nn.Module):
    """Simplified Classifier"""
    def __init__(self, input_size=768, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, h):
        return self.layer(h)
        
class Discriminator(nn.Module):
    """Simplified Discriminator"""
    def __init__(self, input_size=768):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, h):
        return self.layer(h)
        

src_device = 'a'
tgt_device = 's3'
PATH = './TAU-urban-acoustic-scenes-2020-mobile-development'

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
    # original dcase dataset
    # train_domain = ['a' , 'b', 'c', 's1', 's2', 's3']
    # test_domain = ['a', 'b', 'c', 's1', 's2', 's3', 's4', 's5', 's6' ]

    print("Source Length" , len(train_with_meta[train_with_meta['source_label'] == src_device ]))
    print("Target Length" , len(train_with_meta[train_with_meta['source_label'] == tgt_device ]))

    train_src = ['a']
    train_tgt = ['b', 'c', 's1', 's2', 's3']
    test_src = ['a']
    test_tgt = ['b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']

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

src_devices_mixup , target_devices_mixup, label_mix_up , train_src_df, train_tgt_df, test_src_df, test_tgt_df  = Passt_with_mix()

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
            return torch.zeros(self.target_sr, dtype=torch.float32)
# return waveform, label
TARGET_SAMPLE_RATE = 32000
src_dataset = SimpleAudioDataset(file_df=train_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
src_loader = DataLoader(src_dataset, batch_size=4, shuffle=True, num_workers=2)

tgt_dataset = SimpleAudioDataset(file_df=train_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
tgt_loader = DataLoader(tgt_dataset, batch_size=4, shuffle=True, num_workers=2)

src_test_dataset = SimpleAudioDataset(file_df=test_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
src_test_loader = DataLoader(src_test_dataset, batch_size=4, shuffle=True, num_workers=2)

tgt_test_dataset = SimpleAudioDataset(file_df=test_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=4, shuffle=True, num_workers=2)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
num_classes = 10
# Data loader
source_loader = src_loader
target_loader = tgt_loader
eval_loader = src_test_loader
test_loader = tgt_test_loader


# Model initialization
F = PaSSTFeatureExtractor().to(DEVICE)  
C = Classifier(num_classes=num_classes).to(DEVICE)
discrepancy = NuclearWassersteinDiscrepancy(C) 

# D = Discriminator().to(DEVICE)


# Loss functions
# bce = nn.BCELoss()
xe = nn.CrossEntropyLoss()

# Optimizer
F_opt = torch.optim.Adam(F.parameters(), lr=1e-5) 
C_opt = torch.optim.Adam(C.parameters(), lr=1e-3)
# D_opt = torch.optim.Adam(D.parameters(), lr=1e-3)


max_epoch = 20  
step = 0


def create_combined_loader():
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

def get_lambda(epoch, max_epoch):
    
    p = float(epoch) / float(max_epoch)  
    return 2. / (1. + np.exp(-10. * p)) - 1.


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


def evaluate_model(data_loader, name="", max_batches=50):
    F.eval()
    C.eval()
    corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for idx, (data, labels) in enumerate(data_loader):
            if idx >= max_batches:  
                break
            labels = label_to_numerical(labels)
            data, labels = data.to(DEVICE), labels.to(DEVICE)
                    # labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

            features = F(data)
            outputs = C(features)
            _, preds = torch.max(outputs, 1)
            
            corrects += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = corrects / total_samples if total_samples > 0 else 0
    print(f'***** {name} Result: {accuracy:.4f} (based on {total_samples} samples)')
    return accuracy

# Training loop
print(f"Starting training for {max_epoch} epochs...")
print("="*60)

clear_memory()

steps_per_epoch = min(len(source_loader), len(target_loader))
print(f"Steps per epoch: {steps_per_epoch}")
with ipdb.launch_ipdb_on_exception():
    for epoch in range(1, max_epoch + 1):
        print(f"\nEpoch {epoch}/{max_epoch}")
        epoch_start_time = datetime.datetime.now()
        
       
        current_lambda = get_lambda(epoch - 1, max_epoch)  
    
        for a  in  enumerate(create_combined_loader()):
            batch_idx = a[0]
            src_obj = a[1][0]
            src_wave_form  = src_obj[0]
            src_label = src_obj[1]
            tgt_obj = a[1][1]
            tgt_wave_form = tgt_obj[0]
            tgt_label = tgt_obj[1]
    
            print(f"\nEpoch {epoch}/{max_epoch} Batch : {batch_idx}")
            
    
            src_label = label_to_numerical(src_label , label_keys)
            src = src_wave_form.to(DEVICE)
            labels = src_label.to(DEVICE)
            # Converting the labels to one hot encoded vectors 
            labels = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            tgt = tgt_wave_form.to(DEVICE)
    
                
            current_batch_size = min(tgt.size(0), src.size(0))
            tgt = tgt[:current_batch_size]
            src = src[:current_batch_size]
            labels = labels[:current_batch_size]
            D_src = torch.ones(current_batch_size, 1).to(DEVICE)
            D_tgt = torch.zeros(current_batch_size, 1).to(DEVICE)
            D_labels = torch.cat([D_src, D_tgt], dim=0)
    
               
            src_features = F(src)
            tgt_features = F(tgt)
            h = torch.cat([src_features, tgt_features], dim=0)
            discrepancy_loss = -discrepancy(h)

                
            # D.zero_grad()
            # y = D(h.detach())
            # Ld = bce(y, D_labels)
            # Ld.backward()
            # D_opt.step()
    
              
            C.zero_grad()
            F.zero_grad()
            c = C(src_features)
            # y = D(h)
            Lc = xe(c, labels)
            # Ld_adv = bce(y, D_labels)
            Ld = discrepancy_loss
            lamda = get_lambda(epoch, max_epoch)
            Ltot = Lc + lamda*Ld
               
            Ltot.backward()
            C_opt.step()
            F_opt.step()  
    
                
            if step % 100 == 0:
                dt = datetime.datetime.now().strftime('%H:%M:%S')
                print(f'Epoch: {epoch}/{max_epoch}, Step: {step}, Batch: {batch_idx+1}/{steps_per_epoch}, '
                      f'D Loss: {Ld.item():.4f}, C Loss: {Lc.item():.4f}, '
                      f'lambda: {current_lambda:.4f} ---- {dt}')
                
    
                # src_acc = evaluate_model(eval_loader, f"Source (Epoch {epoch})")
                # tgt_acc = evaluate_model(test_loader, f"Target (Epoch {epoch})")
    
            step += 1
                
           
            if step % 50 == 0:
                clear_memory()
                    
        
       
        print(f"\nEvaluation after 10 epoch {epoch}:")
        if  epoch % 10 == 0 :     
            src_acc = evaluate_model(eval_loader, f"Source (Epoch {epoch})")
            tgt_acc = evaluate_model(test_loader, f"Target (Epoch {epoch})")
            
      
        #if epoch == 1:
            print(f"Baseline accuracies - Source: {src_acc:.4f}, Target: {tgt_acc:.4f}, Test: {tgt_acc:.4f}")
        #else:
            print(f"Current lambda: {current_lambda:.4f}")
            print(f"Domain gap (Target - Source): {tgt_acc - src_acc:.4f}")
        
        
        F.train()
        C.train()
        
        epoch_time = datetime.datetime.now() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time}")
        print("-" * 50)
        
        clear_memory()
    
    
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    final_src_acc = evaluate_model(eval_loader, "Final Source", max_batches=50)
    final_test_acc = evaluate_model(test_loader, "Final Target", max_batches=50)
    
    print("Training completed!")
    print(f"Final test accuracy: {final_test_acc:.4f}")
    
    # Save models
    os.makedirs('./saved/models/' ,exist_ok=True)
    torch.save({
        'FE' : F.state_dict(),
        'classifier': C.state_dict(), 
        'final_accuracy': final_test_acc
    }, f'./saved/models/daln_model_final_acc_{tgt_device}_{final_test_acc:.4f}.pth')
    
    print("Model saved successfully!")
    clear_memory()








