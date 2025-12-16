# import warnings; warnings.simplefilter('ignore')
# import logging; logging.basicConfig(level=logging.INFO)
import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import zipfile
import librosa  
from utils import *
from portable_m2d import PortableM2D
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix



cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
PATH = '/DATA/G3/Datasets/archive/Original_split/TAU-urban-acoustic-scenes-2020-mobile-development'
evaluation_setup = 'evaluation_setup'
wav_file = os.path.join(PATH, 'audio')
meta_scp = os.path.join(PATH, 'meta.csv')
d_label_ASC_pth = os.path.join(PATH, evaluation_setup, 'd_label.pkl')
fold_trn = os.path.join(PATH, evaluation_setup, 'fold1_train.csv')
fold_evl = os.path.join(PATH, evaluation_setup, 'fold1_evaluate.csv')
verbose = 1 
bs_ASC = 16 
nb_worker = 24
nb_mels = 128 
nb_frames_ASC = 250
NUM_EPOCHS = 20
TARGET_SAMPLE_RATE = 16000
weights_save_dir = './weights_M2D_source_per_epoch'



# --- Define Source and Target Devices ---


print(f"Starting testing ")

    # ===================================================================
    #              NEW: EVALUATION & SAVING PER EPOCH
    # ===================================================================
    
    # --- Evaluation Phase ---]
epoch = 4
checkpoint_path = os.path.join(weights_save_dir, f'model_epoch_{epoch}.pt')
weight_file = torch.load(checkpoint_path)
# weight_file['epoch']

# weight_file['optimizer_fe_state_dict'],
# weight_file['optimizer_c_state_dict']
# weight_file['loss']
print(f"Saved checkpoint to {checkpoint_path}\n")

for dev in ['b' , 'c' , 's1' , 's2' , 's3']:
    src_device = 'a'
    tgt_device = dev

    # --- Setup Directories and Files for Results ---
    results_csv_path = f'./results_M2D_source_finetuned.csv'

    # --- Data Loading and Preprocessing ---

    src_devices_mixup , target_devices_mixup, label_mix_up , train_src_df, train_tgt_df, test_src_df, test_tgt_df  = Passt_with_mix(tgt_device , PATH , src_device)


    src_dataset = SimpleAudioDataset(file_df=train_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_loader = DataLoader(src_dataset, batch_size=bs_ASC, shuffle=True, num_workers=2)

    tgt_dataset = SimpleAudioDataset(file_df=train_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_loader = DataLoader(tgt_dataset, batch_size=bs_ASC, shuffle=True, num_workers=2)

    src_test_dataset = SimpleAudioDataset(file_df=test_src_df[src_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    src_test_loader = DataLoader(src_test_dataset, batch_size=bs_ASC, shuffle=True, num_workers=2)

    tgt_test_dataset = SimpleAudioDataset(file_df=test_tgt_df[tgt_device], root=PATH, target_sr=TARGET_SAMPLE_RATE)
    tgt_test_loader = DataLoader(tgt_test_dataset, batch_size=bs_ASC, shuffle=True, num_workers=2)



    fe = PortableM2D(weight_file='m2d_clap_vit_base-80x608p16x16-240128/checkpoint-300.pth', flat_features=True)
    fe.load_state_dict(weight_file['feature_extractor_state_dict'])
    fe = fe.to('cuda')

    classifier = Classifier()
    classifier.load_state_dict(weight_file['classifier_state_dict'])
    classifier = classifier.to('cuda')

    criterion =  nn.CrossEntropyLoss().to('cuda')
    bce = nn.BCELoss()
    loss_tt_epoch = []
    criterion = nn.CrossEntropyLoss()

    fe.eval()
    classifier.eval()

    y_pred = []
    y_true = []
    with torch.no_grad():
        for m_batch, m_label in tgt_test_loader:
            m_label = label_to_numerical(m_label)
            m_batch, m_label = m_batch.to(device), m_label.to(device)
           
            out = fe.encode_clap_audio(m_batch)
            batch_size = out.size(0)
            out = classifier(out)
            out = F.softmax(out, dim=-1).view(-1, batch_size, out.size(1))
            m_label = list(m_label.detach().cpu().numpy())
            y_pred.extend(list(out.cpu().numpy()))
            y_true.extend(m_label)
            
        y_pred = np.argmax(np.concat(y_pred), axis=1).tolist()
        conf_mat = confusion_matrix(y_true = y_true, y_pred = y_pred)
        nb_cor = np.trace(conf_mat)
        src_accuracy = (nb_cor / len(y_true)) * 100 


    print(f'Accuracy on Target ({tgt_device}) eval set: {src_accuracy:.2f} %')
    
    # --- Logging Phase ---
    with open(results_csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f'{src_accuracy:.2f}' ])
    
print('Finished testing and Evaluation.')
# model.eval();