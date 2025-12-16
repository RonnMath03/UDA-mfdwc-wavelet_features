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


# --- Define Source and Target Devices ---
src_device = 'a'
for dev in ['b' , 'c' , 's1' , 's2' , 's3']:
    tgt_device = dev

    # --- Setup Directories and Files for Results ---
    results_csv_path = f'./results_M2D_source_a-{tgt_device}.csv'
    weights_save_dir = f'./weights_M2D_source_per_epoch_{tgt_device}'
    os.makedirs(weights_save_dir, exist_ok=True)

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
    fe = fe.to('cuda')

    classifier = Classifier()
    classifier = classifier.to('cuda')

    disc = Discriminator()
    disc = disc.to('cuda')

    criterion =  nn.CrossEntropyLoss().to('cuda')
    bce = nn.BCELoss()
    loss_tt_epoch = []
    criterion = nn.CrossEntropyLoss()
    f_opt = optim.Adam(fe.parameters(), lr=0.0001)
    c_opt = optim.Adam(classifier.parameters(), lr=0.0001)
    d_opt = optim.Adam(disc.parameters(), lr=0.0001)

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        fe.train()
        classifier.train()
        total_cls_loss = 0
        src_acc = 0
        current_lambda = get_lambda(epoch - 1, NUM_EPOCHS + 1 )
        # --- Training Phase ---
        # Using only the source data for training as in the original script
        for batch_idx , (src_data , tgt_data)  in enumerate(create_combined_loader(src_loader, tgt_loader)):
            m_batch, m_label = src_data 
            m_label = label_to_numerical(m_label)
            t_batch, _ = tgt_data

            current_batch_size = min(t_batch.size(0), m_batch.size(0))
            m_batch = m_batch[:current_batch_size]
            t_batch = t_batch[:current_batch_size]
            m_label = m_label[: current_batch_size]
            D_src = torch.zeros(current_batch_size, 1 ).to(device)
            D_tgt = torch.ones(current_batch_size , 1 ).to(device)
            D_labels = torch.cat([D_src, D_tgt], dim=0)

            m_batch, m_label = m_batch.to(device), m_label.to(device)
            t_batch = t_batch.to(device)

            out_ASC = fe.encode_clap_audio(m_batch)
            out_tgt = fe.encode_clap_audio(t_batch)
            h = torch.cat([out_ASC , out_tgt] , dim = 0 )

            ## Solo Training of Discriminator
            disc.zero_grad()
            y = disc(h.detach())
            disc_loss = bce(y, D_labels)
            disc_loss.backward()
            d_opt.step()
            
            ## Joint training of Discriminator , Feature Extractor and Classifier 

            classifier.zero_grad()
            fe.zero_grad()

            pred_src = classifier(out_ASC)
            cls_loss = criterion(pred_src, m_label)
            y = disc(h)
            adv_loss = bce(y ,D_labels)

            
            Ltot = cls_loss - current_lambda * adv_loss
            Ltot.backward()
            
            f_opt.step()
            c_opt.step()
            
            total_cls_loss += Ltot.item()
            
            if (batch_idx + 1) % 1 == 0:
                print(f'Epoch [{epoch}/{NUM_EPOCHS}], Step [{batch_idx+1}], Loss: {Ltot.item():.4f}')
        
        avg_loss = total_cls_loss / len(src_loader)
        print(f"\n--- End of Epoch {epoch} ---")
        print(f"Average Training Loss: {avg_loss:.4f}")

        # ===================================================================
        #              NEW: EVALUATION & SAVING PER EPOCH
        # ===================================================================
        
        # --- Evaluation Phase ---
        fe.eval()
        classifier.eval()

        y_pred = []
        y_true = []
        with torch.no_grad():
            for m_batch, m_label in src_test_loader:
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


        print(f'Accuracy on Source ({src_device}) eval set: {src_accuracy:.2f} %')
        
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
            tgt_accuracy = (nb_cor / len(y_true)) * 100 


        print(f'Accuracy on target ({tgt_device}) eval set: {tgt_accuracy:.2f} %')
        # --- Logging Phase ---
        with open(results_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, src_device , f'{src_accuracy:.2f}', tgt_device , f'{src_accuracy:.2f}'])
            
        # --- Saving Weights Phase ---
        checkpoint_path = os.path.join(weights_save_dir, f'model_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'feature_extractor_state_dict': fe.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_fe_state_dict': f_opt.state_dict(),
            'optimizer_c_state_dict': c_opt.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}\n")
        # ===================================================================

    print('Finished Training and Evaluation.')
# model.eval();