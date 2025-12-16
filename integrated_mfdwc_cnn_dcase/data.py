import os
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

PATH = '/DATA/G3/Datasets/archive/Original_split/TAU-urban-acoustic-scenes-2020-mobile-development'


label_keys = {'airport': 0, 'shopping_mall': 1, 'park': 2, 'street_pedestrian': 3,
              'street_traffic': 4, 'metro_station': 5, 'public_square': 6, 'metro': 7, 'bus': 8,
              'tram': 9}

def data_split( src_device , tgt_device , data_path=PATH):
    df = pd.read_csv(os.path.join(data_path, 'meta.csv'), sep='\t')
    evaluate_df = pd.read_csv(os.path.join(data_path, 'fold1_evaluate.csv'), sep='\t')
    train_df = pd.read_csv(os.path.join(data_path, 'fold1_train.csv'), sep='\t')
    df['file_number'] = df['filename'].apply(lambda x: x.split('-')[2] + x.split('-')[3])

    train_with_meta = pd.merge(train_df, df[['filename', 'scene_label', 'source_label', 'file_number']], on=['filename', 'scene_label'], how='left')
    evaluate_with_meta = pd.merge(evaluate_df, df[['filename', 'scene_label', 'source_label', 'file_number']], on=['filename', 'scene_label'], how='left')

    train_src = ['a']
    train_tgt = ['b', 'c', 's1', 's2', 's3']
    test_src = ['a']
    test_tgt = ['b', 'c', 's1', 's2', 's3', 's4', 's5', 's6']

    train_src_df = {d: train_with_meta[train_with_meta['source_label'] == d].copy() for d in train_src}
    train_tgt_df = {d: train_with_meta[train_with_meta['source_label'] == d].copy() for d in train_tgt}
    test_src_df = {d: evaluate_with_meta[evaluate_with_meta['source_label'] == d].copy() for d in test_src}
    test_tgt_df = {d: evaluate_with_meta[evaluate_with_meta['source_label'] == d].copy() for d in test_tgt}

    src_devices_mixup = []
    target_devices_mixup = []
    label_mix_up = []
    filtered_df = train_with_meta[train_with_meta['source_label'].isin([src_device, tgt_device])]
    for _, group in filtered_df.groupby('file_number'):
        if len(group) > 1:
            label_mix_up.append(group['scene_label'].iloc[0])
            for _, row in group.iterrows():
                if row['source_label'] == src_device:
                    src_devices_mixup.append(row['filename'])
                elif row['source_label'] == tgt_device:
                    target_devices_mixup.append(row['filename'])

    return src_devices_mixup, target_devices_mixup, label_mix_up, train_src_df, train_tgt_df, test_src_df, test_tgt_df

class SimpleAudioDataset(Dataset):
    """
    Dataset for loading audio waveforms for MFDWC feature extraction.
    
    Args:
        root: Root directory containing audio files
        file_df: DataFrame with filenames and labels
        target_sr: Target sample rate for MFDWC extraction (default: 44100 Hz)
    """
    def __init__(self, root, file_df, target_sr=44100):
        self.file_paths = file_df
        self.target_sr = target_sr
        self.scene_labels = []
        self.path = []
        
        for _, pd_dict in file_df.iterrows():
            path = os.path.join(root, pd_dict['filename'])
            self.path.append(path)
            self.scene_labels.append(pd_dict['scene_label'])

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        audio_path = self.path[idx]
        try:
            # Load raw waveform for MFDWC feature extraction
            waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            # Return as 1D tensor
            tensor = torch.tensor(waveform, dtype=torch.float32)
            return tensor, self.scene_labels[idx]
        except Exception:
            # Return 10 seconds of silence on error
            return torch.zeros(self.target_sr * 10, dtype=torch.float32), "error"

def create_combined_loader(source_loader, target_loader):
    source_iter = iter(source_loader)
    target_iter = iter(target_loader)

    for _ in range(len(source_loader)):
        src_data = next(source_iter)
        try:
            tgt_data = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            tgt_data = next(target_iter)
        yield src_data, tgt_data

def label_to_numerical(label_list, label_keys=label_keys):
    label_list_new = [label_keys.get(i, -1) for i in label_list]
    label_list_new = np.array(label_list_new)
    return torch.tensor(label_list_new) 