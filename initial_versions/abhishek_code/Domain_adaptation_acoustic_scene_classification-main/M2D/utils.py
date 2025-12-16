import os
import numpy as np 
import pandas as pd 
import torch 
import torch.nn as nn
import librosa 
from torch.utils import data

class SimpleAudioDataset(data.Dataset):
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
            print('File-Error')
            return torch.zeros(self.target_sr, dtype=torch.float32)

class Discriminator(nn.Module):
    """Simplified Classifier"""
    def __init__(self, input_size=768, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, h):
        return self.layer(h)

def Passt_with_mix( tgt_device , data_path , src_device='a' ):
    #######################
    ### Load CSV files ####
    #######################
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

class Classifier(nn.Module):
    """Simplified Classifier"""
    def __init__(self, input_size=768, num_classes=10):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Linear(128, num_classes),
        )

    def forward(self, h):
        return self.layer(h)

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

def get_lambda(epoch, max_epoch):
    p = float(epoch) / float(max_epoch)  
    return 2. / (1. + np.exp(-10. * p)) - 1.


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
        # FIXED: Corrected typo from --1 to -1
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))

    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float()

    if next(ad_net.parameters()).is_cuda:
        dc_target = dc_target.to(next(ad_net.parameters()).device)

    return nn.BCEWithLogitsLoss()(ad_out, dc_target)
