import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Function
import pandas as pd
import ipdb
import models
from models import Feature_extractor as FeatureExtractor 
from models import Discriminator_bridge as AdversarialNetwork 
from models import cnn_GVB as Classifier
import loss 
import data
from data import Passt_with_mix , create_combined_loader , label_to_numerical , SimpleAudioDataset , label_keys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# --- CHANGED: Configuration and Hyperparameters updated for new data loading logic ---
METHOD = 'CDAN-E' # 'CDAN' or 'CDAN-E'
# Define the specific source and target devices to be used from the datasets
src_device = 'a'
tgt_device = 'b'
TARGET_SAMPLE_RATE = 32000
NUM_CLASS = 10 
USE_GPU = True
BATCH_SIZE =256
NUM_EPOCHS = 200
FE_LR = 1e-5
LR = 0.0001
trade_off = 1.0
RANDOM = True # Use random layer in CDAN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATH = '/DATA/s23103/TAU-urban-acoustic-scenes-2020-mobile-development'
GVBG = True
GVBD = True

torch.autograd.set_detect_anomaly(True)

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
            outputs,_ = classifier(features)
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
    DEVICE = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

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

    feature_dim = 384
    feature_extractor = FeatureExtractor().to(DEVICE)
    classifier = Classifier().to(DEVICE)
    # classifier = Classifier().to(DEVICE)
    # classifier_bridge = Bridge(input_dim=feature_dim).t(DEVICE)
    # ad_net_input_dim = feature_dim * NUM_CLASS

    # ipdb.set_trace()    
    if RANDOM:
        random_layer = RandomLayer([feature_dim, NUM_CLASS], 384).to(DEVICE)
        ad_net_input_dim = 384
    else:
        random_layer = None

    # discriminator = AdversarialNetwork(input_size=ad_net_input_dim).to(device)
    discriminator = AdversarialNetwork().to(DEVICE)


    # Optimizers
    F_opt = optim.Adam(feature_extractor.parameters(), lr=FE_LR)
    C_opt = torch.optim.Adam(classifier.parameters(), lr=LR)
    D_opt = optim.Adam(discriminator.parameters(), lr=LR)

    criterion_cls = nn.CrossEntropyLoss()
    
    # --- NEW: Setup for logging results ---
    results_log = []
    output_csv_path = 'training_results_gvb_dcase_cnn.csv'
    max_iter = NUM_EPOCHS * min(len(src_loader), len(tgt_loader))
    iter_num = 0
    num_batches = min(len(src_loader), len(tgt_loader))

    print("Starting training...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        feature_extractor.train()
        classifier.train()
        discriminator.train()

        total_cls_loss, total_adv_loss, total_entropy_loss = 0, 0, 0
        total_f_grad_norm, total_c_grad_norm, total_d_grad_norm = 0, 0, 0

        for batch_idx, (src_data, tgt_data) in enumerate(create_combined_loader(src_loader, tgt_loader)):
            src_wave_form, src_label = src_data
            tgt_wave_form, _ = tgt_data
            if batch_idx % 100 == 0 : 
                print(f"{epoch}@{batch_idx+1}" ,end=",")

            src_label = label_to_numerical(src_label , label_keys)
            src = src_wave_form.to(DEVICE)
            labels = src_label.to(DEVICE)
            tgt = tgt_wave_form.to(DEVICE)

            current_batch_size = min(len(src), len(tgt))
            if current_batch_size == 0: 
                print('skipped\n')
                continue

            src, labels, tgt = src[:current_batch_size], labels[:current_batch_size], tgt[:current_batch_size]
            
            F_opt.zero_grad()
            C_opt.zero_grad()
            D_opt.zero_grad()

            feat_source = feature_extractor(src)
            feat_target = feature_extractor(tgt)
            outputs_source , gvbg_source = classifier(feat_source)
            outputs_target, gvbg_target = classifier(feat_target)
            
            features = torch.cat((feat_source, feat_target), dim=0)
            outputs = torch.cat((outputs_source, outputs_target), dim=0)

            if GVBG == True : 
                outputs_source = outputs_source - gvbg_source
                outputs_target = outputs_target - gvbg_target 
                gvbg = torch.mean(torch.abs(gvbg_source))/2 + torch.mean(torch.abs(gvbg_target))/2
            else:
                gvbg = 0

            softmax_src = nn.Softmax(dim=1)(outputs_source)
            softmax_tgt = nn.Softmax(dim=1)(outputs_target)
            softmax_out = torch.cat((softmax_src, softmax_tgt), dim=0)

            if METHOD == 'CDAN-E':           
                entropy = loss.Entropy(softmax_out)
                transfer_loss,gvbd = loss.CDAN([features, softmax_out], discriminator, entropy, models.calc_coeff(iter_num), random_layer,GVBD = GVBD)
            elif METHOD == 'DANN+E':          
                entropy = loss.Entropy(softmax_out)
                transfer_loss,gvbd = loss.DANN(features, discriminator, entropy, models.calc_coeff(iter_num))
            elif METHOD  == 'CDAN':
                transfer_loss,gvbd = loss.CDAN([features, softmax_out], discriminator, None, None, random_layer)
            elif METHOD  == 'DANN':
                transfer_loss,gvbd = loss.DANN(features, discriminator)
            else:
                raise ValueError('Method cannot be recognized.')


        

        
            classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels)
            total_loss = trade_off * transfer_loss + classifier_loss + gvbg+ gvbd*GVBD
            total_loss.backward()
            F_opt.step()
            C_opt.step()
            D_opt.step()
        if epoch % 1  == 0:
            src_accuracy = test(feature_extractor,classifier,src_test_loader,DEVICE)
            tgt_accuracy = test(feature_extractor,classifier,tgt_test_loader,DEVICE)
            log_str = "iter: {:05d}, classification: {:.5f}, transfer: {:.5f}, gvbg: {:.5f}, gvbd:{:.5f}".format(iter_num, classifier_loss, transfer_loss, gvbg,gvbd)
            print(log_str,f'src_accuray : {src_accuracy}, tgt_accuracy : {tgt_accuracy}')
        # torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))

    print(f"Training finished. Results saved to {output_csv_path}")


if __name__ == '__main__':
    train()
