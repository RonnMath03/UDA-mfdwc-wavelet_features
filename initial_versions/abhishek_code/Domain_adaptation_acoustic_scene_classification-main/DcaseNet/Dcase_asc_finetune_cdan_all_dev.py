from models.DcaseNet_seperated import get_DcaseNet_v3 
from models.DcaseNet_seperated import  Classifier , Discriminator , DcaseNet_v3_FE
from utils import * 
import torch 
import os
import torch.nn as nn 
import torch.nn.functional as F
import pickle as pk
from torch.utils import data
from itertools import cycle as cycler
from dataloaders import Dataset_DCASE2020_t1, get_loaders_SED, get_loaders_TAG
from trainer import evaluate_ASC
import csv
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np
from datetime import datetime
import warnings
from torch.autograd import Function

warnings.filterwarnings('ignore')
def create_combined_loader(loader1, loader2):
    return zip(loader1, cycler(loader2))

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
PATH = '/DATA/s23103/TAU-urban-acoustic-scenes-2020-mobile-development'
evaluation_setup = 'evaluation_setup'
wav_file = os.path.join(PATH  , 'audio')
meta_scp = os.path.join(PATH  , 'meta.csv')
d_label_ASC_pth = os.path.join(PATH , evaluation_setup , 'd_label.pkl')
fold_trn = os.path.join(PATH , evaluation_setup , 'fold1_train.csv')
fold_evl = os.path.join(PATH , evaluation_setup , 'fold1_evaluate.csv')
save_path = './results/'
os.makedirs(save_path , exist_ok=True)
verbose = 1 
d_label_ASC = [] 
l_label_ASC = []
bs_ASC = 128 
nb_worker = 24
nb_mels = 128 
nb_frames_ASC = 250
src_device = 'a'
for dev in ['b' , 'c' , 's1' , 's2' , 's3']:
    tgt_device = dev
    print(f'Training for {src_device} , {tgt_device} Adaptation')
    NUM_EPOCHS = 50
    feature_dim = 384
    NUM_CLASS = 10
    METHOD = 'CDAN-E'
    results_csv_path = f'./results_dcase_net_grl_a-{tgt_device}.csv'
    lines_ASC = get_utt_list(os.path.join(PATH , wav_file))
    RANDOM = True
    if os.path.exists(d_label_ASC_pth):
        d_label_ASC, l_label_ASC = pk.load(open(d_label_ASC_pth, 'rb'))
        print('Pickle File Exists')
    else :
        with open(meta_scp , 'r') as f:
            l_meta_ASC = f.readlines()
        d_label_ASC, l_label_ASC = make_d_label(l_meta_ASC[1:])
        pk.dump([d_label_ASC, l_label_ASC], open(d_label_ASC_pth, 'wb'))
        print('Pickle File Dumped')

    trn_lines_ASC = split_dcase2020_fold_strict(fold_scp = fold_trn, lines = lines_ASC)
    evl_lines_ASC = split_dcase2020_fold_strict(fold_scp = fold_evl, lines = lines_ASC)

    if verbose > 0 :
        print('ASC DB statistics')
        print('# trn samp: {}\n# evl samp: {}'.format(len(trn_lines_ASC), len(evl_lines_ASC)))
        print(d_label_ASC)
        print(l_label_ASC)

    def get_loaders_ASC(loader_args):
        trnset = Dataset_DCASE2020_t1(
            lines = loader_args['trn_lines'],
            base_dir = wav_file,
            d_label = loader_args['d_label'],
            verbose = verbose
        )
        trnset_gen = data.DataLoader(
            trnset,
            batch_size = bs_ASC,
            shuffle = True,
            num_workers = nb_worker,
            pin_memory = True,
            drop_last = True
        )

        evlset = Dataset_DCASE2020_t1(
            lines = loader_args['evl_lines'],
            trn = False,
            base_dir = wav_file,
            d_label = loader_args['d_label'],
            verbose = verbose
        )
        evlset_gen = data.DataLoader(
            evlset,
            batch_size = bs_ASC//3,
            shuffle = False,
            num_workers = nb_worker//2,
            pin_memory = True,
            drop_last = False
        )
        return trnset_gen, evlset_gen

    class GradReverse(Function):
        @staticmethod
        def forward(ctx, x, alpha):
            ctx.alpha = alpha
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad_output):
            output = grad_output.neg() * ctx.alpha
            return output, None


    with open(results_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'classifier_loss', 'source_accuracy', 'target_accuracy'])

    DcaseNet_v3 = DcaseNet_v3_FE()
    fe =  DcaseNet_v3
    pre_trained_model = './weights/Joint/weights/best_ASC.pt'
    fe.load_state_dict(torch.load(pre_trained_model), strict=False)
    classifier = Classifier()
    disc = Discriminator()
    if RANDOM : 
        random_layer = RandomLayer([feature_dim, NUM_CLASS], 384).to(device)
    else :
        random_layer = None

    fe = fe.to(device)
    classifier = classifier.to(device)
    disc = disc.to(device)


    trn_src_ASC = filter_with_device(trn_lines_ASC , src_device)
    trn_tgt_ASC = filter_with_device(trn_lines_ASC , tgt_device)
    evl_src_ASC = filter_with_device(evl_lines_ASC , src_device)
    evl_tgt_ASC = filter_with_device(evl_lines_ASC , tgt_device)

    larg_src = {
            'trn_lines': trn_src_ASC,
            'evl_lines': evl_src_ASC,
            'd_label': d_label_ASC,
        }
    larg_tgt = {
            'trn_lines': trn_tgt_ASC,
            'evl_lines': evl_tgt_ASC,
            'd_label': d_label_ASC,
        }

    trnset_src_ASC, evlset_src_ASC = get_loaders_ASC(larg_src)
    trnset_tgt_ASC, evlset_tgt_ASC = get_loaders_ASC(larg_tgt)

    criterion =  nn.CrossEntropyLoss().to('cuda')
    bce = nn.BCELoss()
    arg = {"nb_mels" : nb_mels , "nb_frames_ASC" : nb_frames_ASC , "verbose" : verbose}
    loss_tt_epoch = []
    criterion = nn.CrossEntropyLoss()
    f_opt = optim.Adam(fe.parameters(), lr=0.0001)
    c_opt = optim.Adam(classifier.parameters(), lr=0.0001)
    d_opt = optim.Adam(disc.parameters(), lr=0.0001)

    best_acc = -1
    for epoch in range(1 , NUM_EPOCHS + 1 ):
        fe.train()
        classifier.train()

        total_cls_loss  = 0 
        src_acc = 0
        current_lambda = get_lambda(epoch - 1, NUM_EPOCHS + 1 ) 

        for batch_idx , (src_data , tgt_data) in enumerate(create_combined_loader(trnset_src_ASC , trnset_tgt_ASC)):

            m_batch, m_label = src_data 
            t_batch, t_label = tgt_data

            print(f"Epoch {epoch}, Batch {batch_idx+1}" ,end=",")

            current_batch_size = min(t_batch.size(0), m_batch.size(0))
            m_batch = m_batch[:current_batch_size]
            t_batch = t_batch[:current_batch_size]
            m_label = m_label[: current_batch_size]
            D_src = torch.zeros(current_batch_size, 1 ).to(device)
            D_tgt = torch.ones(current_batch_size , 1 ).to(device)
            D_labels = torch.cat([D_src, D_tgt], dim=0)

            d_opt.zero_grad()
            c_opt.zero_grad()
            f_opt.zero_grad()

            m_batch, m_label = m_batch.to(device), m_label.to(device)
            t_batch = t_batch.to(device)

            out_ASC = fe(m_batch, mode = ['ASC'])['ASC']
            out_tgt = fe(t_batch, mode = ['ASC'])['ASC']
            h = torch.cat([out_ASC , out_tgt] , dim = 0 )
            
            
            ## Solo Training of Discriminator CDAN

            y = disc(h)

            pred_src = classifier(out_ASC)
            cls_loss = criterion(pred_src, m_label)

            pred_tgt = classifier(out_tgt)
            pred_combined = torch.cat((pred_src, pred_tgt), dim=0)
            softmax_combined = nn.Softmax(dim=1)(pred_combined)

            reversed_features = GradReverse.apply(h, current_lambda)


            adv_loss = CDAN([reversed_features, softmax_combined], disc, random_layer)

            Ltot = cls_loss + adv_loss

            if METHOD == 'CDAN-E':
                    softmax_target = nn.Softmax(dim=1)(pred_tgt)
                    entropy_loss = torch.mean(Entropy(softmax_target))
                    Ltot += (entropy_loss * current_lambda)

        
            ## Joint training of Discriminator , Feature Extractor and Classifier 

            Ltot.backward()
            
            f_opt.step()
            c_opt.step()
            d_opt.step()
            total_cls_loss += Ltot.item()
            
        avg_epoch_loss = total_cls_loss / len(trnset_src_ASC)
        loss_tt_epoch.append(total_cls_loss)
        
        fe.eval()
        classifier.eval()

        y_pred = []
        y_true = []
        with torch.no_grad():
            for m_batch, m_label in evlset_src_ASC:
                m_batch = m_batch.view(-1, 1, nb_mels, nb_frames_ASC+1).to(device)
                out = fe(m_batch, mode = ['ASC'])['ASC']
                out = classifier(out)
                out = F.softmax(out, dim=-1).view(-1, 3, out.size(1)).mean(dim=1, keepdim=False)
                m_label = list(m_label.numpy())
                y_pred.extend(list(out.cpu().numpy()))
                y_true.extend(m_label)
                
            y_pred = np.argmax(np.array(y_pred), axis=1).tolist()
            conf_mat = confusion_matrix(y_true = y_true, y_pred = y_pred)
            nb_cor = np.trace(conf_mat)
            src_accuracy = (nb_cor / len(y_true)) * 100 
        print(f'\nAccuracy on Source device {src_device}: {src_accuracy:.2f} %')

        y_pred = []
        y_true = []
        with torch.no_grad():
            for m_batch, m_label in evlset_tgt_ASC:
                m_batch = m_batch.view(-1, 1, nb_mels, nb_frames_ASC+1).to(device)
                out = fe(m_batch, mode = ['ASC'])['ASC']
                out = classifier(out)
                out = F.softmax(out, dim=-1).view(-1, 3, out.size(1)).mean(dim=1, keepdim=False)
                m_label = list(m_label.numpy())
                y_pred.extend(list(out.cpu().numpy()))
                y_true.extend(m_label)

            y_pred = np.argmax(np.array(y_pred), axis=1).tolist()
            conf_mat = confusion_matrix(y_true = y_true, y_pred = y_pred)
            nb_cor = np.trace(conf_mat)
            tgt_accuracy = (nb_cor / len(y_true)) * 100 

        print(f'Accuracy on Target device {tgt_device}: {tgt_accuracy:.2f} %')
        if tgt_accuracy > best_acc : 
            save_set = {
            'FE' : fe.state_dict(),
            'classifier' : classifier.state_dict(),
            'disc' : disc.state_dict()
            }
            timestamp = datetime.now()
            torch.save(save_set , f'{save_path}/best_grl_{tgt_device}_{timestamp}.pth')
        with open(results_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_epoch_loss, src_accuracy, tgt_accuracy])

