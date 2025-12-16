from models.DcaseNet import get_DcaseNet_v3
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

# from parser import get_args

# args = get_args()
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
PATH = '/DATA/G3/Datasets/archive/Original_split/TAU-urban-acoustic-scenes-2020-mobile-development'
evaluation_setup = 'evaluation_setup'
wav_file = os.path.join(PATH  , 'audio')
meta_scp = os.path.join(PATH  , 'meta.csv')
d_label_ASC_pth = os.path.join(PATH , evaluation_setup , 'd_label.pkl')
fold_trn = os.path.join(PATH , evaluation_setup , 'fold1_train.csv')
fold_evl = os.path.join(PATH , evaluation_setup , 'fold1_evaluate.csv')
results_csv_path = './results.csv'
verbose = 1 
d_label_ASC = [] 
l_label_ASC = []
bs_ASC = 32 
nb_worker = 24
nb_mels = 128 
nb_frames_ASC = 250
src_device = 'a'
tgt_device = 'b'
# if args.reproducible:
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

### For the ASC Dcase Model pre-trained ###
lines_ASC = get_utt_list(os.path.join(PATH , wav_file))

# try :

if os.path.exists(d_label_ASC_pth):
    d_label_ASC, l_label_ASC = pk.load(open(d_label_ASC_pth, 'rb'))
    print('Pickle File Exists')
# except FileNotFoundError: 
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

with open(results_csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['weight', 'src_device', 'tgt_device', 'accuracy'])

# DcaseNet_v3 = get_DcaseNet_v3()
# model =  DcaseNet_v3
# pre_trained_model = ('./weights/Joint/weights/best_ASC.pt')
# model.load_state_dict(torch.load(pre_trained_model))
# trn_src_ASC = filter_with_device(trn_lines_ASC , src_device)
# trn_tgt_ASC = filter_with_device(trn_lines_ASC , tgt_device)
# evl_src_ASC = filter_with_device(evl_lines_ASC , src_device)
# evl_tgt_ASC = filter_with_device(evl_lines_ASC , tgt_device)
# largs = {
#         'trn_lines': trn_tgt_ASC,
#         'evl_lines': evl_tgt_ASC,
#         'd_label': d_label_ASC,
#     }
# trnset_gen_ASC, evlset_gen_ASC = get_loaders_ASC(largs)
# trnset_gen_ASC_itr = cycle(trnset_gen_ASC)

# # save_dir = args.save_dir+args.name+'/'
# # if not os.path.exists(save_dir): os.makedirs(save_dir)
# # if not os.path.exists(save_dir+'results/'): os.makedirs(save_dir+'results/')
# # if not os.path.exists(save_dir+'weights/'): os.makedirs(save_dir+'weights/')
# arg = {"nb_mels" : nb_mels , "nb_frames_ASC" : nb_frames_ASC , "verbose" : verbose}
# acc, conf_mat = evaluate_ASC(model = model,
#             evlset_gen = evlset_gen_ASC,
#             device = device,
#             arguments = arg,
#         )
# print('ASC acc:\t{}'.format(acc))



# Assuming these are your functions and variables:
# get_DcaseNet_v3, filter_with_device, get_loaders_ASC, evaluate_ASC
# trn_lines_ASC, evl_lines_ASC, d_label_ASC
# nb_mels, nb_frames_ASC, verbose

weights_dict = {
    'joint': './weights/Joint/weights/best_ASC.pt',
    'fine_tuneASC': './weights/fine-tuneASC/weights/best_ASC.pt'  # example path, adjust accordingly
}

devices = ['a', 'b', 'c', 's1', 's2', 's3']
import ipdb
ipdb.set_trace()
for weight in ['joint', 'fine_tuneASC']:
    pre_trained_model_path = weights_dict[weight]
    DcaseNet_v3 = get_DcaseNet_v3()
    model = DcaseNet_v3
    model.load_state_dict(torch.load(pre_trained_model_path))
    model.eval()  # switch to evaluation mode if needed
    for src_device in ['a']:
        for tgt_device in devices:
            print(f"Evaluating with weight: {weight}, src_device: {src_device}, tgt_device: {tgt_device}")
            
            # Initialize model and load weights
            
            
            # Filter lines by device
            trn_src_ASC = filter_with_device(trn_lines_ASC, src_device)
            trn_tgt_ASC = filter_with_device(trn_lines_ASC, tgt_device)
            evl_src_ASC = filter_with_device(evl_lines_ASC, src_device)
            evl_tgt_ASC = filter_with_device(evl_lines_ASC, tgt_device)
            
            # Prepare data loaders
            largs = {
                'trn_lines': trn_tgt_ASC,
                'evl_lines': evl_tgt_ASC,
                'd_label': d_label_ASC,
            }
            trnset_gen_ASC, evlset_gen_ASC = get_loaders_ASC(largs)
            trnset_gen_ASC_itr = cycler(trnset_gen_ASC)
            
            # Evaluation arguments
            arg = {
                "nb_mels": nb_mels,
                "nb_frames_ASC": nb_frames_ASC,
                "verbose": verbose
            }
            
            # Run evaluation
            acc, conf_mat = evaluate_ASC(
                model=model,
                evlset_gen=evlset_gen_ASC,
                device=device,  # Assuming device means target device here
                arguments=arg,
            )
            
            print(f'ASC accuracy (weight={weight}, src_device={src_device}, tgt_device={tgt_device}): {acc}\n')

            with open(results_csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([weight, src_device, tgt_device, acc])