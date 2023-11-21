import argparse
import numpy as np
import torch
import torch.nn as nn
import random
from utils import get_dataset, train, eval, get_divs_per_group, scaled_reverse_count_group
from model import TransAm, ConvTransAm
import time
import pickle
import os
import pandas as pd
import sys
import csv
import datetime
import pytz

import warnings
warnings.filterwarnings(action='ignore')
#-------------------------------------Argparse-------------------------------------------------------

parser = argparse.ArgumentParser()

# wandb
parser.add_argument('--wandb', action='store_true', help="wandb_on_off")

# random seed
parser.add_argument('--seed', type=int, default=0)

# Data Load
parser.add_argument('--data_path', type=str, default='/mlainas/MIMIC/finaldata/', help='Path for data') # 정제된 데이터의 경로
parser.add_argument('--meta_data_path', type=str, default='/mlainas/MIMIC/p09group_BP/', help='Path for meta data') #p09group_BP 경로
parser.add_argument('--max_per_patient', type=int, default=100, help="record per each patient") # 한 환자당 불러올 최대 ppg 개수
parser.add_argument('--sampling', type=str, default="normal", choices=["normal", "upsampling", "downsampling", "small"],
                    help='Choose sampling methods to treat imbalance') # upsampling과 downsampling은 ERM method시에만 활용
parser.add_argument('--small_ratio', type=float, default=None, help='Ratio of Training Data when using small loader')
# sampling이 small일 때, original data의 얼마 정도로 줄일 것인지 예) 0.2 : 20% 데이터만 활용
parser.add_argument('--up_small', action='store_true', help='Upsampling for small ratio set') # small + upsampling
parser.add_argument('--down_small', action='store_true', help='Downsampling for small ratio set') # small + downsampling
parser.add_argument('--tr_ratio', type=float, default=0.7) # train으로 사용할 데이터 비율
parser.add_argument('--val_ratio', type=float, default=0.1) # validation set으로 사용할 데이터 비율
parser.add_argument('--te_ratio', type=float, default=0.2)  # test set으로 사용할 데이터 비율

parser.add_argument('--max', type=float, default=1.0, help="preprocessing") # ppg 변환 시 최대 ppg 값
parser.add_argument('--min', type=float, default=-1.0, help="preprocessing") # ppg 변환 시 최소 ppg값
parser.add_argument('--save_pkl', action='store_true', help='option saving dataloader as pickle file') # loader 생성

# Model
parser.add_argument('--mode', type=str, default='all',
                    choices=['all', 'hypo', 'normal','prehyper', 'hyper2', 'crisis'],
                    help='select group for training') # train으로 사용할 그룹 설정/ 모든 그룹 train시 all 사용
parser.add_argument('--method', type=str, default='erm', choices=['erm', 'dro', 'vrex', 'crex', 'drex', 'cdrex'],
                    help='Choose learning method') # crex : C-REx / drex : D-REx / cdrex : CD-REx

parser.add_argument('--model', type=str, default='ConvTransformer', choices=['Transformer', 'ConvTransformer'])
parser.add_argument('--d_input', type=int, default=64, help="model input and PE dimension")
parser.add_argument('--d_output', type=int, default=2, help="model output") # SBP, DBP 각각 추정 = 2
parser.add_argument('--d_model', type=int, default=96, help="model hidden dimension") # hidden dimension
parser.add_argument('--dropout', type=float, default=0.1, help="")
parser.add_argument('--num_layers', type=int, default=2, help="") # Transformer Encoder Layer 개수
parser.add_argument('--num_heads', type=int, default=4, help="") # Transformer 내 head 개수
parser.add_argument('--random_splits',action='store_true')

parser.add_argument('--num_filters', type=int, default=8, help='Number of filter per 3/5/7/9 size') # ConvTransformer filter 개수

parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--wd', type=float, default=5e-3, help="weight decay")
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--printing_step', type=int, default=10)
parser.add_argument('--sweep_name', type=str, default="default_sweep")
parser.add_argument('--resume', type=str, default=None)  # 학습 완료된 모델을 추가로 학습할 때 경로와 함께 설정
parser.add_argument('--erm_loader', action="store_true") # 다른 method들의 loader가 아닌 erm에서 사용한 loader 그대로 사용할 때
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--resume_dfr', action='store_true', help='Run with pre-trained model')
parser.add_argument('--pretrained_model_path', type=str, default="", help='pre-trained model path')
parser.add_argument('--exit', action='store_true', help="Exit after making loader")

#### for DRO
parser.add_argument('--robust_step_size', type=float, default=0.01)

#### for V-REx
parser.add_argument('--sbp_beta', type=float, default=1)        # V-REX의 SBP variance를 조정할 coefficient
parser.add_argument('--dbp_beta', type=float, default=0.1)         # V-REX의 DBP variance를 조정할 coefficient
parser.add_argument('--annealing_epoch', type=float, default=500) # 처음 ERM으로 학습시킬 Epoch

#### for DFR
parser.add_argument('--DFR', action='store_true', help='option to run DFR')
parser.add_argument('--min_cls_thres', type=float, default=0.1, help='minimum count of minority class (e.g. 10% of whole data)')
parser.add_argument('--dfr_epoch', type=int, default=100, help='epoch for DFR')

### for Ours
parser.add_argument('--tukey', action='store_true', help='Use tukey transformation to get divergence')
parser.add_argument('--beta', type=float, default=0.5, help='parameter for Tukey transformation (Default : 0.5)')

parser.add_argument('--C1', type=float, default=1e-5, help='scaling variance with number of training data per group (Default : 1e-5)')
parser.add_argument('--C21', type=float, default=1e-4, help='scaling variance for sbp loss with kl divergence per group (Default : 1e-4)')
parser.add_argument('--C22', type=float, default=5e-4, help='scaling variance for dbp loss with kl divergence per group (Default : 5e-4)')

### Mix-up
parser.add_argument('--time_cutmix', action='store_true', help='Run Mix-Up for data augmentation')

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.feature_size = 1 ### Input Shape : batch_size x 1000 x 1  (1000: ppg time stamp length/ 1: ppg signal)

## wandb config
if args.wandb:
    import wandb
    wandb.init(project="pakdd_etri", entity="mlai_uos_etri") # wandb 사용시 project와 entitiy 정보 등을 다시 설정해줘야 합니다.
    wandb.config.update(args)


#----------------------------------Setting---------------------------------------------------------

# 모델 저장을 위한 시간 설정
tz = pytz.timezone('Asia/Seoul')
datetime_here = datetime.datetime.now(tz)
local_time = datetime_here.strftime("%m-%d_%H-%M-%S")
args.local_time = local_time

# Control Randomness
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)
np.random.seed(args.seed)
random.seed(args.seed)

# Organizing
os.makedirs('results',exist_ok=True)
os.makedirs('results/csv_log', exist_ok=True)
os.makedirs('results/csv_log/val', exist_ok=True)
os.makedirs('results/csv_log/test', exist_ok=True)
os.makedirs('results/csv_log/worst', exist_ok=True)
os.makedirs('results/csv_log/best', exist_ok=True)
os.makedirs('results/csv_log/group_best', exist_ok=True)
os.makedirs('results/model_save/', exist_ok=True)
os.makedirs('results/txt_log', exist_ok=True)
os.makedirs('data',exist_ok=True)
os.makedirs('results/best_val', exist_ok=True)
os.makedirs('results/group_best_val', exist_ok=True)
os.makedirs('results/best_worst_val', exist_ok=True)


# Elapsed Time
current_time = time.time()

# In ERM and DRO, there is no annealing.
if args.method in ['erm', 'dro']:
    args.annealing_epoch = -1


#---------------------------------Loader------------------------------------------------------------------
# Create Loader
if args.save_pkl == True:
    print("Start to save loaders as pickle file")
    if args.erm_loader:
        tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = get_dataset(
            args)
        with open(f"data/erm_{args.mode}_{args.sampling}_{args.seed}.pkl", 'wb') as f:
            pickle.dump([tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader],
                        f)
    else:
        tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = get_dataset(
            args)
        if args.sampling == "small":
            if args.up_small:
                with open(f"data/{args.method}_{args.mode}_{args.sampling}_ratio_{args.small_ratio}_up_{args.seed}.pkl", 'wb') as f:
                    pickle.dump([tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader],
                                f)
            elif args.down_small:
                with open(f"data/{args.method}_{args.mode}_{args.sampling}_ratio_{args.small_ratio}_down_{args.seed}.pkl", 'wb') as f:
                    pickle.dump([tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader],
                                f)
            else:
                with open(f"data/{args.method}_{args.mode}_{args.sampling}_ratio_{args.small_ratio}_{args.seed}.pkl", 'wb') as f:
                    pickle.dump([tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader],
                                f)
        else:
            with open(f"data/{args.method}_{args.mode}_{args.sampling}_{args.seed}.pkl", 'wb') as f:
                pickle.dump([tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader],
                            f)
        
    if args.exit:
        sys.exit()

# load Loader
else:
    if args.erm_loader:
        if args.sampling == "small":
            if args.up_small:
                print("Small ratio=0.5 and Upsampling")
                with open(f"data/erm_{args.mode}_{args.sampling}_ratio_{args.small_ratio}_up_{args.seed}.pkl", 'rb') as f:
                    tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = pickle.load(f)
            elif args.down_small:
                print("Small ratio=0.5 and Downsampling")
                with open(f"data/erm_{args.mode}_{args.sampling}_ratio_{args.small_ratio}_down_{args.seed}.pkl", 'rb') as f:
                    tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = pickle.load(f)
            else:
                with open(f"data/erm_{args.mode}_{args.sampling}_ratio_{args.small_ratio}_{args.seed}.pkl", 'rb') as f:
                    tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = pickle.load(f)
        else:
            with open(f"data/erm_{args.mode}_{args.sampling}_{args.seed}.pkl", 'rb') as f:
                tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = pickle.load(
                    f)
    else:
        if args.sampling == "small":
            if args.up_small:
                print("Small ratio=0.5 and Upsampling")
                with open(f"data/erm_{args.mode}_{args.sampling}_ratio_{args.small_ratio}_up_{args.seed}.pkl", 'rb') as f:
                    tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = pickle.load(f)
            elif args.down_small:
                print("Small ratio=0.5 and Downsampling")
                with open(f"data/erm_{args.mode}_{args.sampling}_ratio_{args.small_ratio}_down_{args.seed}.pkl", 'rb') as f:
                    tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = pickle.load(f)
            else:
                with open(f"data/erm_{args.mode}_{args.sampling}_ratio_{args.small_ratio}_{args.seed}.pkl", 'rb') as f:
                    tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = pickle.load(f)
        else:
            with open(f"data/{args.method}_{args.mode}_{args.sampling}_{args.seed}.pkl", 'rb') as f:
                tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = pickle.load(
                    f)
print("Complete to Load Data Loader")

#---------------------------------Prepareing Saving Best/Worst Model--------------------------------------
if args.method == "erm":
    name = f"{args.method}_{args.mode}_{args.sampling}_{args.wd}_{args.lr}_{args.dropout}_{args.d_input}_{args.num_layers}"
elif args.method == "dro":
    name = f"{args.method}_{args.mode}_{args.robust_step_size}_{args.wd}_{args.lr}_{args.dropout}_{args.d_input}_{args.num_layers}"
elif args.method == "vrex":
    name = f"{args.method}_{args.mode}_{args.sbp_beta}_{args.dbp_beta}_{args.annealing_epoch}_{args.wd}_{args.lr}_{args.dropout}_{args.d_input}_{args.num_layers}"
elif args.method in ['crex', 'drex', 'cdrex']:
    name = f"{args.method}_{args.mode}_{args.sbp_beta}_{args.dbp_beta}_{args.annealing_epoch}_{args.wd}_{args.lr}_{args.dropout}_{args.d_input}_{args.num_layers}_{args.C1}_{args.C21}_{args.C22}"

if args.wandb:
    wandb.run.name = name

if args.DFR:
    previous_name = name
    name = "pretrained_" + name
    args.method = "pretrained_" + args.method

logger_name = f"results/txt_log/{local_time}_overview_{name}.txt"


#---------------------------------Training----------------------------------------------------------------
# ## Get Number of Group list
if args.method in ['crex', 'cdrex']:
    count_group_list = scaled_reverse_count_group(args, tr_dataset_loader)
    print("Get count per group!!")
    print(count_group_list)
else:
    count_group_list = None

# ## Get Distribution of training set per group
if args.method in ['drex', 'cdrex']:
    group_div_list = get_divs_per_group(args, tr_dataset_loader)
    print("Get KL divergence between groups and total data distribution!!")
    print(group_div_list)
else:
    group_div_list = None
    

## Training

# Select model
if args.model == 'Transformer':
    model = TransAm(args).to(args.device)
elif args.model == 'ConvTransformer':
    model = ConvTransAm(args).to(args.device)
print(f"Load Model {model.model_type}")


# Set Training Loss Criterion
if args.method == 'erm' or args.method == 'pretrained_erm': # pretrained_erm은 DFR 방식을 위함
    args.criterion = nn.MSELoss()
elif args.method in ['dro','vrex', 'crex', 'drex', 'cdrex']:
    args.criterion = nn.MSELoss(reduction='none') 
else:
    NotImplementedError

# Set Test Loss Criterion
args.criterion_eval = nn.L1Loss()

# Set Optimizer
args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

# Set scheduler
args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, 1.0, gamma=0.95) 

## DFR을 쓰지 않을 때: 논문화 시킨 내용은 전부 이 if문 안에서만 돌림.
if not args.resume_dfr:
    best_val_loss = 99999
    best_group_loss = 99999
    best_worst_loss = 99999
    if not args.DFR:
        args.dfr_epoch = 0
    if args.method == "dro":
        args.sbp_adv_probs = torch.ones(5).cuda() / 5 # group probability initialization
        args.dbp_adv_probs = torch.ones(5).cuda() / 5  # group probability initialization
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
    print("Start Training....")
    for epoch in range(args.max_epochs-args.dfr_epoch):
        current_time = time.time()
        args.epoch = epoch

        # Train
        train_loss, n_tr, n_tr_hypo, n_tr_normal, n_tr_prehyper, n_tr_hyper2, n_tr_crisis = train(args,model, tr_dataset_loader, count_group_list, group_div_list)

        # Eval Val
        val_sbp_loss, val_dbp_loss, n_val, \
            val_hypo_sbp_loss, val_hypo_dbp_loss, n_val_hypo, \
            val_normal_sbp_loss, val_normal_dbp_loss, n_val_normal, \
            val_prehyper_sbp_loss, val_prehyper_dbp_loss, n_val_prehyper, \
            val_hyper2_sbp_loss, val_hyper2_dbp_loss,n_val_hyper2, \
            val_crisis_sbp_loss, val_crisis_dbp_loss, n_val_crisis = eval(args,model,val_dataset_loader)

        # Calculate Val loss
        val_group_avg_loss = ((val_hypo_sbp_loss + val_normal_sbp_loss + val_prehyper_sbp_loss + val_hyper2_sbp_loss + val_crisis_sbp_loss)/5 +
                    (val_hypo_dbp_loss+ val_normal_dbp_loss + val_prehyper_dbp_loss + val_hyper2_dbp_loss + val_crisis_dbp_loss)/5)

        val_worst_loss = max(val_hypo_dbp_loss+val_hypo_sbp_loss, val_prehyper_sbp_loss+val_prehyper_dbp_loss, val_normal_sbp_loss+val_normal_dbp_loss,
               val_crisis_sbp_loss+val_crisis_dbp_loss, val_hyper2_sbp_loss+val_hyper2_dbp_loss)

        # Eval Test
        te_sbp_loss, te_dbp_loss, n_te, \
            te_hypo_sbp_loss, te_hypo_dbp_loss, n_te_hypo, \
            te_normal_sbp_loss, te_normal_dbp_loss,n_te_normal, \
            te_prehyper_sbp_loss, te_prehyper_dbp_loss, n_te_prehyper,\
            te_hyper2_sbp_loss, te_hyper2_dbp_loss, n_te_hyper2, \
            te_crisis_sbp_loss, te_crisis_dbp_loss, n_te_crisis, = eval(args,model,te_dataset_loader)

        # Calculate Test loss
        te_group_avg_loss = ((te_hypo_sbp_loss + te_normal_sbp_loss + te_prehyper_sbp_loss + te_hyper2_sbp_loss + te_crisis_sbp_loss)/5 +
                    (te_hypo_dbp_loss+ te_normal_dbp_loss + te_prehyper_dbp_loss + te_hyper2_dbp_loss + te_crisis_dbp_loss)/5)

        te_worst_loss = max(te_hypo_dbp_loss+te_hypo_sbp_loss, te_prehyper_sbp_loss+te_prehyper_dbp_loss, te_normal_sbp_loss+te_normal_dbp_loss,
               te_crisis_sbp_loss+te_crisis_dbp_loss, te_hyper2_sbp_loss+te_hyper2_dbp_loss)
            
        ## wandb
        if args.wandb:
            wandb.log({"Training loss" : train_loss,
                    "Validation Total loss": val_sbp_loss + val_dbp_loss, "Validation SBP loss" : val_sbp_loss, "Validation DBP loss" : val_dbp_loss, "Validation Group Average loss" : val_group_avg_loss, "Validation Worst loss" : val_worst_loss,
                    "Test Total loss" : te_sbp_loss + te_dbp_loss, "Test SBP loss" : te_sbp_loss, "Test DBP loss" : te_dbp_loss, "Test Group Average loss" : te_group_avg_loss, "Test Worst loss" : te_worst_loss,
                    "Val Hypo SBP Loss" : val_hypo_sbp_loss, "Val Normal SBP Loss" : val_normal_sbp_loss, "Val Prehyper SBP Loss" : val_prehyper_sbp_loss, "Val Hyper2 SBP Loss" : val_hyper2_sbp_loss, "Val Crisis SBP Loss" : val_crisis_sbp_loss,
                    "Val Hypo DBP Loss" : val_hypo_dbp_loss, "Val Normal DBP Loss" : val_normal_dbp_loss, "Val Prehyper DBP Loss" : val_prehyper_dbp_loss, "Val Hyper2 DBP Loss" : val_hyper2_dbp_loss, "Val Crisis DBP Loss" : val_crisis_dbp_loss,
                    "Te Hypo SBP Loss" : te_hypo_sbp_loss, "Te Normal SBP Loss" : te_normal_sbp_loss, "Te Prehyper SBP Loss" : te_prehyper_sbp_loss, "Te Hyper2 SBP Loss" : te_hyper2_sbp_loss, "Te Crisis SBP Loss" : te_crisis_sbp_loss,
                    "Te Hypo DBP Loss" : te_hypo_dbp_loss, "Te Normal DBP Loss" : te_normal_dbp_loss, "Te Prehyper DBP Loss" : te_prehyper_dbp_loss, "Te Hyper2 DBP Loss" : te_hyper2_dbp_loss, "Te Crisis DBP Loss" : te_crisis_dbp_loss,
                    })

        # logging
        train_log = f"Epoch: {epoch}, Train Loss: {train_loss}, Elapsed Time: {time.time() - current_time}, Num Train: {n_tr}, \n" \
                    f"Num Tr Hypo: {n_tr_hypo},,Num Tr Normal: {n_tr_normal}, Num Tr prehyper: {n_tr_prehyper}, Num Tr Hyper2: {n_tr_hyper2}," \
                    f"Num Tr Crisis: {n_tr_crisis}, \n"
        val_log = f"Epoch: {epoch}, Val SBP Loss: {val_sbp_loss}, Val DBP Loss: {val_dbp_loss}, Num Val: {n_val},\n" \
                  f" Val Hypo-SBP Loss: {val_hypo_sbp_loss}, Val Hypo-DBP Loss: {val_hypo_dbp_loss}, Num Val Hypo: {n_val_hypo}\n" \
                  f" Val Normal-SBP Loss: {val_normal_sbp_loss}, Val Normal-DBP Loss: {val_normal_dbp_loss}, Num Val Normal: {n_val_normal}\n" \
                  f" Val prehyper-SBP Loss: {val_prehyper_sbp_loss}, Val prehyper-DBP Loss: {val_prehyper_dbp_loss}, Num Val prehyper: {n_val_prehyper}\n" \
                  f" Val Hyper2-SBP Loss: {val_hyper2_sbp_loss}, Val Hyper2-DBP Loss: {val_hyper2_dbp_loss}, Num Val Hyper2: {n_val_hyper2}\n"\
                  f" Val Crisis-SBP Loss: {val_crisis_sbp_loss}, Val Crisis-DBP Loss: {val_crisis_dbp_loss}, Num Val Crisis: {n_val_crisis}\n" \
                  f" Val Group Average Loss: {val_group_avg_loss}, Val Worst Group Loss: {val_worst_loss}"
        te_log = f"Epoch: {epoch}, Te SBP Loss: {te_sbp_loss}, Te DBP Loss: {te_dbp_loss}, Num Te: {n_te},\n" \
                 f" Te Hypo-SBP Loss: {te_hypo_sbp_loss}, Te Hypo-DBP Loss: {te_hypo_dbp_loss}, Num Te Hypo: {n_te_hypo}\n" \
                 f" Te Normal-SBP Loss: {te_normal_sbp_loss}, Te Normal-DBP Loss: {te_normal_dbp_loss}, Num Te Normal: {n_te_normal}\n" \
                 f" Te prehyper-SBP Loss: {te_prehyper_sbp_loss}, Te prehyper-DBP Loss: {te_prehyper_dbp_loss}, Num Te prehyper: {n_te_prehyper}\n" \
                 f" Te Hyper2-SBP Loss: {te_hyper2_sbp_loss}, Te Hyper2-DBP Loss: {te_hyper2_dbp_loss}, Num Te Hyper2: {n_te_hyper2}\n"\
                 f" Te Crisis-SBP Loss: {te_crisis_sbp_loss}, Te Crisis-DBP Loss: {te_crisis_dbp_loss}, Num Te Crisis: {n_te_crisis}\n" \
                 f" Te Group Average Loss: {te_group_avg_loss}, Te Worst Group Loss: {te_worst_loss}"
        log = "\n".join([train_log, val_log, te_log])

        val_log_dict = {"local_time": local_time, "epoch": epoch,
                    "total_loss": val_sbp_loss + val_dbp_loss, "sbp_loss": val_sbp_loss, "dbp_loss": val_dbp_loss,
                    "total_hypo": val_hypo_dbp_loss + val_hypo_sbp_loss,
                    "total_normal": val_normal_dbp_loss + val_normal_sbp_loss,
                    "total_prehyper": val_prehyper_dbp_loss + val_prehyper_sbp_loss,
                    "total_hyper2": val_hyper2_dbp_loss + val_hyper2_sbp_loss,
                    "total_crisis": val_crisis_dbp_loss + val_crisis_sbp_loss,
                    "sbp_hypo": val_hypo_sbp_loss, "sbp_normal": val_normal_sbp_loss,
                    "sbp_prehyper": val_prehyper_sbp_loss,
                    "sbp_hyper2": val_hyper2_sbp_loss, "sbp_crisis": val_crisis_sbp_loss,
                    "dbp_hypo": val_hypo_dbp_loss, "dbp_normal": val_normal_dbp_loss,
                    "dbp_prehyper": val_prehyper_dbp_loss,
                    "dbp_hyper2": val_hyper2_dbp_loss, "dbp_crisis": val_crisis_dbp_loss,
                    "group_avg_loss": val_group_avg_loss, "worst_group_loss": val_worst_loss,
                    **vars(args)}
        del val_log_dict['optimizer'], val_log_dict['scheduler'], val_log_dict['device']

        te_log_dict = {"local_time": local_time, "epoch": epoch,
                    "total_loss": te_sbp_loss + te_dbp_loss, "sbp_loss": te_sbp_loss, "dbp_loss": te_dbp_loss,
                    "total_hypo": te_hypo_dbp_loss + te_hypo_sbp_loss,
                    "total_normal": te_normal_dbp_loss + te_normal_sbp_loss,
                    "total_prehyper": te_prehyper_dbp_loss + te_prehyper_sbp_loss,
                    "total_hyper2": te_hyper2_dbp_loss + te_hyper2_sbp_loss,
                    "total_crisis": te_crisis_dbp_loss + te_crisis_sbp_loss,
                    "sbp_hypo": te_hypo_sbp_loss, "sbp_normal": te_normal_sbp_loss,
                    "sbp_prehyper": te_prehyper_sbp_loss,
                    "sbp_hyper2": te_hyper2_sbp_loss, "sbp_crisis": te_crisis_sbp_loss,
                    "dbp_hypo": te_hypo_dbp_loss, "dbp_normal": te_normal_dbp_loss,
                    "dbp_prehyper": te_prehyper_dbp_loss,
                    "dbp_hyper2": te_hyper2_dbp_loss, "dbp_crisis": te_crisis_dbp_loss,
                    "group_avg_loss": te_group_avg_loss, "worst_group_loss": te_worst_loss,
                    **vars(args)}
        del te_log_dict['optimizer'], te_log_dict['scheduler'], te_log_dict['device']
        if args.method=="dro":
            del val_log_dict["sbp_adv_probs"], val_log_dict["dbp_adv_probs"], te_log_dict["sbp_adv_probs"], te_log_dict["dbp_adv_probs"]

        if (epoch+10) % args.printing_step == 0 or epoch == 0:
            print("-"*20+"\n")
            print(log)
        if (epoch+1)%100 ==0:
            torch.save(model.state_dict(), f"./results/model_save/{local_time}_epoch{epoch}_{name}.pt")

        with open(logger_name,"a") as f:
            f.write(log)

        ## Best Performance Model
        if (val_sbp_loss + val_dbp_loss < best_val_loss) and (args.epoch >= args.annealing_epoch) :
            best_val_loss = val_sbp_loss + val_dbp_loss
            best_model_name = f"./results/best_val/{local_time}_{name}.pt"
            torch.save(model.state_dict(), best_model_name)
            log_dict = te_log_dict

        ## Best Group Loss Model
        if (val_group_avg_loss < best_group_loss) and (args.epoch >= args.annealing_epoch):
            best_group_loss = val_group_avg_loss
            best_group_model_name = f"./results/group_best_val/{local_time}_{name}.pt"
            torch.save(model.state_dict(), best_group_model_name)
            group_best_log_dict = te_log_dict

        ## Best Robust Model
        if (val_worst_loss < best_worst_loss) and (args.epoch >= args.annealing_epoch):
            best_worst_loss = val_worst_loss
            best_worst_model_name = f"./results/best_worst_val/{local_time}_{name}.pt"
            torch.save(model.state_dict(), best_worst_model_name)
            worst_log_dict = te_log_dict

        logger_dict = {"val": val_log_dict, "test": te_log_dict}
        if args.epoch == 0:
            for split in ["val", "test"]:
                csv_path = os.path.join(os.getcwd(), f"results/csv_log/{split}", f"{local_time}_{split}_{args.method}_{args.mode}_{args.sampling}.csv")
                file = open(csv_path, 'w')
                writer = csv.DictWriter(file, fieldnames=logger_dict[split].keys())
                writer.writeheader()
                writer.writerow(logger_dict[split])
        else:
            for split in ["val", "test"]:
                csv_path = os.path.join(os.getcwd(), f"results/csv_log/{split}", f"{local_time}_{split}_{args.method}_{args.mode}_{args.sampling}.csv")
                file = open(csv_path, 'a')
                writer = csv.DictWriter(file, fieldnames=logger_dict[split].keys())
                writer.writerow(logger_dict[split])


#########################################################
# DFR
#########################################################
if args.DFR:
    if args.resume_dfr:
        ckpt_path = args.pretrained_model_path   # DFR시 사전학습한 ERM을 불러올 path
        model = TransAm(args).to(args.device)

        args.criterion = nn.MSELoss()  # Loss function ###
        args.criterion_eval = nn.L1Loss()
        args.optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        args.scheduler = torch.optim.lr_scheduler.StepLR(args.optimizer, 1.0, gamma=0.95)

        with open(f"data/{args.method}_{args.mode}_{args.sampling}_{args.seed}.pkl", 'rb') as f:
            tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader = pickle.load(f)
    else:
        ckpt_path = f"./best_val/{local_time}_{name}.pt" ###

    best_val_loss = 99999
    best_group_loss = 99999
    best_worst_loss = 99999

    model.load_state_dict(torch.load(ckpt_path))     # 사전학습 할 모델 불러옴.

    name = "dfr_"+previous_name
    args.method = "dfr_"+args.method

    for layer, param in model.named_modules():
        if 'output_layer' in layer:
            param.requires_grad = True
            print(layer, param, "will be trained")
        else:
            param.requires_grad = False  # freeze except for output_layer

    for epoch in range(args.dfr_epoch):

        # Train DFR
        train_loss, n_tr, n_tr_hyper2, n_tr_normal, n_tr_hypo, n_tr_crisis, n_tr_prehyper = train(args, model,
                                                                                                  bal_dataset_loader)

        # Eval val DFR
        val_sbp_loss, val_dbp_loss, n_val, \
        val_hypo_sbp_loss, val_hypo_dbp_loss, n_val_hypo, \
        val_normal_sbp_loss, val_normal_dbp_loss, n_val_normal, \
        val_hyper2_sbp_loss, val_hyper2_dbp_loss, n_val_hyper2, \
        val_prehyper_sbp_loss, val_prehyper_dbp_loss, n_val_prehyper,\
        val_crisis_sbp_loss, val_crisis_dbp_loss, n_val_crisis = eval(args, model, val2_dataset_loader)

        # Calculate val loss
        val_group_avg_loss = ((val_hypo_sbp_loss + val_normal_sbp_loss + val_prehyper_sbp_loss + val_hyper2_sbp_loss + val_crisis_sbp_loss)/5 +
                    (val_hypo_dbp_loss+ val_normal_dbp_loss + val_prehyper_dbp_loss + val_hyper2_dbp_loss + val_crisis_dbp_loss)/5)

        val_worst_loss = max(val_hypo_dbp_loss+val_hypo_sbp_loss, val_prehyper_sbp_loss+val_prehyper_dbp_loss, val_normal_sbp_loss+val_normal_dbp_loss,
               val_crisis_sbp_loss+val_crisis_dbp_loss, val_hyper2_sbp_loss+val_hyper2_dbp_loss)

        # Eval test DFR
        te_sbp_loss, te_dbp_loss, n_te, \
        te_hypo_sbp_loss, te_hypo_dbp_loss, n_te_hypo, \
        te_normal_sbp_loss, te_normal_dbp_loss, n_te_normal, \
        te_prehyper_sbp_loss, te_prehyper_dbp_loss, n_te_prehyper,\
        te_hyper2_sbp_loss, te_hyper2_dbp_loss, n_te_hyper2, \
        te_crisis_sbp_loss, te_crisis_dbp_loss, n_te_crisis, = eval(args,model,te_dataset_loader)

        # Calculate test loss
        te_group_avg_loss = ((te_hypo_sbp_loss + te_normal_sbp_loss + te_prehyper_sbp_loss + te_hyper2_sbp_loss + te_crisis_sbp_loss)/5 +
                    (te_hypo_dbp_loss+ te_normal_dbp_loss + te_prehyper_dbp_loss + te_hyper2_dbp_loss + te_crisis_dbp_loss)/5)

        te_worst_loss = max(te_hypo_dbp_loss+te_hypo_sbp_loss, te_prehyper_sbp_loss+te_prehyper_dbp_loss, te_normal_sbp_loss+te_normal_dbp_loss,
               te_crisis_sbp_loss+te_crisis_dbp_loss, te_hyper2_sbp_loss+te_hyper2_dbp_loss)

        # logging
        train_log = f"Epoch: {epoch}, Train Loss: {train_loss}, Elapsed Time: {time.time() - current_time}, Num Train: {n_tr}, \n" \
                    f"Num Tr Hypo: {n_tr_hypo},,Num Tr Normal: {n_tr_normal}, Num Tr prehyper: {n_tr_prehyper}, Num Tr Hyper2: {n_tr_hyper2}," \
                    f"Num Tr Crisis: {n_tr_crisis}, \n"

        val_log = f"Epoch: {epoch}, Val SBP Loss: {val_sbp_loss}, Val DBP Loss: {val_dbp_loss}, Num Val: {n_val},\n" \
                  f" Val Hypo-SBP Loss: {val_hypo_sbp_loss}, Val Hypo-DBP Loss: {val_hypo_dbp_loss}, Num Val Hypo: {n_val_hypo}\n" \
                  f" Val Normal-SBP Loss: {val_normal_sbp_loss}, Val Normal-DBP Loss: {val_normal_dbp_loss}, Num Val Normal: {n_val_normal}\n" \
                  f" Val prehyper-SBP Loss: {val_prehyper_sbp_loss}, Val prehyper-DBP Loss: {val_prehyper_dbp_loss}, Num Val prehyper: {n_val_prehyper}\n" \
                  f" Val Hyper2-SBP Loss: {val_hyper2_sbp_loss}, Val Hyper2-DBP Loss: {val_hyper2_dbp_loss}, Num Val Hyper2: {n_val_hyper2}\n"\
                  f" Val Crisis-SBP Loss: {val_crisis_sbp_loss}, Val Crisis-DBP Loss: {val_crisis_dbp_loss}, Num Val Crisis: {n_val_crisis}\n" \
                  f" Val Group Average Loss: {val_group_avg_loss}, Val Worst Group Loss: {val_worst_loss}"

        te_log = f"Epoch: {epoch}, Te SBP Loss: {te_sbp_loss}, Te DBP Loss: {te_dbp_loss}, Num Te: {n_te},\n" \
                 f" Te Hypo-SBP Loss: {te_hypo_sbp_loss}, Te Hypo-DBP Loss: {te_hypo_dbp_loss}, Num Te Hypo: {n_te_hypo}\n" \
                 f" Te Normal-SBP Loss: {te_normal_sbp_loss}, Te Normal-DBP Loss: {te_normal_dbp_loss}, Num Te Normal: {n_te_normal}\n" \
                 f" Te prehyper-SBP Loss: {te_prehyper_sbp_loss}, Te prehyper-DBP Loss: {te_prehyper_dbp_loss}, Num Te prehyper: {n_te_prehyper}\n" \
                 f" Te Hyper2-SBP Loss: {te_hyper2_sbp_loss}, Te Hyper2-DBP Loss: {te_hyper2_dbp_loss}, Num Te Hyper2: {n_te_hyper2}\n"\
                 f" Te Crisis-SBP Loss: {te_crisis_sbp_loss}, Te Crisis-DBP Loss: {te_crisis_dbp_loss}, Num Te Crisis: {n_te_crisis}\n" \
                 f" Te Group Average Loss: {te_group_avg_loss}, Te Worst Group Loss: {te_worst_loss}"
                
        log = "\n".join([train_log, val_log, te_log])

        if (epoch + 10) % args.printing_step == 0 or epoch == 0:
            print("-" * 20 + "\n")
            print(log)
        with open(logger_name, "a") as f:
            if epoch == 0:
                f.write("\n--------------DFR-------------\n")
            f.write(log)
        if args.epoch > args.annealing_epoch:
            ## Best Performance Model
            if val_sbp_loss + val_dbp_loss < best_val_loss:
                best_val_loss = val_sbp_loss + val_dbp_loss
                best_model_name = f"./best_val/{local_time}_{name}.pt"
                torch.save(model.state_dict(), best_model_name)
                log_dict = {"time": local_time, "model_path": best_model_name, "epoch": epoch,
                            "total_loss": te_sbp_loss + te_dbp_loss, "sbp_loss": te_sbp_loss, "dbp_loss": te_dbp_loss,
                            "total_hypo": te_hypo_dbp_loss + te_hypo_sbp_loss,
                            "total_normal": te_normal_dbp_loss + te_normal_sbp_loss,
                            "total_prehyper": te_prehyper_dbp_loss + te_prehyper_sbp_loss,
                            "total_hyper2": te_hyper2_dbp_loss + te_hyper2_sbp_loss,
                            "total_crisis": te_crisis_dbp_loss + te_crisis_sbp_loss,
                            "sbp_hypo": te_hypo_sbp_loss, "sbp_normal": te_normal_sbp_loss,
                            "sbp_prehyper": te_prehyper_sbp_loss,
                            "sbp_hyper2": te_hyper2_sbp_loss, "sbp_crisis": te_crisis_sbp_loss,
                            "dbp_hypo": te_hypo_dbp_loss, "dbp_normal": te_normal_dbp_loss,
                            "dbp_prehyper": te_prehyper_dbp_loss,
                            "dbp_hyper2": te_hyper2_dbp_loss, "dbp_crisis": te_crisis_dbp_loss,
                            "group_avg_loss": te_group_avg_loss, "worst_group_loss": te_worst_loss,
                            **vars(args)}
                del log_dict['optimizer'], log_dict['scheduler'], log_dict['device']

            ## Best Group Loss Model
            if (val_group_avg_loss < best_group_loss):
                best_group_loss = val_group_avg_loss
                best_group_model_name = f"./group_best_val/{local_time}_{name}.pt"
                torch.save(model.state_dict(), best_group_model_name)
                group_best_log_dict = {"time": local_time, "model_path": best_group_model_name, "epoch": epoch,
                            "total_loss": te_sbp_loss + te_dbp_loss, "sbp_loss": te_sbp_loss, "dbp_loss": te_dbp_loss,
                            "total_hypo": te_hypo_dbp_loss + te_hypo_sbp_loss,
                            "total_normal": te_normal_dbp_loss + te_normal_sbp_loss,
                            "total_prehyper": te_prehyper_dbp_loss + te_prehyper_sbp_loss,
                            "total_hyper2": te_hyper2_dbp_loss + te_hyper2_sbp_loss,
                            "total_crisis": te_crisis_dbp_loss + te_crisis_sbp_loss,
                            "sbp_hypo": te_hypo_sbp_loss, "sbp_normal": te_normal_sbp_loss,
                            "sbp_prehyper": te_prehyper_sbp_loss,
                            "sbp_hyper2": te_hyper2_sbp_loss, "sbp_crisis": te_crisis_sbp_loss,
                            "dbp_hypo": te_hypo_dbp_loss, "dbp_normal": te_normal_dbp_loss,
                            "dbp_prehyper": te_prehyper_dbp_loss,
                            "dbp_hyper2": te_hyper2_dbp_loss, "dbp_crisis": te_crisis_dbp_loss,
                            "group_avg_loss": te_group_avg_loss, "worst_group_loss": te_worst_loss,
                            **vars(args)}
                del group_best_log_dict['optimizer'], group_best_log_dict['scheduler'], group_best_log_dict['device']
            
            if val_worst_loss < best_worst_loss:
                ## Best Robust Model
                best_worst_loss = val_worst_loss
                best_worst_model_name = f"./best_worst_val/{local_time}_{name}.pt"
                torch.save(model.state_dict(), best_worst_model_name)
                worst_log_dict = {"time": local_time, "model_path": best_worst_model_name, "epoch": epoch,
                                  "total_loss": te_sbp_loss + te_dbp_loss, "sbp_loss": te_sbp_loss, "dbp_loss": te_dbp_loss,
                                  "total_hypo": te_hypo_dbp_loss + te_hypo_sbp_loss,
                                  "total_normal": te_normal_dbp_loss + te_normal_sbp_loss,
                                  "total_prehyper": te_prehyper_dbp_loss + te_prehyper_sbp_loss,
                                  "total_hyper2": te_hyper2_dbp_loss + te_hyper2_sbp_loss,
                                  "total_crisis": te_crisis_dbp_loss + te_crisis_sbp_loss,
                                  "sbp_hypo": te_hypo_sbp_loss, "sbp_normal": te_normal_sbp_loss,
                                  "sbp_prehyper": te_prehyper_sbp_loss,
                                  "sbp_hyper2": te_hyper2_sbp_loss, "sbp_crisis": te_crisis_sbp_loss,
                                  "dbp_hypo": te_hypo_dbp_loss, "dbp_normal": te_normal_dbp_loss,
                                  "dbp_prehyper": te_prehyper_dbp_loss,
                                  "dbp_hyper2": te_hyper2_dbp_loss, "dbp_crisis": te_crisis_dbp_loss,
                                  "group_avg_loss": te_group_avg_loss, "worst_group_loss": te_worst_loss,
                                  **vars(args)}
                del worst_log_dict['optimizer'], worst_log_dict['scheduler'], worst_log_dict['device']


# Save Results

# Avg loss val이 제일 좋은 best model
logger_path = os.path.join(os.getcwd(), "results/csv_log/best", f"{args.method}_{args.mode}_{args.sampling}_best_logger.csv")
if os.path.exists(logger_path):
    logger = pd.read_csv(logger_path)
    logger = pd.concat((logger, pd.DataFrame([log_dict])))
    logger.to_csv(logger_path, index=False)
else:
    logger = pd.DataFrame([log_dict])
    logger.to_csv(logger_path, index=False)

# group avg loss val이 제일 좋은 best model
group_best_logger_path = os.path.join(os.getcwd(), "results/csv_log/group_best", f"{args.method}_{args.mode}_{args.sampling}_group_best_logger.csv")
if os.path.exists(group_best_logger_path):
    group_best_logger = pd.read_csv(group_best_logger_path)
    group_best_logger = pd.concat((group_best_logger, pd.DataFrame([group_best_log_dict])))
    group_best_logger.to_csv(group_best_logger_path, index=False)
else:
    group_best_logger = pd.DataFrame([group_best_log_dict])
    group_best_logger.to_csv(group_best_logger_path, index=False)

# worst group loss val이 제일 좋은 best model
worst_logger_path = os.path.join(os.getcwd(), "results/csv_log/worst", f"{args.method}_{args.mode}_{args.sampling}_worst_logger.csv")
if os.path.exists(worst_logger_path):
    worst_logger = pd.read_csv(worst_logger_path)
    worst_logger = pd.concat((worst_logger, pd.DataFrame([worst_log_dict])))
    worst_logger.to_csv(worst_logger_path, index=False)
else:
    worst_logger = pd.DataFrame([worst_log_dict])
    worst_logger.to_csv(worst_logger_path, index=False)



