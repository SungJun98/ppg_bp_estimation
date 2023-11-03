'''
Test만을 진행하기 위한 python file
(Training X)
'''

import argparse
import numpy as np
import torch
import torch.nn as nn
import random
from utils import eval
from model import TransAm, ConvTransAm
import time
import pickle
import os
import pandas as pd
import datetime
import pytz

# TODO
# overall, prehyper MAE check

#-------------------------------------Argparse-------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)

# Data Load
parser.add_argument('--sampling', type=str, default="normal", choices=["normal", "upsampling", "downsampling", "small"],
                    help='Choose sampling methods to treat imbalance')  # upsampling과 downsampling은 ERM method시에만 활용
parser.add_argument('--small_ratio', type=float, default=None, help='Ratio of Training Data when using small loader')
# sampling이 small일 때, original data의 얼마 정도로 줄일 것인지 예) 0.2 : 20% 데이터만 활용
parser.add_argument('--up_small', action='store_true', help='Upsampling for small ratio set')  # small + upsampling
parser.add_argument('--down_small', action='store_true', help='Downsampling for small ratio set')  # small + downsampling

parser.add_argument('--max', type=float, default=1.0, help="preprocessing")  # ppg 변환 시 최대 ppg 값
parser.add_argument('--min', type=float, default=-1.0, help="preprocessing")  # ppg 변환 시 최소 ppg값

# Model
parser.add_argument('--mode', type=str, default='all',
                    choices=['all', 'hypo', 'normal','prehyper', 'hyper2', 'crisis'],
                    help='select group for training')  # train으로 사용할 그룹 설정/ 모든 그룹 train시 all 사용
parser.add_argument('--method', type=str, default='erm', choices=['erm', 'dro', 'vrex', 'ours-1', 'ours-2', 'ours-both'],
                    help='Choose learning method') # ours-1 : C-REx / ours-2 : D-REx / ours-both : CD-REx
parser.add_argument('--model', type=str, default='ConvTransformer', choices=['Transformer', 'ConvTransformer'])
parser.add_argument('--d_input', type=int, default=64, help="model input and PE dimension")
parser.add_argument('--d_output', type=int, default=2, help="model input and PE dimension")  # SBP, DBP 각각 추정 = 2
parser.add_argument('--d_model', type=int, default=96, help="model hidden dimension") # hidden dimension
parser.add_argument('--dropout', type=float, default=0.1, help="")
parser.add_argument('--num_layers', type=int, default=2, help="") # Transformer Encoder Layer 개수
parser.add_argument('--num_heads', type=int, default=4, help="") # Transformer 내 head 개수
parser.add_argument('--random_splits',action='store_true')

parser.add_argument('--num_filters', type=int, default=8, help='Number of filter per 3/5/7/9 size') # ConvTransformer filter 개수

parser.add_argument('--batch_size', type=int, default=64)


parser.add_argument('--resume', type=str, default=None)   # 학습 완료된 모델을 test할 때 경로와 함께 설정 (필수)

### RMSE
parser.add_argument('--rmse', action='store_true', help='Get RMSE loss') # Evaluation 기준을 RMSE로 설정 (기본은 MAE)

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.feature_size = 1 ### Input Shape : batch_size x 1000 x 1  (1000: ppg time stamp length/ 1: ppg signal)



#----------------------------------Setting---------------------------------------------------------
# 모델 저장을 위한 시간 설정
tz = pytz.timezone('Asia/Seoul') #TODO annoymous 제출시 꼭 삭제
datetime_here = datetime.datetime.now(tz)
local_time = datetime_here.strftime("%m-%d_%H-%M-%S")
args.local_time = local_time

### Control Randomness
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.use_deterministic_algorithms(True)
np.random.seed(args.seed)
random.seed(args.seed)

### Organizing
if args.rmse:
    os.makedirs('results/csv_log/rmse', exist_ok=True)
else:
    os.makedirs('results/csv_log/mae', exist_ok=True)

### Elapsed Time
current_time = time.time()


#---------------------------------Loader------------------------------------------------------------------
## Data Load
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

print("Complete to Load Data Loader")


#--------------------------------Setting------------------------------------------------------------
## Model Setting
if args.model == 'Transformer':
    model = TransAm(args).to(args.device)
elif args.model == 'ConvTransformer':
    model = ConvTransAm(args).to(args.device)
print(f"Load Model {model.model_type}")


## Set Test Loss Criterion
if args.rmse:
    import loss
    args.criterion_eval = loss.RMSELoss()
else:
    args.criterion_eval = nn.L1Loss()


## Load Pre-Trained State dict
if args.resume:
    model.load_state_dict(torch.load(args.resume))


#---------------------------------Test----------------------------------------------------------------
print("Start Test....")
te_sbp_loss, te_dbp_loss, n_te, \
        te_hypo_sbp_loss, te_hypo_dbp_loss, n_te_hypo, \
        te_normal_sbp_loss, te_normal_dbp_loss,n_te_normal, \
        te_prehyper_sbp_loss, te_prehyper_dbp_loss, n_te_prehyper,\
        te_hyper2_sbp_loss, te_hyper2_dbp_loss, n_te_hyper2, \
        te_crisis_sbp_loss, te_crisis_dbp_loss, n_te_crisis, = eval(args,model,te_dataset_loader)

te_group_avg_loss = ((te_hypo_sbp_loss + te_normal_sbp_loss + te_prehyper_sbp_loss + te_hyper2_sbp_loss + te_crisis_sbp_loss)/5 +
                (te_hypo_dbp_loss+ te_normal_dbp_loss + te_prehyper_dbp_loss + te_hyper2_dbp_loss + te_crisis_dbp_loss)/5)

te_worst_loss = max(te_hypo_dbp_loss+te_hypo_sbp_loss, te_prehyper_sbp_loss+te_prehyper_dbp_loss, te_normal_sbp_loss+te_normal_dbp_loss,
            te_crisis_sbp_loss+te_crisis_dbp_loss, te_hyper2_sbp_loss+te_hyper2_dbp_loss)


te_log_dict = {"local_time": local_time,
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



if args.rmse:
    logger_path = os.path.join(os.getcwd(), "results/csv_log/rmse", f"{args.method}_{args.mode}_{args.sampling}_test_logger.csv")
else:
    logger_path = os.path.join(os.getcwd(), "results/csv_log/mae", f"{args.method}_{args.mode}_{args.sampling}_test_logger.csv")

## group avg loss val이 제일 좋은 best model 저장
if os.path.exists(logger_path):
    group_best_logger = pd.read_csv(logger_path)
    group_best_logger = pd.concat((group_best_logger, pd.DataFrame([te_log_dict])))
    group_best_logger.to_csv(logger_path, index=False)
else:
    group_best_logger = pd.DataFrame([te_log_dict])
    group_best_logger.to_csv(logger_path, index=False)