import os 
import re
from pathlib import Path
import argparse
import torch
import torch.nn
import numpy as np
import pandas as pd
import mlflow as mf
from shutil import rmtree
from pyampd.ampd import find_peaks
from mlflow.tracking import MlflowClient
from mlflow.utils.autologging_utils.safety import try_mlflow_log
import scipy
import random
import copy
    
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def set_device(gpu_id):
    print(gpu_id)
    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_id))
        print("Using GPU: ", torch.cuda.current_device())
    else:
        n_threads = torch.get_num_threads()
        n_threads = min(n_threads, 8)
        torch.set_num_threads(n_threads)
        print("Using {} CPU Core".format(n_threads))

def get_nested_fold_idx(kfold):
    for fold_test_idx in range(kfold):
        fold_val_idx = (fold_test_idx+1)%kfold
        fold_train_idx = [fold for fold in range(kfold) if fold not in [fold_test_idx, fold_val_idx]]
        yield fold_train_idx, [fold_val_idx], [fold_test_idx]

def get_ckpt(r):
    ckpts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "restored_model_checkpoint")]
    return r.info.artifact_uri, ckpts

def mat2df(data):
    data.pop('__header__')
    data.pop('__version__')
    data.pop('__globals__')
    # convert to dataframe
    df = pd.DataFrame()
    for k, v in data.items():
        v = list(np.squeeze(v))
        # deal with trailing whitespace
        if isinstance(v[0], str):
            v = [re.sub(r"\s+$","",ele) for ele in v]
        # convert string nan to float64
        v = [np.nan if ele=='nan' else ele for ele in v]
        # df[k] = list(np.squeeze(v))
        df[k] = v
    return df
        
def norm_data(train_df, val_df, test_df, labels_feats=['patient','trial','SP', 'DP']):
    from sklearn.preprocessing import MinMaxScaler

    df_train = train_df.copy()
    df_val = val_df.copy()
    df_test = test_df.copy()

    df_train_norm=df_train[labels_feats].reset_index(drop=True)
    df_train_norm['SP'] = global_norm(df_train['SP'].values, 'SP')
    df_train_norm['DP'] = global_norm(df_train['DP'].values, 'DP')
    df_train_feats = df_train.drop(columns=labels_feats)
    feats = df_train_feats.columns

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(df_train_feats.values)
    df_train_norm = pd.concat([df_train_norm,pd.DataFrame(X_train,columns=feats)],axis=1)

    df_val_norm=df_val[labels_feats].reset_index(drop=True)
    df_val_norm['SP'] = global_norm(df_val['SP'].values, 'SP')
    df_val_norm['DP'] = global_norm(df_val['DP'].values, 'DP')
    df_val_feats = df_val.drop(columns=labels_feats)
    X_val = scaler.transform(df_val_feats.values)
    df_val_norm = pd.concat([df_val_norm,pd.DataFrame(X_val,columns=feats)],axis=1)

    df_test_norm=df_test[labels_feats].reset_index(drop=True)
    df_test_norm['SP'] = global_norm(df_test['SP'].values, 'SP')
    df_test_norm['DP'] = global_norm(df_test['DP'].values, 'DP')
    df_test_feats = df_test.drop(columns=labels_feats)
    X_test = scaler.transform(df_test_feats.values)
    df_test_norm = pd.concat([df_test_norm,pd.DataFrame(X_test,columns=feats)],axis=1)

    return df_train_norm, df_val_norm, df_test_norm
    
#%% Global Normalization
def global_norm(x, signal_type): 
    if signal_type == "SP": (x_min, x_max) = (80, 200)   # mmHg
    elif signal_type == "DP": (x_min, x_max) = (50, 110)   # mmHg
    elif signal_type == "ptt": (x_min, x_max) = (100, 900)  # 100ms - 900ms
    else: return None

    # Return values normalized between 0 and 1
    return (x - x_min) / (x_max - x_min)
    
def global_denorm(x, signal_type):
    if signal_type == "SP": (x_min, x_max) = (80, 200)   # mmHg
    elif signal_type == "DP": (x_min, x_max) = (50, 110)   # mmHg
    elif signal_type == "ptt": (x_min, x_max) = (100, 900)  # 100ms - 900ms
    else: return None

    # Return values normalized between 0 and 1
    return x * (x_max-x_min) + x_min

def glob_demm(x, config, type='SP'): 
    # sensors global max, min
    if type=='SP':
        x_min, x_max = config.param_loader.SP_min, config.param_loader.SP_max
    elif type=='DP':
        x_min, x_max = config.param_loader.DP_min, config.param_loader.DP_max
    elif type=='ppg':
        x_min, x_max = config.param_loader.ppg_min, config.param_loader.ppg_max
    elif type=='abp':
        x_min, x_max = config.param_loader.abp_min, config.param_loader.abp_max
    return x * (x_max-x_min) + x_min

def glob_mm(x, config, type='SP'): 
    # sensors global max, min
    if type=='SP':
        x_min, x_max = config.param_loader.SP_min, config.param_loader.SP_max
    elif type=='DP':
        x_min, x_max = config.param_loader.DP_min, config.param_loader.DP_max
    elif type=='ppg':
        x_min, x_max = config.param_loader.ppg_min, config.param_loader.ppg_max
    elif type=='abp':
        x_min, x_max = config.param_loader.abp_min, config.param_loader.abp_max
    return (x - x_min) / (x_max - x_min)

def glob_dez(x, config, type='SP'): 
    # sensors global max, min
    if type=='SP':
        x_mean, x_std = config.param_loader.SP_mean, config.param_loader.SP_std
    elif type=='DP':
        x_mean, x_std = config.param_loader.DP_mean, config.param_loader.DP_std
    elif type=='ppg':
        x_mean, x_std = config.param_loader.ppg_mean, config.param_loader.ppg_std
    elif type=='abp':
        x_mean, x_std = config.param_loader.abp_mean, config.param_loader.abp_std
    return x * (x_std + 1e-6) + x_mean

def glob_z(x, config, type='sbp'): 
    # sensors global max, min
    if type=='SP':
        x_mean, x_std = config.param_loader.SP_mean, config.param_loader.SP_std
    elif type=='DP':
        x_mean, x_std = config.param_loader.DP_mean, config.param_loader.DP_std
    elif type=='ppg':
        x_mean, x_std = config.param_loader.ppg_mean, config.param_loader.ppg_std
    elif type=='abp':
        x_mean, x_std = config.param_loader.abp_mean, config.param_loader.abp_std
    return (x - x_mean)/(x_std + 1e-6)

#%% Local normalization
def loc_mm(x,config, type='SP'):
    return (x - x.min())/(x.max() - x.min() + 1e-6)

def loc_demm(x,config, type='SP'):
    return x * (x.max() - x.min() + 1e-6) + x.min()

def loc_z(x,config, type='SP'):
    return (x - x.mean())/(x.std() + 1e-6)

def loc_dez(x,config, type='SP'):
    return x * (x.std() + 1e-6) + x.mean()

#%% Compute bps
def compute_sp_dp(sig, fs=125, pk_th=0.6):
    sig = sig.astype(np.float64)
    try:
        peaks = find_peaks(sig,fs)
    except: # if prediction is too jitering.
        peaks = find_peaks(butter_lowpass_filter(sig, 8, fs, 5),fs)
        
    try:
        valleys = find_peaks(-sig,fs)
    except: # if prediction is too jitering.
        valleys = find_peaks(-butter_lowpass_filter(sig, 8, fs, 5),fs)
    
    sp, dp = -1 , -1
    flag1 = False
    flag2 = False
    
    ### Remove first or last if equal to 0 or len(sig)-1
    if peaks[0] == 0:
        peaks = peaks[1:]
    if valleys[0] == 0:
        valleys = valleys[1:]
    
    if peaks[-1] == len(sig)-1:
        peaks = peaks[:-1]
    if valleys[-1] == len(sig)-1:
        valleys = valleys[:-1]
    
    '''
    ### HERE WE SHOULD REMOVE THE FIRST AND LAST PEAK/VALLEY
    if peaks[0] < valleys[0]:
        peaks = peaks[1:]
    else:
        valleys = valleys[1:]
        
    if peaks[-1] > valleys[-1]:
        peaks = peaks[:-1]
    else:
        valleys = valleys[:-1]
    '''
    
    ### START AND END IN VALLEYS
    while len(peaks)!=0 and peaks[0] < valleys[0]:
        peaks = peaks[1:]
    
    while len(peaks)!=0 and peaks[-1] > valleys[-1]:
        peaks = peaks[:-1]
    
    ## Remove consecutive peaks with one considerably under the other
    new_peaks = []
    mean_vly_amp = np.mean(sig[valleys])
    if len(peaks)==1:
        new_peaks = peaks
    else:
        # define base case:

        for i in range(len(peaks)-1):
            if sig[peaks[i]]-mean_vly_amp > (sig[peaks[i+1]]-mean_vly_amp)*pk_th:
                new_peaks.append(peaks[i])
                break

        for j in range(i+1,len(peaks)):
            if sig[peaks[j]]-mean_vly_amp > (sig[new_peaks[-1]]-mean_vly_amp)*pk_th:
                new_peaks.append(peaks[j])
                
        if not np.array_equal(peaks,new_peaks):
            flag1 = True
            
        if len(valleys)-1 != len(new_peaks):
            flag2 = True
            
        if len(valleys)-1 == len(new_peaks):
            for i in range(len(valleys)-1):
                if not(valleys[i] < new_peaks[i] and new_peaks[i] < valleys[i+1]):
                    flag2 = True
        
        
    return np.median(sig[new_peaks]), np.median(sig[valleys]), flag1, flag2, new_peaks, valleys

def butter_lowpass_filter(data, lowcut, fs, order):
    """ Butterworth band-pass filter
    Parameters
    ----------
    data : array
        Signal to be filtered.
    lowcut : float
        Frequency lowcut for the filter. 
    highcut : float}
        Frequency highcut for the filter.
    fs : float
        Sampling rate.
    order: int
        Filter's order.

    Returns
    -------
    array
        Signal filtered with butterworth algorithm.
    """  
    nyq = fs * 0.5  # https://en.wikipedia.org/wiki/Nyquist_frequency
    lowcut = lowcut / nyq  # Normalize
    #highcut = highcut / nyq
    # Numerator (b) and denominator (a) polynomials of the IIR filter
    b, a = scipy.signal.butter(order, lowcut, btype='low', analog=False)
    return scipy.signal.filtfilt(b, a, data)
    
def get_bp_pk_vly_mask(data):
    try:
        _,_,_,_,pks, vlys = compute_sp_dp(data, 125, pk_th=0.6)

        pk_mask = np.zeros_like(data)
        vly_mask = np.zeros_like(data)
        pk_mask[pks] = 1
        vly_mask[vlys] = 1

    except:
        # print("!!! No peaks and vlys found for peak_vly_mask !!!")
        pk_mask = np.zeros_like(data)
        vly_mask = np.zeros_like(data)
    
    return np.array(pk_mask, dtype=bool), np.array(vly_mask, dtype=bool)

#%% Compute statistics for normalization
def cal_statistics(config, all_df):
    import pandas as pd
    from omegaconf import OmegaConf,open_dict
    all_df = pd.concat(all_df)
    OmegaConf.set_struct(config, True)

    with open_dict(config):
        for x in ['SP', 'DP']:
            config.param_loader[f'{x}_mean'] = float(all_df[x].mean())
            config.param_loader[f'{x}_std'] = float(all_df[x].std())
            config.param_loader[f'{x}_min'] = float(all_df[x].min())
            config.param_loader[f'{x}_max'] = float(all_df[x].max())
        
        # ppg
        if config.param_loader.ppg_norm.startswith('glob'):
            config.param_loader[f'ppg_mean'] = float(np.vstack(all_df['signal']).mean())
            config.param_loader[f'ppg_std'] = float(np.vstack(all_df['signal']).std())
            config.param_loader[f'ppg_min'] = float(np.vstack(all_df['signal']).min())
            config.param_loader[f'ppg_max'] = float(np.vstack(all_df['signal']).max())
            
        if 'abp_signal' in all_df.columns:
            config.param_loader[f'abp_mean'] = float(np.vstack(all_df['abp_signal']).mean())
            config.param_loader[f'abp_std'] = float(np.vstack(all_df['abp_signal']).std())
            config.param_loader[f'abp_min'] = float(np.vstack(all_df['abp_signal']).min())
            config.param_loader[f'abp_max'] = float(np.vstack(all_df['abp_signal']).max())
        else: #dummy stats
            config.param_loader[f'abp_mean'] = 0.0
            config.param_loader[f'abp_std'] = 1.0
            config.param_loader[f'abp_min'] = 0.0
            config.param_loader[f'abp_max'] = 1.0
            
            
    return config

#%% Compute metric
def cal_metric(err_dict, metric={}, mode='val'):
    for k, v in err_dict.items():
        metric[f'{k}_mae'] = np.mean(np.abs(v))
        metric[f'{k}_std'] = np.std(v)
        metric[f'{k}_me'] = np.mean(v)
    metric = {f'{mode}/{k}':round(v.item(),3) for k,v in metric.items()}
    return metric

#%% print/logging tools
def print_criterion(sbps, dbps):
    print("The percentage of SBP above 160: (0.10)", len(np.where(sbps>=160)[0])/len(sbps)) 
    print("The percentage of SBP above 140: (0.20)", len(np.where(sbps>=140)[0])/len(sbps)) 
    print("The percentage of SBP below 100: (0.10)", len(np.where(sbps<=100)[0])/len(sbps)) 
    print("The percentage of DBP above 100: (0.05)", len(np.where(dbps>=100)[0])/len(dbps)) 
    print("The percentage of DBP above 85: (0.20)", len(np.where(dbps>=85)[0])/len(dbps)) 
    print("The percentage of DBP below 70: (0.10)", len(np.where(dbps<=70)[0])/len(dbps)) 
    print("The percentage of DBP below 60: (0.05)", len(np.where(dbps<=60)[0])/len(dbps)) 

def get_cv_logits_metrics(fold_errors, loader, pred_sbp, pred_dbp, pred_abp, 
                            true_sbp, true_dbp, true_abp, 
                            sbp_naive, dbp_naive, mode="val"):

    fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
    fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
    fold_errors[f"{mode}_sbp_naive"].append([sbp_naive])
    fold_errors[f"{mode}_sbp_pred"].append([pred_sbp])
    fold_errors[f"{mode}_sbp_label"].append([true_sbp])
    fold_errors[f"{mode}_dbp_naive"].append([dbp_naive])
    fold_errors[f"{mode}_dbp_pred"].append([pred_dbp])
    fold_errors[f"{mode}_dbp_label"].append([true_dbp])
    fold_errors[f"{mode}_abp_true"].append([true_abp])
    fold_errors[f"{mode}_abp_pred"].append([pred_abp])

#%% mlflow
def init_mlflow(config):
    mf.set_tracking_uri(str(Path(config.path.mlflow_dir).absolute()))  # set up connection
    mf.set_experiment(config.exp.exp_name)          # set the experiment

def log_params_mlflow(config):
    mf.log_params(config.get("exp"))
    # mf.log_params(config.get("param_feature"))
    try_mlflow_log(mf.log_params, config.get("param_preprocess"))
    try_mlflow_log(mf.log_params, config.get("param_trainer"))
    try_mlflow_log(mf.log_params, config.get("param_early_stop"))
    mf.log_params(config.get("param_loader"))
    # mf.log_params(config.get("param_trainer"))
    # mf.log_params(config.get("param_early_stop"))
    if config.get("param_aug"):
        if config.param_aug.get("filter"):
            for k,v in dict(config.param_aug.filter).items():
                mf.log_params({k:v})
    # mf.log_params(config.get("param_aug"))
    mf.log_params(config.get("param_model"))

def log_config(config_path):
    # mf.log_artifact(os.path.join(os.getcwd(), 'core/config/unet_sensors_5s.yaml'))
    # mf.log_dict(config, "config.yaml")
    mf.log_artifact(config_path)

def log_hydra_mlflow(name):
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/config.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/hydra.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), '.hydra/overrides.yaml'))
    mf.log_artifact(os.path.join(os.getcwd(), f'{name}.log'))
    rmtree(os.path.join(os.getcwd()))


def remove_outlier(list_of_df):
    output_list = []
    for df in list_of_df:
        df_ = df[(80<df.SP) & (df.SP<200) & (40<df.DP) & (df.DP<130)].reset_index()
        output_list.append(df_)
    return output_list

def group_annot(list_of_df):
    output_list = []
    for df in list_of_df:
        df['group'] = 100
        df.loc[((180 <= df.SP) | (120 <= df.DP)), 'group'] = 4  # Crisis
        df.loc[(((140 <= df.SP) & (df.SP < 180)) | ((90 <= df.DP) & (df.DP < 120))) & (df.group==100) , 'group'] = 3  # Hyper2
        df.loc[(((120 <= df.SP) & (df.SP < 140)) | ((80 <= df.DP) & (df.DP < 90))) & (df.group==100) , 'group'] = 2  # Prehyper
        df.loc[(((90 <= df.SP) & (df.SP < 120)) | ((60 <= df.DP) & (df.DP < 80))) & (df.group==100) , 'group'] = 1  # Normal
        df.loc[((df.SP < 90) | (df.DP < 60)) & (df.group==100) , 'group'] = 0 # Hypo
        value_counts = df['group'].value_counts()
        remain = value_counts.get(100, 0)
        if remain:
            assert 1==2, "Annotating Group is Failed"
        output_list.append(df)
    return output_list

def per_group_loss(losses, group):
    '''
    losses: (bs,2)
    group: (bs,1)
    '''
    group_type = torch.arange(5).to(group.device)
    group_map = group==group_type # (bs, 5)
    group_count = group_map.sum(0)
    full_group_loss = losses.unsqueeze(2)*group_map.unsqueeze(1) # (bs, 2, 5)
    per_group = full_group_loss.mean(0)
    return per_group, group_count

def reversed_total_group_count(df):
    group = df['group']
    group = torch.tensor(np.array(group))
    total = len(group)
    group_type = torch.arange(5).unsqueeze(1)
    group_map = group_type==group
    group_count = group_map.sum(1)
    return (total/group_count).tolist()

def get_divs_per_group(df, config):
    from torch.distributions import normal, kl_divergence

    group = torch.tensor(np.array(df['group']))
    sbp = torch.tensor(np.array(df['SP']))
    sbp = glob_mm(sbp, config, type="SP") # [0,1]
    dbp = torch.tensor(np.array(df['DP']))
    dbp = glob_mm(dbp, config, type="DP") # [0,1]
    
    if config.tukey:
        if config.beta != 0:
            sbp = torch.pow(sbp, config.beta)
            dbp = torch.pow(dbp, config.beta)
        elif config.beta == 0:
            sbp = torch.log(sbp)
            dbp = torch.log(dbp)
        else:
            sbp = (-1) * torch.pow(sbp, confg.beta)
            dbp = (-1) * torch.pow(dbp, config.beta)

    group_type = torch.arange(5).unsqueeze(1).float()
    group_map = (group_type==group).float()
    
    sbp_mean = sbp.mean(0) ; dbp_mean = dbp.mean(0)
    sbp_var = torch.var(sbp) ; dbp_var = torch.var(dbp)
    
    sbp_tot_dist = normal.Normal(sbp_mean, sbp_var)
    dbp_tot_dist = normal.Normal(dbp_mean, dbp_var)

    sbp_div_list = [] ; dbp_div_list = []
    for i in range(len(group_type)): # Hypo, Normal, Prehyper, hyper2, crisis
        idx = (group_map[i].nonzero(as_tuple=True)[0])
        group_sbp = sbp[idx]; group_dbp = dbp[idx]
        
        group_sbp_mean = group_sbp.mean(0)
        group_sbp_var = torch.var(group_sbp)
        if group_sbp_var == 0:
            group_sbp_var = 1e-5
        sbp_group_dist = normal.Normal(group_sbp_mean, group_sbp_var)
        sbp_kld = (kl_divergence(sbp_group_dist, sbp_tot_dist) + kl_divergence(sbp_tot_dist, sbp_group_dist)) / 2

        sbp_div_list.append(sbp_kld.item())

        group_dbp_mean = group_dbp.mean(0)
        group_dbp_var = torch.var(group_dbp)
        if group_dbp_var == 0:
            group_dbp_var = 1e-5
        dbp_group_dist = normal.Normal(group_dbp_mean, group_dbp_var)
        dbp_kld = (kl_divergence(dbp_group_dist, dbp_tot_dist) + kl_divergence(dbp_tot_dist, dbp_group_dist)) / 2
        dbp_div_list.append(dbp_kld.item())

    # Scaling
    sbp_div_list = [div/sum(sbp_div_list) for div in sbp_div_list]
    dbp_div_list = [div/sum(dbp_div_list) for div in dbp_div_list]

    div_list = {"sbp": sbp_div_list, "dbp": dbp_div_list}
    return div_list

def group_time_cutmix_all(ppg, y, group):
      
    # Creat Copy
    ppg_s = torch.zeros_like(ppg)     # bs, 1, length
    y_s = torch.zeros_like(y)         # bs, 2
    group_s = torch.zeros_like(group)   # bs
    sig_len = ppg_s.shape[-1]         #

    lamb = torch.rand(len(ppg)).to(ppg_s.device)       # bs
    lamb_f = torch.round(sig_len*lamb).int() 

    # Creat Mask
    mask_a = []
    for i in lamb_f :
        mask_a.append((i >= torch.range(1,sig_len).to(ppg_s.device)).float())
    mask_a = torch.stack(mask_a).unsqueeze(1) # bs, 1, length
    mask_b = 1-mask_a

    # Group Inform
    group_type = torch.arange(5).unsqueeze(1).float().to(group.device)
    group_map = (group_type == group).float()
    idx_0 = (group_map[0].nonzero(as_tuple=True))[0]
    idx_1 = (group_map[1].nonzero(as_tuple=True))[0]
    idx_2 = (group_map[2].nonzero(as_tuple=True))[0]
    idx_3 = (group_map[3].nonzero(as_tuple=True))[0]
    idx_4 = (group_map[4].nonzero(as_tuple=True))[0]

    len_0 = len(idx_0)
    len_1 = len(idx_1)
    len_2 = len(idx_2)
    len_3 = len(idx_3)
    len_4 = len(idx_4)



    pg_s, y_s, group_s = one_group(
                                    ppg, y, group, ppg_s, y_s, group_s, 
                                    idx_0, lamb,0, len_0, mask_a, mask_b
                                                    )
    pg_s, y_s, group_s = one_group(
                                   ppg, y, group, ppg_s, y_s, group_s, 
                                   idx_1, lamb,len_0, len_0+len_1, mask_a, mask_b
                                                    )
    pg_s, y_s, group_s = one_group(
                                   ppg, y, group, ppg_s, y_s, group_s, 
                                   idx_2, lamb, len_0+len_1, len_0+len_1+len_2, mask_a, mask_b
                                                    )
    pg_s, y_s, group_s = one_group(
                                   ppg, y, group, ppg_s, y_s, group_s, 
                                   idx_3, lamb, len_0+len_1+len_2, len_0+len_1+len_2+len_3, mask_a, mask_b
                                                    )
    pg_s, y_s, group_s = one_group(
                                    ppg, y, group, ppg_s, y_s, group_s, 
                                    idx_4, lamb,len_0+len_1+len_2+len_3, len_0+len_1+len_2+len_3+len_4, mask_a, mask_b
                                                    )


    mixed_ppg = torch.cat((ppg, ppg_s),dim=0)
    mixed_y = torch.cat((y, y_s),dim=0)
    mixed_group = torch.cat((group, group_s),dim=0)

    return mixed_ppg, mixed_y, mixed_group

def one_group(ppg, y, group, ppg_s, y_s, group_s, idx, lamb,a, b, mask_a, mask_b):
    idx_ = torch.randperm(len(idx))
    idx_ = idx[idx_]

    target_a = ppg[idx] 
    target_b = ppg[idx_] 

    y_mixed = lamb[a:b].unsqueeze(1)*y[idx] + (1-lamb[a:b].unsqueeze(1))*y[idx_]

    ppg_s[a:b] =  target_a*mask_a[a:b] + target_b*mask_b[a:b]   
    y_s[a:b] = y_mixed
    group_s[a:b] = group[idx_]
    return ppg_s, y_s, group_s

