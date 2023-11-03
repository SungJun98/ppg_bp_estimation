import glob
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import random
import math
from natsort import natsorted
from utils_dro import compute_group_avg, compute_robust_loss
from torch.utils.data.sampler import WeightedRandomSampler
from time_cutmix import group_time_cutmix_all

class TensorData(Dataset):
    def __init__(self, sbp_data, dbp_data, time_data, ppg_data, type_data):
        # print(sbp_data.type)
        self.sbp_data = torch.FloatTensor(sbp_data)
        self.dbp_data = torch.FloatTensor(dbp_data)
        self.time_data = torch.FloatTensor(time_data)
        self.ppg_data = torch.FloatTensor(ppg_data)
        self.type_data = torch.FloatTensor(type_data)
        self.len = self.sbp_data.shape[0]

    def __getitem__(self, index):
        return self.sbp_data[index], self.dbp_data[index], self.time_data[index], torch.unsqueeze(self.ppg_data[index],
                                                                                                  -1), self.type_data[
                   index]

    def __len__(self):
        return self.len


def preprocessing(args, sbp, dbp, ppg):  ## default: args.max = +1 , args.min = -1
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    sbp = (sbp - 80) / (200 - 80)
    sbp = sbp * (args.max - args.min) + args.min

    dbp = (dbp - 40) / (120 - 40)
    dbp = dbp * (args.max - args.min) + args.min
    ppg = np.asarray(ppg)
    ppg = (ppg - ppg.min(axis=0)) / (ppg.max(axis=0) - ppg.min(axis=0))
    ppg = ppg * (args.max - args.min) + args.min
    ppg = list(ppg)
    return sbp, dbp, ppg


def assign_type(args, data_dir):
    file_list, sbp_list, dbp_list, time_list, ppg_list, type_list = [], [], [], [], [], []

    for meta_data in tqdm(os.listdir(args.meta_data_path)):
        with open(f'{args.meta_data_path}/{meta_data}', "r") as meta:
            df_meta = pd.read_csv(meta, index_col=0)
            if df_meta.shape[0] == 0:
                continue
            patientID = df_meta.iloc[0, 0]

            # Sampling max 100 segment per patient
            count = 0
            df_meta = df_meta.sample(frac=1, random_state=args.seed).reset_index(drop=True) # shuffling and reset index
            for row in range(df_meta.shape[0]):
                if count == args.max_per_patient:
                    break
                segmentID = df_meta.iloc[row, 2]
                file = f'{args.data_path}{patientID}/{segmentID}.csv'

                sbp = df_meta.iloc[row, 5]
                dbp = df_meta.iloc[row, 6]
                with open(file, "r") as f:
                    content = pd.read_csv(f, header=None)
                    time = list(content.iloc[:, 0])
                    ppg = list(content.iloc[:, 1])

                    # Assign Domain
                    # https://www.vertex42.com/ExcelTemplates/blood-pressure-chart.html - High vs Low Blood Pressure Chart
                    # Order for if-elif-else should be considered
                    if (sbp <= 80) or (sbp >= 200) or (dbp >= 130) or (dbp <= 40) or np.isnan(sbp) or np.isnan(ppg).sum():  # Not considered
                        continue
                    elif (180 <= sbp) or (120 <= dbp):
                        type = 4  # 'crisis'
                    elif (140 <= sbp) or (90 <= dbp):
                        type = 3  # 'hyper2'
                    elif (120 <= sbp) or (80 <= dbp):
                        type = 2  # 'prehyper.'  <= Prehypertension + Hyper1
                    elif (90 <= sbp) or (60 <= dbp):
                        type = 1  # 'normal'
                    elif (sbp < 90) or (dbp < 60):
                        type = 0  # 'hypo'
                    assert 80 <= sbp <= 200 and 40 <= dbp <= 130
                    count += 1
                    if args.mode == 'all':
                        file_list.append(file)
                        sbp, dbp, ppg = preprocessing(args, sbp, dbp, ppg)
                        sbp_list.append(sbp)
                        dbp_list.append(dbp)
                        time_list.append(time)
                        ppg_list.append(ppg)
                        type_list.append(type)
                    elif (args.mode == 'hypo') and (type == 0):
                        file_list.append(file)
                        sbp, dbp, ppg = preprocessing(args, sbp, dbp, ppg)
                        sbp_list.append(sbp)
                        dbp_list.append(dbp)
                        time_list.append(time)
                        ppg_list.append(ppg)
                        type_list.append(type)
                    elif (args.mode == 'normal') and (type == 1):
                        file_list.append(file)
                        sbp, dbp, ppg = preprocessing(args, sbp, dbp, ppg)
                        sbp_list.append(sbp)
                        dbp_list.append(dbp)
                        time_list.append(time)
                        ppg_list.append(ppg)
                        type_list.append(type)
                    elif (args.mode == 'prehyper') and (type == 2):
                        file_list.append(file)
                        sbp, dbp, ppg = preprocessing(args, sbp, dbp, ppg)
                        sbp_list.append(sbp)
                        dbp_list.append(dbp)
                        time_list.append(time)
                        ppg_list.append(ppg)
                        type_list.append(type)
                    elif (args.mode == 'hyper2') and (type == 3):
                        file_list.append(file)
                        sbp, dbp, ppg = preprocessing(args, sbp, dbp, ppg)
                        sbp_list.append(sbp)
                        dbp_list.append(dbp)
                        time_list.append(time)
                        ppg_list.append(ppg)
                        type_list.append(type)
                    elif (args.mode == 'crisis') and (type == 4):
                        file_list.append(file)
                        sbp, dbp, ppg = preprocessing(args, sbp, dbp, ppg)
                        sbp_list.append(sbp)
                        dbp_list.append(dbp)
                        time_list.append(time)
                        ppg_list.append(ppg)
                        type_list.append(type)
                    assert len(ppg) == len(time) == 1000
    return file_list, sbp_list, dbp_list, time_list, ppg_list, type_list


def shuffling(seed, *shuffee):
    for i in shuffee:
        random.seed(seed)
        random.shuffle(i)


def assign_sampling(args, file_list, sbp_list, dbp_list, time_list, ppg_list, type_list):
    ### different sampling
    if (args.sampling == "upsampling" or args.sampling == "downsampling") and args.mode == "all":
        new_file_list, new_sbp_list, new_dbp_list, new_time_list, new_ppg_list, new_type_list = [], [], [], [], [], []
        # indexing per type
        total_ids = []
        ids_hypo, ids_normal, ids_prehyper, ids_hyper2, ids_crisis = [], [], [], [], []
        for i in range(len(type_list)):
            if type_list[i] == 0:  ## hypo
                ids_hypo.append(i)
            elif type_list[i] == 1:  ## normal
                ids_normal.append(i)
            elif type_list[i] == 2:  ## prehyper
                ids_prehyper.append(i)
            elif type_list[i] == 3:  ## hyper2
                ids_hyper2.append(i)
            elif type_list[i] == 4:  ## crisis
                ids_crisis.append(i)

        ids_len = [len(ids_hypo), len(ids_normal), len(ids_prehyper), len(ids_hyper2), len(ids_crisis)]

        ## upsampling
        if args.sampling == "upsampling":
            ref = max(ids_len)
            # choose indices for upsampling
            for ids in [ids_hypo, ids_normal, ids_prehyper, ids_hyper2, ids_crisis]:
                reff = math.ceil(ref / len(ids))
                per_upsampled = ids * reff
                upsampled = per_upsampled[:ref]
                total_ids += upsampled

            # sample from indices
            for i in total_ids:
                new_file_list.append(file_list[i])
                new_sbp_list.append(sbp_list[i])
                new_dbp_list.append(dbp_list[i])
                new_time_list.append(time_list[i])
                new_ppg_list.append(ppg_list[i])
                new_type_list.append(type_list[i])

        ## downsampling
        if args.sampling == "downsampling":
            ref = min(ids_len)
            # choose indices for downsampling
            for ids in [ids_hypo, ids_normal, ids_prehyper, ids_hyper2, ids_crisis]:
                total_ids += ids[:ref]

            # sample from indices
            for i in total_ids:
                new_file_list.append(file_list[i])
                new_sbp_list.append(sbp_list[i])
                new_dbp_list.append(dbp_list[i])
                new_time_list.append(time_list[i])
                new_ppg_list.append(ppg_list[i])
                new_type_list.append(type_list[i])

    ## Making small loader
    elif args.sampling == "small" and args.mode == "all":
        assert 0 <= args.small_ratio <= 1, "Invalid small loader ratio."
        num_total = len(file_list)
        sample_interval = num_total//math.floor(num_total*args.small_ratio)
        new_file_list = file_list[::sample_interval]
        new_sbp_list = sbp_list[::sample_interval]
        new_dbp_list = dbp_list[::sample_interval]
        new_time_list = time_list[::sample_interval]
        new_ppg_list = ppg_list[::sample_interval]
        new_type_list = type_list[::sample_interval]

        ###########################################################################
        if (args.down_small == True) or (args.up_small == True):
            file_list = new_file_list
            sbp_list = new_sbp_list
            dbp_list = new_dbp_list
            time_list = new_time_list
            ppg_list = new_ppg_list
            type_list = new_type_list

            new_file_list, new_sbp_list, new_dbp_list, new_time_list, new_ppg_list, new_type_list = [], [], [], [], [], []
            # indexing per type
            total_ids = []
            ids_hypo, ids_normal, ids_prehyper, ids_hyper2, ids_crisis = [], [], [], [], []
            for i in range(len(type_list)):
                if type_list[i] == 0:  ## hypo
                    ids_hypo.append(i)
                elif type_list[i] == 1:  ## normal
                    ids_normal.append(i)
                elif type_list[i] == 2:  ## prehyper
                    ids_prehyper.append(i)
                elif type_list[i] == 3:  ## hyper2
                    ids_hyper2.append(i)
                elif type_list[i] == 4:  ## crisis
                    ids_crisis.append(i)

            ids_len = [len(ids_hypo), len(ids_normal), len(ids_prehyper), len(ids_hyper2), len(ids_crisis)]

            ## upsampling
            if args.up_small == True:
                ref = max(ids_len)
                # choose indices for upsampling
                for ids in [ids_hypo, ids_normal, ids_prehyper, ids_hyper2, ids_crisis]:
                    reff = math.ceil(ref / len(ids))
                    per_upsampled = ids * reff
                    upsampled = per_upsampled[:ref]
                    total_ids += upsampled

                # sample from indices
                for i in total_ids:
                    new_file_list.append(file_list[i])
                    new_sbp_list.append(sbp_list[i])
                    new_dbp_list.append(dbp_list[i])
                    new_time_list.append(time_list[i])
                    new_ppg_list.append(ppg_list[i])
                    new_type_list.append(type_list[i])

            ## downsampling
            if args.down_small == True:
                ref = min(ids_len)
                # choose indices for downsampling
                for ids in [ids_hypo, ids_normal, ids_prehyper, ids_hyper2, ids_crisis]:
                    total_ids += ids[:ref]

                # sample from indices
                for i in total_ids:
                    new_file_list.append(file_list[i])
                    new_sbp_list.append(sbp_list[i])
                    new_dbp_list.append(dbp_list[i])
                    new_time_list.append(time_list[i])
                    new_ppg_list.append(ppg_list[i])
                    new_type_list.append(type_list[i])
        ###########################################################################

    file_list, sbp_list, dbp_list, time_list, ppg_list, type_list = new_file_list, new_sbp_list, new_dbp_list, new_time_list, new_ppg_list, new_type_list
    return file_list, sbp_list, dbp_list, time_list, ppg_list, type_list



def make_loader(args, sbp_list, dbp_list, time_list, ppg_list, type_list, train):
    data = TensorData(sbp_list, dbp_list, time_list, ppg_list, type_list)
    if (train==True) and (args.erm_loader==True): # erm_loader = True
        shuffle=True
        sampler=None

    elif (train == True) and (args.method in ['dro', 'vrex', 'crex', 'drex', 'cdrex']):
        # Calculate sampling weight per group for DRO and V-REx
        group_idx = torch.arange(5).unsqueeze(1)
        group_counts = (group_idx==torch.tensor(type_list)).sum(1)
        group_samp_weights = len(ppg_list)/group_counts
        samp_weight = group_samp_weights[type_list]

        shuffle = False
        sampler = WeightedRandomSampler(samp_weight, len(ppg_list), replacement=True)

    elif (train ==True) and (args.method == 'erm'):
        shuffle=True
        sampler=None
    else:
        shuffle=False
        sampler=None

    dataset_loader = DataLoader(data, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler)
    return dataset_loader



def get_dataset(args):
    dir_list = natsorted(os.listdir(args.data_path))  ## natsorted to load files in order
    random.shuffle(dir_list)

    ## Assign Type
    file_list, sbp_list, dbp_list, time_list, ppg_list, type_list = assign_type(args, dir_list)
    shuffling(args.seed, file_list, sbp_list, dbp_list, time_list, ppg_list, type_list)

    tr_split_ids = int(len(file_list) * args.tr_ratio)
    print(f"Number of Trainig data Before sampling : {tr_split_ids}")
    ## Assign sampling
    if args.sampling != "normal":

        sam_file_list = file_list[:tr_split_ids]
        sam_sbp_list = sbp_list[:tr_split_ids]
        sam_dbp_list = dbp_list[:tr_split_ids]
        sam_time_list = time_list[:tr_split_ids]
        sam_ppg_list = ppg_list[:tr_split_ids]
        sam_type_list = type_list[:tr_split_ids]
        sam_file_list, sam_sbp_list, sam_dbp_list, sam_time_list, sam_ppg_list, sam_type_list = assign_sampling(args, sam_file_list,
                                                                                        sam_sbp_list, sam_dbp_list,
                                                                                        sam_time_list, sam_ppg_list,
                                                                                        sam_type_list)
        file_list = sam_file_list + file_list[tr_split_ids:]
        sbp_list = sam_sbp_list + sbp_list[tr_split_ids:]
        dbp_list = sam_dbp_list + dbp_list[tr_split_ids:]
        time_list = sam_time_list + time_list[tr_split_ids:]
        ppg_list = sam_ppg_list + ppg_list[tr_split_ids:]
        type_list = sam_type_list + type_list[tr_split_ids:]

        # Update tr_split_ids (number of training data) for oversampling or downsampling
        tr_split_ids = len(sam_file_list)

    assert len(file_list) == len(sbp_list) == len(dbp_list) == len(ppg_list) == len(type_list)
    

    ## Assign indices per type for val2, bal, test data set
    ids_0 = []; ids_1 = []; ids_2 = []; ids_3 = []; ids_4 = []
    for i in range(len(type_list[tr_split_ids:])):
        if type_list[tr_split_ids:][i] == 0:  ## hypo
            ids_0.append(i)
        elif type_list[tr_split_ids:][i] == 1:  ## normal
            ids_1.append(i)
        elif type_list[tr_split_ids:][i] == 2:  ## prehyper
            ids_2.append(i)
        elif type_list[tr_split_ids:][i] == 3:  ## hyper2
            ids_3.append(i)
        elif type_list[tr_split_ids:][i] == 4:  ## crisis
            ids_4.append(i)
    assert len(file_list[tr_split_ids:]) == len(ids_0) + len(ids_1) + len(ids_2) + len(ids_3) + len(ids_4)

    ## training / validation 2 / balance / test dataset split (validation 2 + balance => validation for phase 1) for DFR
    tr_sbp_list, tr_dbp_list, tr_time_list, tr_ppg_list, tr_type_list = [], [], [], [], []
    val2_sbp_list, val2_dbp_list, val2_time_list, val2_ppg_list, val2_type_list = [], [], [], [], []
    bal_sbp_list, bal_dbp_list, bal_time_list, bal_ppg_list, bal_type_list = [], [], [], [], []
    te_sbp_list, te_dbp_list, te_time_list, te_ppg_list, te_type_list = [], [], [], [], []

    ## Get Tr data
    tr_sbp_list = sbp_list[:tr_split_ids] 
    tr_dbp_list = dbp_list[:tr_split_ids]
    tr_time_list = time_list[:tr_split_ids]
    tr_ppg_list = ppg_list[:tr_split_ids]
    tr_type_list = type_list[:tr_split_ids]
    
    print(f"Number of Type 0 Tr Data : {tr_type_list.count(0)}")
    print(f"Number of Type 1 Tr Data : {tr_type_list.count(1)}")
    print(f"Number of Type 2 Tr Data : {tr_type_list.count(2)}")
    print(f"Number of Type 3 Tr Data : {tr_type_list.count(3)}")
    print(f"Number of Type 4 Tr Data : {tr_type_list.count(4)}")
    print(f"Number of Trainig data After sampling : {len(tr_type_list)}")

    ## Get Val_2, Bal, Te Data
    hold_out = args.val_ratio + args.te_ratio
    for ids in [ids_0, ids_1, ids_2, ids_3, ids_4]:

        # Val_2 Data
        ids_list = ids[:int(len(ids) * (args.val_ratio/hold_out /2))]
        for val2_ids in ids_list:
            val2_sbp_list.append(sbp_list[tr_split_ids:][val2_ids])
            val2_dbp_list.append(dbp_list[tr_split_ids:][val2_ids])
            val2_time_list.append(time_list[tr_split_ids:][val2_ids])
            val2_ppg_list.append(ppg_list[tr_split_ids:][val2_ids])
            val2_type_list.append(type_list[tr_split_ids:][val2_ids])
        
        # Bal Data
        ids_list = ids[int(len(ids)*(args.val_ratio/hold_out /2)):int(len(ids)*(args.val_ratio/hold_out))]
        for bal_ids in ids_list:
            bal_sbp_list.append(sbp_list[tr_split_ids:][bal_ids])
            bal_dbp_list.append(dbp_list[tr_split_ids:][bal_ids])
            bal_time_list.append(time_list[tr_split_ids:][bal_ids])
            bal_ppg_list.append(ppg_list[tr_split_ids:][bal_ids])
            bal_type_list.append(type_list[tr_split_ids:][bal_ids])

        # Te Data
        ids_list = ids[int(len(ids) * (args.val_ratio/hold_out)):]
        for te_ids in ids_list:
            te_sbp_list.append(sbp_list[tr_split_ids:][te_ids])
            te_dbp_list.append(dbp_list[tr_split_ids:][te_ids])
            te_time_list.append(time_list[tr_split_ids:][te_ids])
            te_ppg_list.append(ppg_list[tr_split_ids:][te_ids])
            te_type_list.append(type_list[tr_split_ids:][te_ids])


    # Validation Set (for phase 1 of DFR)
    val_sbp_list = val2_sbp_list + bal_sbp_list
    val_dbp_list = val2_dbp_list + bal_dbp_list
    val_time_list = val2_time_list + bal_time_list
    val_ppg_list = val2_ppg_list + bal_ppg_list
    val_type_list = val2_type_list + bal_type_list

    print("-"*30)
    print(f"Number of Type 0 Val Data : {val_type_list.count(0)}")
    print(f"Number of Type 1 Val Data : {val_type_list.count(1)}")
    print(f"Number of Type 2 Val Data : {val_type_list.count(2)}")
    print(f"Number of Type 3 Val Data : {val_type_list.count(3)}")
    print(f"Number of Type 4 Val Data : {val_type_list.count(4)}")
    print(f"Number of Validation data After sampling : {len(val_type_list)}")

    print("-"*30)
    print(f"Number of Type 0 Te Data : {te_type_list.count(0)}")
    print(f"Number of Type 1 Te Data : {te_type_list.count(1)}")
    print(f"Number of Type 2 Te Data : {te_type_list.count(2)}")
    print(f"Number of Type 3 Te Data : {te_type_list.count(3)}")
    print(f"Number of Type 4 Te Data : {te_type_list.count(4)}")
    print(f"Number of Test data After sampling : {len(te_type_list)}")

    # Up/Downsampling for balanced dataset of DFR
    if (args.DFR == True or args.method == "erm") and args.mode == "all" and args.sampling == "normal":
        num_type_dict = {0: bal_type_list.count(0), 1: bal_type_list.count(1), 2: bal_type_list.count(2),
                         3: bal_type_list.count(3), 4: bal_type_list.count(4)}
        sorted_dict = dict(sorted(num_type_dict.items(), key=lambda x: x[1]))  # Sort ascending according to type
        assert len(bal_type_list) == sum(sorted_dict.values())

        # Check whether data number per type is over threshold 
        thres_num = len(bal_type_list) * args.min_cls_thres
        for type_num in sorted_dict.values():
            if type_num >= thres_num:
                thres_type = [k for k, v in sorted_dict.items() if v == type_num][0]
                thres_type_num = sorted_dict[thres_type]
                print(f"Set {thres_type}(# of data : {thres_type_num}) as # of data standard type for sampling")
                break  # break when get threshold class

        # Assign indices per type on balanced dataset
        ids_0 = []; ids_1 = []; ids_2 = []; ids_3 = []; ids_4 = []
        for i in range(len(bal_type_list)):
            if bal_type_list[i] == 0:  ## hypo
                ids_0.append(i)
            elif bal_type_list[i] == 1:  ## normal
                ids_1.append(i)
            elif bal_type_list[i] == 2:  ## prehyper
                ids_2.append(i)
            elif bal_type_list[i] == 3:  ## hyper2
                ids_3.append(i)
            elif bal_type_list[i] == 4:  ## crisis
                ids_4.append(i)
        ids_dict = {0: ids_0, 1: ids_1, 2: ids_2, 3: ids_3, 4: ids_4}
        print('# of Data per type before balancing')
        print(f"type 0 : {len(ids_0)}")
        print(f"type 1 : {len(ids_1)}")
        print(f"type 2 : {len(ids_2)}")
        print(f"type 3 : {len(ids_3)}")
        print(f"type 4 : {len(ids_4)}")
        print('-' * 30)

        ## Oversampling, Undersampling for balancing
        new_sbp_list, new_dbp_list, new_time_list, new_ppg_list, new_type_list = [], [], [], [], []
        for type_idx, type_num in sorted_dict.items():
            if type_num == 0:
                raise print("There must be problem to get balance train / val2 / bal / test dataset")
            # Assign indices of list according to type
            if type_num < thres_num:
                # Upsampling type of data less than threshold data count
                idx_list = np.random.choice(ids_dict[type_idx], thres_type_num, replace=True)
            else:
                # Downsampling type of data more than threshold data count
                idx_list = np.random.choice(ids_dict[type_idx], thres_type_num, replace=False)

            for idx in idx_list:
                new_sbp_list.append(bal_sbp_list[idx])
                new_dbp_list.append(bal_dbp_list[idx])
                new_time_list.append(bal_time_list[idx])
                new_ppg_list.append(bal_ppg_list[idx])
                new_type_list.append(bal_type_list[idx])

        print(f"Complete Up/DownSampling for Minority/Majority Classes to DFR")
        print('-' * 30)
        print("# od Data per type after balancing")
        print(f"type 0 : {new_type_list.count(0)}")
        print(f"type 1 : {new_type_list.count(1)}")
        print(f"type 2 : {new_type_list.count(2)}")
        print(f"type 3 : {new_type_list.count(3)}")
        print(f"type 4 : {new_type_list.count(4)}")
        
        sbp_list, dbp_list, time_list, ppg_list, type_list = new_sbp_list, new_dbp_list, new_time_list, new_ppg_list, new_type_list
        assert len(sbp_list) == len(dbp_list) == len(time_list) == len(ppg_list) == len(type_list) == (5 * thres_type_num)

        val2_dataset_loader = make_loader(args, val2_sbp_list, val2_dbp_list, val2_time_list, val2_ppg_list,
                                          val2_type_list, train=False)
        bal_dataset_loader = make_loader(args, sbp_list, dbp_list, time_list, ppg_list, type_list, train=True)
    else:
        # No need for validation for phase 2 and balanced dataset except DFR
        val2_dataset_loader = []
        bal_dataset_loader = []

    tr_dataset_loader = make_loader(args, tr_sbp_list, tr_dbp_list, tr_time_list, tr_ppg_list, tr_type_list, train=True)
    val_dataset_loader = make_loader(args, val_sbp_list, val_dbp_list, val_time_list, val_ppg_list, val_type_list, train=False)
    te_dataset_loader = make_loader(args, te_sbp_list, te_dbp_list, te_time_list, te_ppg_list, te_type_list, train=False)

    return tr_dataset_loader, val_dataset_loader, val2_dataset_loader, bal_dataset_loader, te_dataset_loader


def restore_sbp(args, value):
    return (200 - 80) * (value - args.min) / (args.max - args.min) + 80


def restore_dbp(args, value):
    return (120 - 40) * (value - args.min) / (args.max - args.min) + 40



def get_divs_per_group(args, dataset_loader):
    from torch.distributions import normal, kl_divergence

    type = dataset_loader.dataset.type_data 

    sbp = dataset_loader.dataset.sbp_data
    sbp = sbp + 1                               ## Scale from [-1, 1] to [0, 2]

    dbp = dataset_loader.dataset.dbp_data
    dbp = dbp + 1                               ## Scale from [-1, 1] to [0, 2]
    
    ### Tukey transformation
    if args.tukey:
        if args.beta != 0:
            sbp = torch.pow(sbp, args.beta)
            dbp = torch.pow(dbp, args.beta)
        elif args.beta == 0:
            sbp = torch.log(sbp)
            dbp = torch.log(dbp)
        else:
            sbp = (-1) * torch.pow(sbp, args.beta)
            dbp = (-1) * torch.pow(dbp, args.beta)

    ### Assign group indices
    group_idx = torch.arange(5).unsqueeze(1).float()
    group_map = (group_idx==type).float()

    ### Calculate mean and variance per group
    # Total
    sbp_mean = sbp.mean(0) ; dbp_mean = dbp.mean(0)
    sbp_var = torch.var(sbp) ; dbp_var = torch.var(dbp)
    sbp_tot_dist = normal.Normal(sbp_mean, sbp_var) ; dbp_tot_dist = normal.Normal(dbp_mean, dbp_var)

    sbp_div_list = [] ; dbp_div_list = []
    for i in range(len(group_idx)): # Hypo, Normal, Prehyper, hyper2, crisis
        idx = (group_map[i].nonzero(as_tuple=True)[0])
        group_sbp = sbp[idx]; group_dbp = dbp[idx]
        
        group_sbp_mean = group_sbp.mean(0)
        group_sbp_var = torch.var(group_sbp)
        sbp_group_dist = normal.Normal(group_sbp_mean, group_sbp_var)
        sbp_kld = (kl_divergence(sbp_group_dist, sbp_tot_dist) + kl_divergence(sbp_tot_dist, sbp_group_dist)) / 2
    
        sbp_div_list.append(sbp_kld)

        group_dbp_mean = group_dbp.mean(0)
        group_dbp_var = torch.var(group_dbp)
        dbp_group_dist = normal.Normal(group_dbp_mean, group_dbp_var)
        dbp_kld = (kl_divergence(dbp_group_dist, dbp_tot_dist) + kl_divergence(dbp_tot_dist, dbp_group_dist)) / 2
        dbp_div_list.append(dbp_kld)

    # Scaling
    sbp_div_list = [div/sum(sbp_div_list) for div in sbp_div_list]
    dbp_div_list = [div/sum(dbp_div_list) for div in dbp_div_list]

    div_list = {"sbp" : sbp_div_list, "dbp" : dbp_div_list}
    return div_list


def scaled_reverse_count_group(args, dataset_loader):
    # Count number of groups and make N_i / N list
    N_total = len(dataset_loader.dataset.type_data)
    count_group_list = []
    for i in range(5):
        n_group = sum(dataset_loader.dataset.type_data == i)
        count_group_list.append(N_total/n_group)
    return count_group_list


def train(args, model, dataset_loader, count_group_list=None, group_div_list=None):
    model.train()
    total_loss = 0.0
    total_n = 0; hyper2_n = 0 ;normal_n = 0; hypo_n = 0; crisis_n = 0; prehyper_n = 0
    group_idx = torch.arange(5).unsqueeze(1).float()


    # Tracking #samples per domain
    for idx, data in enumerate(dataset_loader):

        args.optimizer.zero_grad()
        sbp, dbp, time, ppg, type = data

        group_map = (group_idx == type).float()
        hypo_idx = (group_map[0].nonzero(as_tuple=True)[0])
        normal_idx = (group_map[1].nonzero(as_tuple=True)[0])
        prehyper_idx = (group_map[2].nonzero(as_tuple=True)[0])
        hyper2_idx = (group_map[3].nonzero(as_tuple=True)[0])
        crisis_idx = (group_map[4].nonzero(as_tuple=True)[0])
        if args.time_cutmix:
            ppg, sbp, dbp, time, type = group_time_cutmix_all(ppg, sbp, dbp, time, type, hypo_idx, normal_idx, prehyper_idx, hyper2_idx, crisis_idx)
            
            group_map = (group_idx == type).float()
            hypo_idx = (group_map[0].nonzero(as_tuple=True)[0])
            normal_idx = (group_map[1].nonzero(as_tuple=True)[0])
            prehyper_idx = (group_map[2].nonzero(as_tuple=True)[0])
            hyper2_idx = (group_map[3].nonzero(as_tuple=True)[0])
            crisis_idx = (group_map[4].nonzero(as_tuple=True)[0])
            
        ppg = ppg.to(args.device)
        output = model(ppg)

        batch_total_n = len(output)
        batch_hypo_n = len(hypo_idx)
        batch_normal_n = len(normal_idx)
        batch_prehyper_n = len(prehyper_idx)
        batch_hyper2_n = len(hyper2_idx)
        batch_crisis_n = len(crisis_idx)

        total_n += batch_total_n
        hypo_n += batch_hypo_n
        normal_n += batch_normal_n
        prehyper_n += batch_prehyper_n
        hyper2_n += batch_hyper2_n
        crisis_n += batch_crisis_n
        sbp_loss = args.criterion(output[:,0], sbp.to(args.device))
        dbp_loss = args.criterion(output[:,1], dbp.to(args.device))
        
        if args.method == 'erm' or args.method == 'pretrained_erm':
            batch_loss = sbp_loss + dbp_loss
            batch_loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            args.optimizer.step()
            total_loss += batch_loss.item()*batch_total_n

        elif args.method == 'dro':
            group_map = group_map.to(args.device)
            sbp_group_loss, group_count = compute_group_avg(sbp_loss, group_map)
            dbp_group_loss, group_count = compute_group_avg(dbp_loss, group_map)
            sbp_actual_loss, sbp_weights = compute_robust_loss(args, sbp_group_loss, args.sbp_adv_probs)
            dbp_actual_loss, dbp_weights = compute_robust_loss(args, dbp_group_loss, args.dbp_adv_probs)
            args.sbp_adv_probs = sbp_weights
            args.dbp_adv_probs = dbp_weights
            actual_loss = sbp_actual_loss + dbp_actual_loss
            actual_loss.backward()
            args.optimizer.step()
            total_loss += actual_loss

        elif args.method in ['vrex', 'crex', 'drex', 'cdrex']:
            group_map = group_map.to(args.device)
            # sbp
            sbp_group_losses, group_count = compute_group_avg(sbp_loss, group_map)
            sbp_group_avg_loss = sbp_group_losses.mean()
            
            # dbp
            dbp_group_losses, _ = compute_group_avg(dbp_loss, group_map)
            dbp_group_avg_loss = dbp_group_losses.mean()

            if args.epoch >= args.annealing_epoch:
                #### SBP
                sbp_group_loss_variance = 0.
                shifted_sbp_group_losses = []
                for i, sbp_group_loss in enumerate(sbp_group_losses):
                    if args.method =='vrex':
                        shifted_sbp_group_loss = sbp_group_loss
                    elif args.method =='crex': # Add regularization concerned with training data per group
                        shifted_sbp_group_loss = sbp_group_loss + args.C1*torch.sqrt(count_group_list[i])
                    elif args.method =='drex': # Add regularization concerned with training distribution per group
                        shifted_sbp_group_loss = sbp_group_loss + args.C21 * group_div_list['sbp'][i]
                    elif args.method =='cdrex':
                        shifted_sbp_group_loss = sbp_group_loss + args.C1*torch.sqrt(count_group_list[i]) + args.C21*group_div_list['sbp'][i]
                    shifted_sbp_group_losses.append(shifted_sbp_group_loss)
                    
                # shifted average
                shifted_sbp_group_avg_loss = torch.tensor(0., device=args.device)
                for j in shifted_sbp_group_losses:
                    shifted_sbp_group_avg_loss += j
                shifted_sbp_group_avg_loss /= len(shifted_sbp_group_losses)

                # variance
                for j in shifted_sbp_group_losses:
                    sbp_group_loss_variance += (j - shifted_sbp_group_avg_loss)**2
                sbp_group_loss_variance /= (len(shifted_sbp_group_losses)-1)


                #### DBP
                dbp_group_loss_variance = 0.
                shifted_dbp_group_losses = []
                for i, dbp_group_loss in enumerate(dbp_group_losses):
                    if args.method == 'vrex':
                        shifted_dbp_group_loss = dbp_group_loss
                    elif args.method == 'crex': # Add regularization concerned with training data per group                        
                        shifted_dbp_group_loss = dbp_group_loss + args.C1*torch.sqrt(count_group_list[i])
                    elif args.method == 'drex': # Add regularization concerned with training distribution per group
                        shifted_dbp_group_loss = dbp_group_loss + args.C22 * group_div_list['dbp'][i]
                    elif args.method == 'cdrex':
                        shifted_dbp_group_loss = dbp_group_loss + args.C1*torch.sqrt(count_group_list[i]) + args.C22*group_div_list['dbp'][i]
                    shifted_dbp_group_losses.append(shifted_dbp_group_loss)

                # shifted average
                shifted_dbp_group_avg_loss = torch.tensor(0., device=args.device)
                for j in shifted_dbp_group_losses:
                    shifted_dbp_group_avg_loss += j
                shifted_dbp_group_avg_loss /= len(shifted_dbp_group_losses)

                # variance
                for j in shifted_dbp_group_losses:
                    dbp_group_loss_variance += (j - shifted_dbp_group_avg_loss)**2
                dbp_group_loss_variance /= (len(shifted_dbp_group_losses)-1)


                actual_loss = sbp_group_avg_loss + dbp_group_avg_loss + args.sbp_beta * sbp_group_loss_variance + args.dbp_beta * dbp_group_loss_variance
                
                if args.sbp_beta + args.dbp_beta > 1:
                    actual_loss /= (args.sbp_beta + args.dbp_beta)

            else:
                actual_loss = (sbp_loss + dbp_loss).mean()

            actual_loss.backward()
            args.optimizer.step()
            total_loss += actual_loss
            
    #args.scheduler.step() #TODO 어느정도 성능 나오면 시도
    total_loss = total_loss / idx
    return total_loss, total_n, hypo_n, normal_n, prehyper_n, hyper2_n, crisis_n




def eval(args,model,dataset_loader):
    model.eval()
    total_sbp_loss = 0.0; total_dbp_loss = 0.0
    hypo_sbp_loss = 0.0; hypo_dbp_loss = 0.0
    normal_sbp_loss = 0.0; normal_dbp_loss = 0.0
    prehyper_sbp_loss = 0.0; prehyper_dbp_loss = 0.0
    hyper2_sbp_loss = 0.0; hyper2_dbp_loss = 0.0
    crisis_sbp_loss = 0.0; crisis_dbp_loss = 0.0
    total_n = 0; hypo_n = 0; normal_n = 0; prehyper_n = 0; hyper2_n = 0;  crisis_n = 0
    group_idx = torch.arange(5).unsqueeze(1).float()

    with torch.no_grad():
        for data in dataset_loader:
            sbp, dbp, time, ppg, type = data
            ppg = ppg.to(args.device)
            output = model(ppg)
            sbp, dbp, output_sbp, output_dbp = restore_sbp(args, sbp), restore_dbp(args, dbp), \
                                               restore_sbp(args, output[:,0]), restore_dbp(args, output[:,1])
            group_map = (group_idx == type).float()

            hypo_idx = (group_map[0].nonzero(as_tuple=True)[0])
            normal_idx = (group_map[1].nonzero(as_tuple=True)[0])
            prehyper_idx = (group_map[2].nonzero(as_tuple=True)[0])
            hyper2_idx = (group_map[3].nonzero(as_tuple=True)[0])
            crisis_idx = (group_map[4].nonzero(as_tuple=True)[0])

            batch_total_n = len(output_sbp)
            batch_hypo_n = len(hypo_idx)
            batch_normal_n = len(normal_idx)
            batch_prehyper_n = len(prehyper_idx)
            batch_hyper2_n = len(hyper2_idx)
            batch_crisis_n = len(crisis_idx)

            total_n += batch_total_n
            hypo_n += batch_hypo_n
            normal_n += batch_normal_n
            prehyper_n += batch_prehyper_n
            hyper2_n += batch_hyper2_n
            crisis_n += batch_crisis_n

            assert batch_total_n == batch_hyper2_n + batch_normal_n + batch_hypo_n + batch_crisis_n + batch_prehyper_n

            # all
            sbp_loss = args.criterion_eval(output_sbp, sbp.to(args.device)) * batch_total_n
            dbp_loss = args.criterion_eval(output_dbp, dbp.to(args.device)) * batch_total_n

            if sbp_loss == sbp_loss: # For NaN
                total_sbp_loss += sbp_loss.item()
                total_dbp_loss += dbp_loss.item()
            else:
                assert 1==2 # NaN exists

            # Hypo
            ## hypotension data가 batch에 없다면, hypo_idx = []이므로, output[[],0]과 sbp[[]]에 의해서 nan이 나올 수 있음
            sbp_loss = args.criterion_eval(output_sbp[hypo_idx], sbp[hypo_idx].to(args.device)) * batch_hypo_n
            dbp_loss = args.criterion_eval(output_dbp[hypo_idx], dbp[hypo_idx].to(args.device)) * batch_hypo_n
            if sbp_loss == sbp_loss: # For NaN
                hypo_sbp_loss += sbp_loss.item()
                hypo_dbp_loss += dbp_loss.item()

            # normal
            sbp_loss = args.criterion_eval(output_sbp[normal_idx], sbp[normal_idx].to(args.device)) * batch_normal_n
            dbp_loss = args.criterion_eval(output_dbp[normal_idx], dbp[normal_idx].to(args.device)) * batch_normal_n
            if sbp_loss == sbp_loss: # For NaN
                normal_sbp_loss += sbp_loss.item()
                normal_dbp_loss += dbp_loss.item()

            # prehyper
            sbp_loss = args.criterion_eval(output_sbp[prehyper_idx], sbp[prehyper_idx].to(args.device)) * batch_prehyper_n
            dbp_loss = args.criterion_eval(output_dbp[prehyper_idx], dbp[prehyper_idx].to(args.device)) * batch_prehyper_n
            if sbp_loss == sbp_loss: # For NaN
                prehyper_sbp_loss += sbp_loss.item()
                prehyper_dbp_loss += dbp_loss.item()

            # hyper2
            sbp_loss = args.criterion_eval(output_sbp[hyper2_idx], sbp[hyper2_idx].to(args.device)) * batch_hyper2_n
            dbp_loss = args.criterion_eval(output_dbp[hyper2_idx], dbp[hyper2_idx].to(args.device)) * batch_hyper2_n
            if sbp_loss == sbp_loss:  # For NaN
                hyper2_sbp_loss += sbp_loss.item()
                hyper2_dbp_loss += dbp_loss.item()
            # Crisis
            sbp_loss = args.criterion_eval(output_sbp[crisis_idx], sbp[crisis_idx].to(args.device)) * batch_crisis_n
            dbp_loss = args.criterion_eval(output_dbp[crisis_idx], dbp[crisis_idx].to(args.device)) * batch_crisis_n
            if sbp_loss == sbp_loss: # For NaN
                crisis_sbp_loss += sbp_loss.item()
                crisis_dbp_loss += dbp_loss.item()


    assert total_n == hyper2_n + normal_n + hypo_n + crisis_n + prehyper_n
    total_sbp_loss = total_sbp_loss / total_n
    total_dbp_loss = total_dbp_loss / total_n
    hypo_sbp_loss = hypo_sbp_loss / np.max([hypo_n,1])
    hypo_dbp_loss = hypo_dbp_loss / np.max([hypo_n,1])
    normal_sbp_loss = normal_sbp_loss / np.max([normal_n,1])
    normal_dbp_loss = normal_dbp_loss / np.max([normal_n,1])
    prehyper_sbp_loss = prehyper_sbp_loss / np.max([prehyper_n, 1])
    prehyper_dbp_loss = prehyper_dbp_loss / np.max([prehyper_n, 1])
    hyper2_sbp_loss = hyper2_sbp_loss / np.max([hyper2_n, 1])
    hyper2_dbp_loss = hyper2_dbp_loss / np.max([hyper2_n, 1])
    crisis_sbp_loss = crisis_sbp_loss / np.max([crisis_n,1])
    crisis_dbp_loss = crisis_dbp_loss / np.max([crisis_n,1])

    return total_sbp_loss, total_dbp_loss, total_n, \
           hypo_sbp_loss, hypo_dbp_loss, hypo_n, \
           normal_sbp_loss,normal_dbp_loss,normal_n, \
           prehyper_sbp_loss, prehyper_dbp_loss, prehyper_n, \
           hyper2_sbp_loss, hyper2_dbp_loss, hyper2_n, \
           crisis_sbp_loss, crisis_dbp_loss, crisis_n