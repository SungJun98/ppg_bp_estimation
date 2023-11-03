import numpy as np
import torch
import random
import copy

# train에서 ppg가 model에 들어가기전 ppg를 mix

# Mix by group

def group_time_cutmix_all(ppg, sbp, dbp, time, group, idx_0, idx_1, idx_2, idx_3, idx_4):
    
    time_s = torch.zeros_like(time)
    group_s = torch.zeros_like(group)
    sbp_s = torch.zeros_like(sbp)
    dbp_s = torch.zeros_like(dbp)
    ppg_s = torch.zeros_like(ppg)
    lamb = torch.rand(len(ppg))
    lamb_f = torch.round(1000*lamb).int()

    mask_a = []
    for i in lamb_f :
        mask_a.append((i >= torch.range(1,1000)).float())
    mask_a = torch.stack(mask_a).view(-1,1000,1)
    mask_b = 1-mask_a

    len_0 = len(idx_0)
    len_1 = len(idx_1)
    len_2 = len(idx_2)
    len_3 = len(idx_3)
    len_4 = len(idx_4)

    ppg_s, sbp_s, dbp_s, time_s, group_s = one_group(
                                                    ppg, sbp, dbp, time, group, ppg_s, sbp_s,
                                                    dbp_s, time_s, group_s, idx_0, lamb, 0,
                                                    len_0, mask_a, mask_b
                                                    )
    ppg_s, sbp_s, dbp_s, time_s, group_s = one_group(
                                                    ppg, sbp, dbp, time, group,
                                                    ppg_s, sbp_s, dbp_s, time_s, 
                                                    group_s, idx_1, lamb, len_0,len_0+len_1,mask_a, mask_b
                                                     )
    ppg_s, sbp_s, dbp_s, time_s, group_s = one_group(
                                                    ppg, sbp, dbp, time, group,
                                                    ppg_s, sbp_s, dbp_s, time_s, group_s,
                                                    idx_2, lamb, len_0+len_1, 
                                                    len_0+len_1+len_2, mask_a, mask_b
                                                    )
    ppg_s, sbp_s, dbp_s, time_s, group_s = one_group(
                                                    ppg, sbp, dbp, time, group, 
                                                    ppg_s, sbp_s, dbp_s, time_s, group_s, 
                                                    idx_3, lamb, len_0+len_1+len_2,
                                                    len_0+len_1+len_2+len_3, mask_a, mask_b)
    ppg_s, sbp_s, dbp_s, time_s, group_s = one_group(
                                                    ppg, sbp, dbp, time, group,
                                                    ppg_s, sbp_s, dbp_s, time_s, group_s, 
                                                    idx_4, lamb, len_0+len_1+len_2+len_3, 
                                                    len_0+len_1+len_2+len_3+len_4, mask_a, mask_b
                                                    )


    mixed_ppg = torch.cat((ppg, ppg_s),dim=0)
    mixed_sbp = torch.cat((sbp, sbp_s),dim=0)
    mixed_dbp = torch.cat((dbp, dbp_s),dim=0)
    mixed_time = torch.cat((time, time_s),dim=0)
    mixed_type = torch.cat((group, group_s),dim=0)
        
    return mixed_ppg, mixed_sbp, mixed_dbp, mixed_time, mixed_type


def one_group(ppg, sbp, dbp, time, group, ppg_s, sbp_s, dbp_s, time_s, group_s, idx, lamb,a, b, mask_a, mask_b):

    idx_ = torch.randperm(idx.size()[0])
    target_a = ppg[idx] 
    target_b = ppg[idx_] 
    sbp_mixed = lamb[a:b]*sbp[idx] + (1-lamb[a:b])*sbp[idx_]
    dbp_mixed = lamb[a:b]*dbp[idx] + (1-lamb[a:b])*dbp[idx_]
    time_s[a:b] = time[idx] 
    group_s[a:b]  = group[idx]
    sbp_s[a:b] = sbp_mixed
    dbp_s[a:b] = dbp_mixed 
    ppg_s[a:b] =  target_a*mask_a[a:b] + target_b*mask_b[a:b]   

    return ppg_s, sbp_s, dbp_s, time_s, group_s