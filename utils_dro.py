import torch
"""
from https://github.com/kohpangwei/group_DRO
"""
def compute_group_avg(losses, group_map):
    '''
    batch의 데이터별 정보가 들어오면 group별 정보로 변환해줍니다.
    '''
    # compute observed counts and mean loss for each group
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count == 0).float()  # avoid nans
    group_loss = (group_map @ losses.view(-1)) / group_denom  ### avg Loss per group
    return group_loss, group_count

def compute_robust_loss(args, group_loss, adv_probs):
    '''
    GroupDRO의 Loss Term을 형성해줍니다.
    현재 step의 Group별 Loss를 반영한 가중치를 group별 loss에 적용합니다.
    '''
    adjusted_loss = group_loss
    adv_probs = adv_probs * torch.exp(args.robust_step_size*adjusted_loss.data)
    adv_probs = adv_probs/(adv_probs.sum())
    robust_loss = group_loss @ adv_probs
    return robust_loss, adv_probs