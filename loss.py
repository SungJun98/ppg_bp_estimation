'''
Evaluation Criterion으로 RMSE을 사용하기 위한 class 정의
'''
import numpy as np
import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-12

    def forward(self, target, pred):
        return torch.sqrt(self.mse(target, pred) + self.eps)