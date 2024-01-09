import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from .base_pl import Regressor
import coloredlogs, logging

####
from core.utils import *
####

coloredlogs.install()
logger = logging.getLogger(__name__)  

class MLPBP(Regressor):
    def __init__(self, param_model, config, random_state=0):
        super(MLPBP, self).__init__(param_model, random_state)
        self.config = config
        self.model = MLPMixer(param_model.in_channels, 
                               param_model.dim, 
                               param_model.num_classes, 
                               param_model.num_patch,
                               param_model.depth, 
                               param_model.token_dim, 
                               param_model.channel_dim, 
                               param_model.dropout)
        self.annealing = True
        
    def _shared_step(self, batch, mode):
        x_ppg, y, x_abp, peakmask, vlymask, group = batch
        ppg = x_ppg['ppg']
        if self.config.method == "cdrex_time" and mode == "train":
            ppg, y, group = group_time_cutmix_all(ppg, y, group)
        
        pred = self.model(ppg)
        losses = self.criterion(pred, y)
        group = group.unsqueeze(1)
        return losses, pred, x_abp, y, group

    def training_step(self, batch, batch_idx):
        mode = "train"
        losses, pred_bp, t_abp, label, group = self._shared_step(batch, mode)
        if self.config.method != "erm" and not self.annealing:
            per_group, group_count = per_group_loss(losses, group) #[2x5] [sbp/dbp, BP_group]
            mask = (group_count != 0) # To avoid 0 bp_group
            per_group_avg = per_group.sum(1)/(mask.sum())

            if self.config.method in ["crex", "cdrex", "cdrex_time"]:
                reversed = torch.tensor(self.config.hijack['reversed_total_group_count']).to(losses.device)
                per_group += self.config.C1*torch.sqrt(reversed.unsqueeze(0))
                
            if self.config.method in ["drex", "cdrex", "cdrex_time"]:
                coeff_tensor = torch.tensor([self.config.C21, self.config.C22]).unsqueeze(1)
                div_list = torch.tensor([self.config.hijack["div_list"]["sbp"],
                                        self.config.hijack["div_list"]["dbp"]])
                per_group += (coeff_tensor*div_list).to(per_group.device)
            
            variance = torch.var(per_group[:, mask], dim=1) # [2,]
            loss = per_group_avg.sum() + self.config.sbp_beta * variance[0] + self.config.dbp_beta * variance[1]
            if self.config.sbp_beta + self.config.dbp_beta > 1:
                loss /= (self.config.sbp_beta + self.config.dbp_beta)
        else:
            loss = losses.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group, "losses": losses}
    
    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_bp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_bp"] for v in train_step_outputs], dim=0)
        if self.config.group_avg:
            group = torch.cat([v["group"].squeeze(1) for v in train_step_outputs], dim=0)
            metrics = self._cal_metric(logit.detach(), label.detach(), group)
        else:
            metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        mode = "val"
        losses, pred_bp, t_abp, label, group = self._shared_step(batch, mode)
        loss = losses.mean()
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group, "losses": losses}

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in val_step_end_out], dim=0)
        if self.config.group_avg:
            group = torch.cat([v["group"].squeeze(1) for v in val_step_end_out], dim=0)
            metrics = self._cal_metric(logit.detach(), label.detach(), group)
        else:
            metrics = self._cal_metric(logit.detach(), label.detach())
        
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        mode = "test"
        losses, pred_bp, t_abp, label, group = self._shared_step(batch,mode)
        loss = losses.mean()
        self.log('test_loss', loss, prog_bar=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group, "losses": losses}

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in test_step_end_out], dim=0)
        if self.config.group_avg:
            group = torch.cat([v["group"].squeeze(1) for v in test_step_end_out], dim=0)
            metrics = self._cal_metric(logit.detach(), label.detach(), group)
        else:
            metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="test")
        return test_step_end_out

    def grouping(self, losses, group):
        group_type = torch.arange(0,5).cuda()
        group_map = (group_type.view(-1,1)==group).float()
        group_count = group_map.sum(1)
        group_loss_map = losses.squeeze(0) * group_map.unsqueeze(2) # (4,bs,2)
        group_loss = group_loss_map.sum(1)                          # (4,2)
        
        # Average only across the existing group
        mask = group_count != 0
        avg_per_group = torch.zeros_like(group_loss)
        avg_per_group[mask, :] = group_loss[mask, :] / group_count[mask].unsqueeze(1)
        exist_group = mask.sum()
        avg_group = avg_per_group.sum(0)/exist_group
        loss = avg_group.sum()/2
        return loss

    def _cal_metric(self, logit: torch.tensor, label: torch.tensor, group=None):
        prev_mse = (logit-label)**2
        prev_mae = torch.abs(logit-label)
        prev_me = logit-label
        mse = torch.mean(prev_mse)
        mae = torch.mean(prev_mae)
        me = torch.mean(prev_me)
        std = torch.std(torch.mean(logit-label, dim=1))
        if self.config.group_avg:
            group_mse = self.grouping(prev_mse, group)
            group_mae = self.grouping(prev_mae, group)
            group_me = self.grouping(prev_me, group)
            return {"mse":mse, "mae":mae, "std": std, "me": me, "group_mse":group_mse, "group_mae":group_mae, "group_me":group_me} 
        else:
            return {"mse":mse, "mae":mae, "std": std, "me": me} 


#%%
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)

        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, num_patch,
                 depth, token_dim, channel_dim, dropout=0.2):
        super().__init__()

        self.num_patch = num_patch
        self.to_patch_embedding = nn.Sequential(
            nn.Conv1d(in_channels, dim, kernel_size=1, stride=1),
            Rearrange('b c t -> b t c'),
        )

        # todo: LSTM emb
        self.lstm_patch_emb = nn.Sequential(
            Rearrange('b c t -> b t c'),
            nn.LSTM(input_size=in_channels, hidden_size=int(0.5*dim), num_layers=1,
                                      bidirectional=True, batch_first=True),
        )

        self.mixer_blocks = nn.ModuleList()
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.conv1d_decode = nn.Sequential(
            nn.Conv1d(num_patch, 2*num_patch, kernel_size=6, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv1d(2*num_patch, 4*num_patch, kernel_size=6, stride=4, padding=1),
            nn.ReLU(),
            # nn.Conv1d(512, 512, kernel_size=6, stride=4, padding=1)
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(4*num_patch, 128),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x = self.to_patch_embedding(x)
        x, (hn, cn) = self.lstm_patch_emb(x)
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_norm(x)
        x = self.conv1d_decode(x)
        x = x.mean(dim=2)
        return self.mlp_head(x)