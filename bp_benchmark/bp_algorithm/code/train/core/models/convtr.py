import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_pl import Regressor
from .resnet import MyConv1dPadSame, MyMaxPool1dPadSame, BasicBlock
import coloredlogs, logging
import pdb

####
from core.utils import *
####

coloredlogs.install()
logger = logging.getLogger(__name__)  
"""
    Pararmetes:
        feature_size: dim of input feature
        d_input: dim of embedding for each feature
        num_filters: num of filters in conv (Use common param in conv3,5,7,9)
        num_heads: num of heads for Transformer Encoder
        d_model: dim of feadforward network in Transformer Encoder
        dropout: the rate of dropout in Transformer Encoder
        batch_first: set the order of dim of input in Trasformer Encoder
        num_layer: num of encoder layer in Transformer
        d_output: dim of final output (For PPG, 2 [SBP, DBP])      
"""

class ConvTransformer(Regressor):
    def __init__(self, param_model, config, random_state=0):
        super(ConvTransformer, self).__init__(param_model, random_state)
        self.config = config
        self.model = ConvTransform(param_model.feature_size, param_model.d_input,
                                param_model.num_filters, param_model.num_heads,
                                param_model.d_model, param_model.dropout,
                                param_model.num_layer, param_model.d_output, param_model.batch_first)
                                
    def _shared_step(self, batch):
        x_ppg, y, x_abp, peakmask, vlymask, group = batch
        #x_ppg, y, x_abp, peakmask, vlymask = batch
        pred = self.model(x_ppg['ppg'])
        losses = self.criterion(pred, y)
        group = group.unsqueeze(1)
        return losses, pred, x_abp, y, group

    def training_step(self, batch, batch_idx):
        losses, pred_bp, t_abp, label, group = self._shared_step(batch)
        if self.config.method != "erm":
            per_group, group_count = per_group_loss(losses, group) #[2x5] [sbp/dbp, BP_group]
            mask = (group_count != 0) # To avoid 0 bp_group
            per_group_avg = per_group.sum(1)/(mask.sum())

            if self.config.method in ["crex", "cdrex", "cdrex_time"]:
                pass
            if self.config.method in ["drex", "cdrex", "cdrex_time"]:
                pass
            
            variance = torch.var(per_group[:, mask], dim=1) # [2,]
            loss = per_group_avg.sum() + self.config.sbp_beta * variance[0] + self.config.dbp_beta * variance[1]
            if self.config.sbp_beta + self.config.dbp_beta > 1:
                loss /= (self.config.sbp_beta + self.config.dbp_beta)

        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group, "losses": losses}
    
    def training_epoch_end(self, train_step_outputs):
        logit = torch.cat([v["pred_bp"] for v in train_step_outputs], dim=0)
        label = torch.cat([v["true_bp"] for v in train_step_outputs], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="train")

    def validation_step(self, batch, batch_idx):
        losses, pred_bp, t_abp, label, group = self._shared_step(batch)
        loss = losses.mean()
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group, "losses": losses}

    def validation_epoch_end(self, val_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in val_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in val_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="val")
        return val_step_end_out

    def test_step(self, batch, batch_idx):
        losses, pred_bp, t_abp, label, group = self._shared_step(batch)
        loss = losses.mean()
        self.log('test_loss', loss, prog_bar=True)
        return {"loss":loss, "pred_bp":pred_bp, "true_abp":t_abp, "true_bp":label, "group": group, "losses": losses}

    def test_epoch_end(self, test_step_end_out):
        logit = torch.cat([v["pred_bp"] for v in test_step_end_out], dim=0)
        label = torch.cat([v["true_bp"] for v in test_step_end_out], dim=0)
        metrics = self._cal_metric(logit.detach(), label.detach())
        self._log_metric(metrics, mode="test")
        return test_step_end_out


class ConvTransform(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        feature_size: dim of input feature
        d_input: dim of embedding for each feature
        num_filters: num of filters in conv (Use common param in conv3,5,7,9)
        num_heads: num of heads for Transformer Encoder
        d_model: dim of feadforward network in Transformer Encoder
        dropout: the rate of dropout in Transformer Encoder
        batch_first: set the order of dim of input in Trasformer Encoder
        num_layer: num of encoder layer in Transformer
        d_output: dim of final output (For PPG, 2 [SBP, DBP])      
    """

    def __init__(self, feature_size, d_input, num_filters, num_heads, d_model, dropout, num_layer, d_output, batch_first=True):
        super(ConvTransform, self).__init__()
        self.src_mask = None
        self.input_layer = nn.Linear(feature_size, d_input)  # [batch_size, len(ppg), d_input]
        
        # Convolution layers
        self.conv_3 = nn.Conv2d(in_channels=d_input, out_channels=num_filters, kernel_size=(d_input, 3), padding='same')
        self.conv_5 = nn.Conv2d(in_channels=d_input, out_channels=num_filters, kernel_size=(d_input, 5), padding='same')
        self.conv_7 = nn.Conv2d(in_channels=d_input, out_channels=num_filters, kernel_size=(d_input, 7), padding='same')
        self.conv_9 = nn.Conv2d(in_channels=d_input, out_channels=num_filters, kernel_size=(d_input, 9), padding='same')

        #self.pos_embedding = torch.nn.Parameter(torch.randn((num_filters*4, 1000)))    # [num_filters*4, len(ppg)] 
        
        self.output_layer = nn.Linear(num_filters*4, d_output)
    
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=num_filters*4, nhead=num_heads, dim_feedforward=d_model, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layer)
        self.init_weights()

    def forward(self, src):
        src = src.permute(0,2,1)                            # [batch_size, length of a signal, feature_size=1]
        src = self.input_layer(src)                         # feature_size=1 --> d_input
        src = src.unsqueeze(3).permute(0,2,1,3)             # [batch_size, d_input, 1000, 1]
        src = torch.cat([self.conv_3(src),                  # [batch_size, num_filters*4, 1000, 1]
                    self.conv_5(src),
                    self.conv_7(src),
                    self.conv_9(src)],
                    1)                           
        src = src.squeeze(3)                                 # [batch_size, num_filters*4, 1000]
        src = src.permute(0, 2, 1)

        output = self.transformer_encoder(src)
        pooled_output, _ = torch.max(output,dim=1)
        pooled_output = torch.squeeze(pooled_output,1)
        final_output = self.output_layer(pooled_output)
        return final_output
    

    def init_weights(self):
        initrange = 0.1
        self.input_layer.bias.data.zero_()
        self.input_layer.weight.data.uniform_(-initrange, initrange)