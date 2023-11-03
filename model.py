import torch
import torch.nn as nn
import math

## Transformer를 위한 P.E 정의
class PositionalEncoding(nn.Module):
    # https://github.com/ctxj/Time-Series-Transformer-Pytorch/blob/main/transformer_model.ipynb
    def __init__(self, args, max_len=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, args.d_input)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, args.d_input, 2).float() * (-math.log(10000.0) / args.d_input))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, x.size(1), :]


## Transformer Model 구현
class TransAm(nn.Module):
    def __init__(self, args):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.input_layer = nn.Linear(args.feature_size, args.d_input)
        self.output_layer = nn.Linear(args.d_input, args.d_output)
        self.pos_encoder = PositionalEncoding(args)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.d_input, nhead=args.num_heads, dim_feedforward=args.d_model, dropout=args.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.num_layers)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_layer.bias.data.zero_()
        self.input_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.input_layer(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        pooled_output, _ = torch.max(output,dim=1)
        pooled_output = torch.squeeze(pooled_output,1)
        final_output = self.output_layer(pooled_output)
        return final_output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


## Convolution layer + Transformer 구현
class ConvTransAm(nn.Module):
    def __init__(self, args):
        super(ConvTransAm, self).__init__()
        self.model_type = 'ConvTransformer'

        self.src_mask = None
        self.input_layer = nn.Linear(args.feature_size, args.d_input)  # [batch_size, len(ppg), d_input]

        # Convolution layers
        self.conv_3 = nn.Conv2d(in_channels=args.d_input, out_channels=args.num_filters, kernel_size=(args.d_input, 3), padding='same', device=args.device)
        self.conv_5 = nn.Conv2d(in_channels=args.d_input, out_channels=args.num_filters, kernel_size=(args.d_input, 5), padding='same', device=args.device)
        self.conv_7 = nn.Conv2d(in_channels=args.d_input, out_channels=args.num_filters, kernel_size=(args.d_input, 7), padding='same', device=args.device)
        self.conv_9 = nn.Conv2d(in_channels=args.d_input, out_channels=args.num_filters, kernel_size=(args.d_input, 9), padding='same', device=args.device)

        self.pos_embedding = torch.nn.Parameter(torch.randn((args.num_filters*4, 1000)))    # [num_filters*4, len(ppg)] # 실제로 사용되진 않음
        
        self.output_layer = nn.Linear(args.num_filters*4, args.d_output)
    
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.num_filters*4, nhead=args.num_heads, dim_feedforward=args.d_model, dropout=args.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.num_layers)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_layer.bias.data.zero_()
        self.input_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.input_layer(src)
        src = src.unsqueeze(3).permute(0,2,1,3)             # [batch_size, d_input, 1000, 1]
        src = torch.cat([self.conv_3(src),                  # [batch_size, num_filters*4, 1000, 1]
                    self.conv_5(src),
                    self.conv_7(src),
                    self.conv_9(src)],
                    1)                           
    
        src = torch.squeeze(src)                                 # [batch_size, num_filters*4, 1000]
        src = src.permute(0, 2, 1)

        output = self.transformer_encoder(src)
        pooled_output, _ = torch.max(output,dim=1)
        pooled_output = torch.squeeze(pooled_output,1)
        final_output = self.output_layer(pooled_output)
        return final_output