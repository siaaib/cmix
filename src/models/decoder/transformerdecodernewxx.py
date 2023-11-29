from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualLSTM(nn.Module):
    def __init__(self, d_model):
        super(ResidualLSTM, self).__init__()
        self.LSTM = nn.LSTM(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1 = nn.Linear(d_model * 2, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        res = x
        x, _ = self.LSTM(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = res + x
        return x

class NonResidualLSTM(nn.Module):
    def __init__(self, d_model):
        super(NonResidualLSTM, self).__init__()
        self.LSTM = nn.LSTM(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1 = nn.Linear(d_model * 2, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        x, _ = self.LSTM(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class ResidualGRU(nn.Module):
    def __init__(self, d_model):
        super(ResidualGRU, self).__init__()
        self.LSTM = nn.GRU(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1 = nn.Linear(d_model * 2, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        res = x
        x, _ = self.LSTM(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = res + x
        return x


class SAKTModel(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        embed_dim=128,
        pos_encode="LSTM",
        nlayers=2,
        rnnlayers=3,
        dropout=0.1,
        nheads=8,
    ):
        super(SAKTModel, self).__init__()
        self.nin = nin
        self.embed_dim = embed_dim
        self.resize = nn.Linear(nin, embed_dim)
        if pos_encode == "LSTM":
            self.pos_encoder = nn.ModuleList([ResidualLSTM(embed_dim) for _ in range(rnnlayers)])
        elif pos_encode == "GRU":
            self.pos_encoder = nn.ModuleList([ResidualGRU(embed_dim) for _ in range(rnnlayers)])
        self.pos_encoder_dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = [
            nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim * 4, dropout)
            for _ in range(nlayers)
        ]
        conv_layers = [
            nn.Conv1d(embed_dim, embed_dim, (nlayers - i) * 2 - 1, stride=1, padding=0)
            for i in range(nlayers)
        ]
        deconv_layers = [
            nn.ConvTranspose1d(embed_dim, embed_dim, (nlayers - i) * 2 - 1, stride=1, padding=0)
            for i in range(nlayers)
        ]
        layer_norm_layers = [nn.LayerNorm(embed_dim) for _ in range(nlayers)]
        layer_norm_layers2 = [nn.LayerNorm(embed_dim) for _ in range(nlayers)]
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norm_layers = nn.ModuleList(layer_norm_layers)
        self.layer_norm_layers2 = nn.ModuleList(layer_norm_layers2)
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.nheads = nheads
        self.pred = nn.Linear(embed_dim, nout)
        self.downsample = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x):
        #x = self.resize(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x.permute(2, 0, 1)
        for lstm in self.pos_encoder:
            lstm.LSTM.flatten_parameters()
            x = lstm(x)
        x = self.pos_encoder_dropout(x)
        x = self.layer_normal(x)

        for conv, transformer_layer, layer_norm1, layer_norm2, deconv in zip(
            self.conv_layers,
            self.transformer_encoder,
            self.layer_norm_layers,
            self.layer_norm_layers2,
            self.deconv_layers,
        ):
            res = x
            x = F.relu(conv(x.permute(1, 2, 0)).permute(2, 0, 1))
            x = layer_norm1(x)
            x = transformer_layer(x)
            x = F.relu(deconv(x.permute(1, 2, 0)).permute(2, 0, 1))
            x = layer_norm2(x)
            x = 0.3*res + 0.7*x

        x = x.permute(1, 0, 2)

        output = self.pred(x)

        return output.squeeze(-1)
