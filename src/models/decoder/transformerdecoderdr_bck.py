from torch import nn
from torch.nn import functional as F
import torch
import segmentation_models_pytorch as smp

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
        x = 0.5*x + 0.5*res
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
        self.GRU = nn.GRU(d_model, d_model, num_layers=1, bidirectional=True)
        self.linear1 = nn.Linear(d_model * 2, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        res = x
        x, _ = self.GRU(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = 0.7*x + 0.3*res
        return x


class SAKTModelDr(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        embed_dim=128,
        nlayers=2,
        rnnlayers=3,
        dropout=0.1,
        nheads=8,
        feature_extractor=None,
        downsample_rate=4,
        extra_channels=0,
    ):
        super(SAKTModelDr, self).__init__()

        self.resize = nn.Linear(nin, embed_dim//2)
        self.fist_downsample = nn.Conv1d(embed_dim//2, embed_dim//2, downsample_rate, stride=downsample_rate)
        self.feature_extractor = feature_extractor
        self.pos_encoder_1 = nn.ModuleList([ResidualLSTM(embed_dim//2) for _ in range(rnnlayers)])
        self.pos_encoder_dropout_1 = nn.Dropout(dropout)
        self.unet_1 = smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=4,
            classes=1,
        )

        self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = [
            nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim * 4)
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
        nn.linear_layers = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(nlayers)])
        self.droputs = nn.ModuleList([nn.Dropout(dropout) for _ in range(nlayers)])
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        self.conv_layers = nn.ModuleList(conv_layers)
        self.layer_norm_layers = nn.ModuleList(layer_norm_layers)
        self.layer_norm_layers2 = nn.ModuleList(layer_norm_layers2)
        self.deconv_layers = nn.ModuleList(deconv_layers)
        self.nheads = nheads
        self.pred = nn.Linear(embed_dim, nout)
        self.downsample = nn.Linear(embed_dim * 2, embed_dim)


    def forward(self, x):
        if self.feature_extractor is not None:
            x_conv_unet = self.feature_extractor(x)
            x_conv_unet = torch.cat(x_conv_unet, dim=1)
        x = self.resize(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.fist_downsample(x)
        x = x.permute(2, 0, 1)
        for lstm in self.pos_encoder_1:
            lstm.LSTM.flatten_parameters()
            x = lstm(x)
        x_lstm = self.pos_encoder_dropout_1(x)
        if self.feature_extractor is not None:
            x_conv_unet = self.unet_1(x_conv_unet).squeeze(1).permute(2, 0, 1)
            x = torch.cat([x_lstm, x_conv_unet], dim=2)
        x = self.layer_normal(x)
        for conv, transformer_layer, layer_norm1, layer_norm2, linear, deconv, dropout in zip(
            self.conv_layers,
            self.transformer_encoder,
            self.layer_norm_layers,
            self.linear_layers,
            self.layer_norm_layers2,
            self.deconv_layers,
            self.droputs,
        ):
            res = x
            x = F.relu(conv(x.permute(1, 2, 0)).permute(2, 0, 1))
            x = layer_norm1(x)
            x = transformer_layer(x)
            x = F.relu(deconv(x.permute(1, 2, 0)).permute(2, 0, 1))
            x = F.relu(linear(x))
            x = dropout(x)
            x = layer_norm2(x)
            x = 0.7*x + 0.3*res

        x = x.permute(1, 0, 2)

        output = self.pred(x)

        return output.squeeze(-1)
