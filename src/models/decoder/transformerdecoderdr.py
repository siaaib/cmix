from torch import nn
from torch.nn import functional as F
import torch
import segmentation_models_pytorch as smp

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
        extra_channels=0,
        downsample_rate=2,
    ):
        super(SAKTModelDr, self).__init__()

        self.fist_downsample = nn.Conv1d(embed_dim, embed_dim, downsample_rate, stride=downsample_rate)
        self.resize = nn.Linear(nin, embed_dim)
        self.feature_extractor = feature_extractor
        self.pos_encoder_1 = nn.GRU(embed_dim, embed_dim//2, num_layers=rnnlayers, bidirectional=True)
        self.pos_encoder_dropout_1 = nn.Dropout(dropout)
        self.unet_dropout = nn.Dropout(dropout)
        self.unet_1 = smp.Unet(
            encoder_name='mit_b0',
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
        )

        self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = [
            nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim * 4, dropout/2)
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
        if self.feature_extractor is not None:
            x_conv_unet = self.feature_extractor(x)
            #print(x_conv_unet[0].shape)
            x_conv_unet = torch.cat(x_conv_unet, dim=1)
            #print(x_conv_unet.shape)
        x = self.resize(x.permute(0, 2, 1)).permute(0, 2, 1)
        #print(x.shape)
        x = self.fist_downsample(x)
        #print(x.shape)
        x = x.permute(2, 0, 1)
        x_lstm, _ = self.pos_encoder_1(x)
        #print(f"lstm: {x_lstm.shape}")
        #x_lstm = self.pos_encoder_dropout_1(x)
        #print(f"lstm drop: {x_lstm.shape}")
        if self.feature_extractor is not None:
            x_conv_unet = self.unet_1(x_conv_unet).squeeze(1).permute(2, 0, 1)
            #print(f"unet out {x_conv_unet.shape}")
            #x_conv_unet = self.unet_dropout(x_conv_unet)
            #print(x_conv_unet.shape)
            #x = torch.cat([x_lstm, x_conv_unet], dim=2)
            x = x_lstm + x_conv_unet
            #print(x.shape)
        x = self.layer_normal(x)
        #print(x.shape)
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
            x = 0.5*x + 0.5*res

        x = x.permute(1, 0, 2)

        output = self.pred(x)

        return output.squeeze(-1)
