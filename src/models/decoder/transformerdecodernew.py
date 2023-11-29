import torch.nn as nn
import torch.nn.functional as F






class SAKTModel(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        embed_dim=128,
        nlayers=2,
        dropout=0.1,
        nheads=8,
    ):
        super(SAKTModel, self).__init__()
        self.nin = nin
        self.embed_dim = embed_dim
        self.layer_normal = nn.LayerNorm(embed_dim)
        encoder_layers = [
            nn.TransformerEncoderLayer(embed_dim, nheads, embed_dim * 4, dropout)
            for _ in range(nlayers)
        ]
        conv_layers = [
            nn.Conv1d(embed_dim, embed_dim, (nlayers - i) * 2 - 1, stride=1, padding=0, dilation=1)
            for i in range(nlayers)
        ]
        deconv_layers = [
            nn.ConvTranspose1d(embed_dim, embed_dim, (nlayers - i) * 2 - 1, stride=1, padding=0, dilation=1)
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
        x = x.permute(2, 0, 1)
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
            x = 0.5*res + 0.5*x

        x = x.permute(1, 0, 2)

        output = self.pred(x)

        return output.squeeze(-1)
