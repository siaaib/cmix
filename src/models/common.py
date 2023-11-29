from typing import Union

import torch.nn as nn

from src.conf import DecoderConfig, FeatureExtractorConfig, InferenceConfig, TrainConfig
from src.models.base import BaseModel
from src.models.centernet import CenterNet
from src.models.decoder.lstmdecoder import LSTMDecoder
from src.models.decoder.mlpdecoder import MLPDecoder
from src.models.decoder.transformercnndecoder import TransformerCNNDecoder
from src.models.decoder.transformerdecoder import TransformerDecoder
from src.models.decoder.unet1ddecoder import UNet1DDecoder
from src.models.decoder.unet1ddecoderlstm import UNet1DDecoderLSTM
from src.models.detr2D import DETR2DCNN
from src.models.feature_extractor.cnn import CNNSpectrogram
from src.models.feature_extractor.lstm import LSTMFeatureExtractor
from src.models.feature_extractor.panns import PANNsFeatureExtractor
from src.models.feature_extractor.spectrogram import SpecFeatureExtractor
from src.models.feature_extractor.cnn_lstm import WAVELSTMFeatureExtractor
from src.models.spec1D import Spec1D
from src.models.spec2Dcnn import Spec2DCNN
from src.models.transformerautomodel import TransformerAutoModel
from src.models.feature_extractor.cnn_wave import WaveNetSpectrogram
from src.models.feature_extractor.cnn_wave_multi import WaveNetSpectrogramMulti
from src.models.specTr import SpecTr
from src.models.specTrDr import SpecTrDr
from src.models.decoder.transformerdecodernew import SAKTModel
from src.models.decoder.transformerdecoderdr import SAKTModelDr

from src.models.decoder.unet1ddecoderlstm import UNet1DDecoderLSTM


FEATURE_EXTRACTOR_TYPE = Union[
    CNNSpectrogram, PANNsFeatureExtractor, LSTMFeatureExtractor, SpecFeatureExtractor, WaveNetSpectrogram, WAVELSTMFeatureExtractor, WaveNetSpectrogramMulti
]
DECODER_TYPE = Union[
    UNet1DDecoder, LSTMDecoder, TransformerDecoder, MLPDecoder, TransformerCNNDecoder, SAKTModel, UNet1DDecoderLSTM, SAKTModelDr
]


def get_feature_extractor(
    cfg: FeatureExtractorConfig, feature_dim: int, num_timesteps: int, **kwargs
) -> FEATURE_EXTRACTOR_TYPE:
    feature_extractor: FEATURE_EXTRACTOR_TYPE
    print(cfg.name)
    if cfg.name == "CNNSpectrogram":
        feature_extractor = CNNSpectrogram(
            in_channels=feature_dim, output_size=num_timesteps, **cfg.params
        )

    elif cfg.name == "WaveNetSpectrogram":
        for k, v in kwargs.items():
            cfg.params[k] = v
        feature_extractor = WaveNetSpectrogram(
            in_channels=feature_dim,
            output_size=num_timesteps,
            **cfg.params
        )
    elif cfg.name == "WaveNetSpectrogramMulti":
        for k, v in kwargs.items():
            cfg.params[k] = v
        feature_extractor = WaveNetSpectrogramMulti(
            in_channels=feature_dim,
            output_size=num_timesteps,
            **cfg.params
        )
    elif cfg.name == "PANNsFeatureExtractor":
        feature_extractor = PANNsFeatureExtractor(
            in_channels=feature_dim, output_size=num_timesteps, conv=nn.Conv1d, **cfg.params
        )
    elif cfg.name == "LSTMFeatureExtractor":
        feature_extractor = LSTMFeatureExtractor(
            in_channels=feature_dim, out_size=num_timesteps, **cfg.params
        )
    elif cfg.name == "SpecFeatureExtractor":
        feature_extractor = SpecFeatureExtractor(
            in_channels=feature_dim, out_size=num_timesteps, **cfg.params
        )
    elif cfg.name == "WAVELSTMFeatureExtractor":
        feature_extractor = WAVELSTMFeatureExtractor(
            in_channels=feature_dim, out_size=num_timesteps, **cfg.params
        )
    else:
        raise ValueError(f"Invalid feature extractor name: {cfg.name}")

    return feature_extractor


def get_decoder(
    cfg: DecoderConfig, n_channels: int, n_classes: int, num_timesteps: int, **kwargs
) -> DECODER_TYPE:
    decoder: DECODER_TYPE
    if cfg.name == "UNet1DDecoder":
        decoder = UNet1DDecoder(
            n_channels=n_channels,
            n_classes=n_classes,
            duration=num_timesteps,
            **cfg.params,
        )
    elif cfg.name == "UNet1DDecoderLSTM":
        decoder = UNet1DDecoderLSTM(
            n_channels=n_channels,
            n_classes=n_classes,
            duration=num_timesteps,
            **cfg.params,
        )
    elif cfg.name == "LSTMDecoder":
        decoder = LSTMDecoder(
            input_size=n_channels,
            n_classes=n_classes,
            **cfg.params,
        )
    elif cfg.name == "TransformerDecoder":
        decoder = TransformerDecoder(
            input_size=n_channels,
            n_classes=n_classes,
            **cfg.params,
        )
    elif cfg.name == "MLPDecoder":
        decoder = MLPDecoder(
            n_channels=n_channels, 
            n_classes=n_classes,
            **cfg.params,
    )
    elif cfg.name == "TransformerCNNDecoder":
        decoder = TransformerCNNDecoder(
            input_size=n_channels,
            n_classes=n_classes,
            **cfg.params,
        )
    elif cfg.name == "SAKTModel":
        decoder = SAKTModel(nin=64, nout=n_classes, **cfg.params)
    elif cfg.name == "SAKTModelDr":
        decoder = SAKTModelDr(nin=19, nout=n_classes, feature_extractor=kwargs['feature_extractor'], extra_channels=kwargs['extra_channels'], **cfg.params)
    elif cfg.name == "UNet1DDecoderLSTM":
        decoder = UNet1DDecoderLSTM(
            n_channels=n_channels,
            n_classes=n_classes,
            duration=num_timesteps,
            **cfg.params,
        )
    else:
        raise ValueError(f"Invalid decoder name: {cfg.name}")

    return decoder


def get_model(
    cfg: TrainConfig | InferenceConfig,
    feature_dim: int,
    n_classes: int,
    num_timesteps: int,
    test: bool = False,
) -> BaseModel:
    model: BaseModel
    if cfg.model.name == "Spec2DCNN":
        feature_extractor = get_feature_extractor(
            cfg.feature_extractor, feature_dim, num_timesteps
        )
        decoder = get_decoder(cfg.decoder, feature_extractor.height, n_classes, num_timesteps)
        if test:
            cfg.model.params["encoder_weights"] = None
        model = Spec2DCNN(
            feature_extractor=feature_extractor,
            decoder=decoder,
            in_channels=feature_extractor.out_chans,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
            **cfg.model.params,
        )
    elif cfg.model.name == "Spec1D":
        feature_extractor = get_feature_extractor(
            cfg.feature_extractor, feature_dim, num_timesteps
        )
        n_channels = feature_extractor.out_chans * feature_extractor.height
        decoder = get_decoder(cfg.decoder, n_channels, n_classes, num_timesteps)
        model = Spec1D(
            feature_extractor=feature_extractor,
            decoder=decoder,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
        )
    elif cfg.model.name == "DETR2DCNN":
        feature_extractor = get_feature_extractor(
            cfg.feature_extractor, feature_dim, num_timesteps
        )
        decoder = get_decoder(
            cfg.decoder, feature_extractor.height, cfg.model.params["hidden_dim"], num_timesteps
        )
        if test:
            cfg.model.params["encoder_weights"] = None
        model = DETR2DCNN(
            feature_extractor=feature_extractor,
            decoder=decoder,
            in_channels=feature_extractor.out_chans,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
            **cfg.model.params,
        )
    elif cfg.model.name == "CenterNet":
        feature_extractor = get_feature_extractor(
            cfg.feature_extractor, feature_dim, num_timesteps
        )
        decoder = get_decoder(cfg.decoder, feature_extractor.height, 6, num_timesteps)
        if test:
            cfg.model.params["encoder_weights"] = None
        model = CenterNet(
            feature_extractor=feature_extractor,
            decoder=decoder,
            in_channels=feature_extractor.out_chans,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
            **cfg.model.params,
        )
    elif cfg.model.name == "TransformerAutoModel":
        model = TransformerAutoModel(
            n_channels=feature_dim,
            n_classes=n_classes,
            out_size=num_timesteps,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
            **cfg.model.params,
        )
    elif cfg.model.name == "SpecTr":
        feature_extractor = get_feature_extractor(cfg.feature_extractor, feature_dim, num_timesteps)
        decoder = get_decoder(cfg.decoder, feature_extractor.height, n_classes, num_timesteps)
        model = SpecTr(
            feature_extractor=feature_extractor,
            decoder=decoder,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
        )
    elif cfg.model.name == "SpecTrDr":
        feature_extractor = get_feature_extractor(cfg.feature_extractor, feature_dim, num_timesteps, )
        try:
            extra_channels = cfg.feature_extractor.params.base_filters * len(cfg.feature_extractor.params.kernel_sizes) * len(cfg.feature_extractor.params.wave_layers)
        except:
            extra_channels = cfg.feature_extractor.params.base_filters * len(cfg.feature_extractor.params.kernel_size) * len(cfg.feature_extractor.params.wave_layers)

        decoder = get_decoder(cfg.decoder, None, n_classes, num_timesteps, feature_extractor=feature_extractor, extra_channels=extra_channels)
        model = SpecTrDr(
            decoder=decoder,
            mixup_alpha=cfg.aug.mixup_alpha,
            cutmix_alpha=cfg.aug.cutmix_alpha,
        )
    
    else:
        raise ValueError(f"Invalid model name: {cfg.model.name}")

    return model
