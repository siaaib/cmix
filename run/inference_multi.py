from pathlib import Path

import hydra
import numpy as np
import polars as pl
import torch
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from src.conf import InferenceConfig
from src.datamodule import load_chunk_features
from src.dataset.common import get_test_ds
from src.models.base import BaseModel
from src.models.common import get_model
from src.utils.common import nearest_valid_size, trace
from src.utils.post_process import post_process_for_seg


def load_model(cfg: InferenceConfig, weight_path) -> BaseModel:
    num_timesteps = nearest_valid_size(int(cfg.duration * cfg.upsample_rate), cfg.downsample_rate)
    model = get_model(
        cfg,
        feature_dim=len(cfg.features),
        n_classes=len(cfg.labels),
        num_timesteps=num_timesteps // cfg.downsample_rate,
    )

    model.load_state_dict(torch.load(weight_path))
    print('load weight from "{}"'.format(weight_path))
    return model


def get_test_dataloader(cfg: InferenceConfig) -> DataLoader:
    """get test dataloader

    Args:
        cfg (DictConfig): config

    Returns:
        DataLoader: test dataloader
    """
    feature_dir = Path("/kaggle/working/processed_data/test")
    series_ids = [x.name for x in feature_dir.glob("*")]
    chunk_features = load_chunk_features(
        duration=cfg.duration,
        feature_names=cfg.features,
        series_ids=series_ids,
        processed_dir=Path(cfg.dir.processed_dir),
        phase=cfg.phase,
    )
    test_dataset = get_test_ds(cfg, chunk_features=chunk_features)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader


def inference(
    duration: int, loader: DataLoader, models: nn.Module, device: torch.device, use_amp, average_type='mean'
) -> tuple[list[str], np.ndarray]:
    for model in models:
        model = model.to(device)
        model.eval()

    preds = []
    keys = []
    for batch in tqdm(loader, desc="inference"):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp):
                x = batch["feature"].to(device)
                all_preds = []               
                for model in models:
                    output = model.predict(
                    x,
                    org_duration=duration,
                    )
                    # Append the predictions to the list
                    all_preds.append(output.preds)

                all_preds = torch.stack(all_preds, dim=0)
                if average_type == 'median':
                    pred, _ = torch.median(all_preds, dim=0)
                elif average_type == 'mean':
                    pred = torch.mean(all_preds, dim=0)
                else:
                    pred = (torch.median(all_preds, dim=0)[0] + torch.mean(all_preds, dim=0))/2
            key = batch["key"]
            preds.append(pred.detach().cpu().numpy())
            keys.extend(key)

    preds = np.concatenate(preds)

    return keys, preds  # type: ignore


def make_submission(
    keys: list[str], preds: np.ndarray, score_th, distance
) -> pl.DataFrame:
    sub_df = post_process_for_seg(
        keys,
        preds,  # type: ignore
        score_th=score_th,
        distance=distance,  # type: ignore
    )

    return sub_df


@hydra.main(config_path="conf", config_name="inference_multi", version_base="1.2")
def main(cfg: InferenceConfig):
    seed_everything(cfg.seed)

    with trace("load test dataloader"):
        test_dataloader = get_test_dataloader(cfg)
    with trace("load model"):
        models = []
        for model_path in cfg.model_path_list:
            model = load_model(cfg, model_path)
            models.append(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with trace("inference"):
        keys, preds = inference(cfg.duration, test_dataloader, models, device, use_amp=cfg.use_amp, average_type=cfg.average_type)

    with trace("make submission"):
        sub_df = make_submission(
            keys,
            preds,
            score_th=cfg.pp.score_th,
            distance=cfg.pp.distance,
        )
    sub_df.write_csv(Path(cfg.dir.sub_dir) / "submission.csv")


if __name__ == "__main__":
    main()
