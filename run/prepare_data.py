import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.common import trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "minute",
    "anglez",
    "anglez_original",
    "anglez_sin",
    "anglez_cos",
    "enmo",
    "hour_sin",
    "hour_cos",
    "anglez_diff_p1",
    "anglez_diff_p2",
    "anglez_diff_p3",
    "anglez_diff_p4",
    "enmo_diff_p1",
    "enmo_diff_p2",
    "enmo_diff_p3",
    "enmo_diff_p4",
    "anglez_diff_rolling_med_60",
    "anglez_diff_rolling_mean_60",
    "anglez_diff_rolling_max_60",
    "anglez_diff_rolling_min_60",
    "anglez_diff_rolling_max_min_60",
    "anglez_diff_rolling_std_60",
    "anglez_diff_rolling_quantile_25_60",
    "anglez_diff_rolling_quantile_975_60",
    "enmo_diff_rolling_med_60",
    "enmo_diff_rolling_mean_60",
    "enmo_diff_rolling_max_60",
    "enmo_diff_rolling_min_60",
    "enmo_diff_rolling_max_min_60",
    "enmo_diff_rolling_std_60",
    "enmo_diff_rolling_quantile_25_60",
    "enmo_diff_rolling_quantile_975_60",
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]

def minute_is_important(x: pl.Expr) -> pl.Expr:
    result = (x % 15).cast(pl.Int8)
    return [result.alias("minute")]

def to_rad_coord(x: pl.Expr, name: str) -> list[pl.Expr]:
    rad = x * np.pi / 180
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def diff_rolling_feats(x: pl.Expr, window: int, name, limited: bool = False) -> pl.Expr:
    x_diff_p1 = x.diff(1).abs().fill_null(0)
    x_diff_p2 = x.diff(2).abs().fill_null(0)
    x_diff_p3 = x.diff(3).abs().fill_null(0)
    x_diff_p4 = x.diff(4).abs().fill_null(0)
    x_diff_rolling_med = x_diff_p1.rolling_median(window, center=True).fill_null(0)
    x_diff_rolling_mean = x_diff_p1.rolling_mean(window, center=True).fill_null(0)
    x_diff_rolling_std = x_diff_p1.rolling_std(window, center=True).fill_null(0)
    x_diff_rolling_quantile_25 = x_diff_p1.rolling_quantile(
        0.025, "nearest", center=True
    ).fill_null(0)
    x_diff_rolling_quantile_975 = x_diff_p1.rolling_quantile(
        0.975, "nearest", center=True
    ).fill_null(0)
    x_diff_rolling_max = x_diff_p1.rolling_max(window, center=True).fill_null(0)
    x_diff_rolling_min = x_diff_p1.rolling_min(window, center=True).fill_null(0)
    x_diff_rolling_max_min = x_diff_rolling_max - x_diff_rolling_min
    if not limited:
        return [
            x_diff_p1.alias(f"{name}_diff_p1"),
            x_diff_p2.alias(f"{name}_diff_p2"),
            x_diff_p3.alias(f"{name}_diff_p3"),
            x_diff_p4.alias(f"{name}_diff_p4"),
            x_diff_rolling_med.alias(f"{name}_diff_rolling_med_{window}"),
            x_diff_rolling_mean.alias(f"{name}_diff_rolling_mean_{window}"),
            x_diff_rolling_max.alias(f"{name}_diff_rolling_max_{window}"),
            x_diff_rolling_min.alias(f"{name}_diff_rolling_min_{window}"),
            x_diff_rolling_max_min.alias(f"{name}_diff_rolling_max_min_{window}"),
            x_diff_rolling_std.alias(f"{name}_diff_rolling_std_{window}"),
            x_diff_rolling_quantile_25.alias(f"{name}_diff_rolling_quantile_25_{window}"),
            x_diff_rolling_quantile_975.alias(f"{name}_diff_rolling_quantile_975_{window}"),
        ]
    else:
        return [
            x_diff_rolling_med.alias(f"{name}_diff_rolling_med_{window}"),
            x_diff_rolling_mean.alias(f"{name}_diff_rolling_mean_{window}"),
            x_diff_rolling_max.alias(f"{name}_diff_rolling_max_{window}"),
            x_diff_rolling_min.alias(f"{name}_diff_rolling_min_{window}"),
            x_diff_rolling_max_min.alias(f"{name}_diff_rolling_max_min_{window}"),
            x_diff_rolling_std.alias(f"{name}_diff_rolling_std_{window}"),
            x_diff_rolling_quantile_25.alias(f"{name}_diff_rolling_quantile_25_{window}"),
            x_diff_rolling_quantile_975.alias(f"{name}_diff_rolling_quantile_975_{window}"),
        ]


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = series_df.with_columns(
        *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
        *minute_is_important(pl.col("timestamp").dt.minute()),
        *to_rad_coord(pl.col("anglez_original"), "anglez"),
        *diff_rolling_feats(pl.col("anglez"), 60, "anglez", False),
        *diff_rolling_feats(pl.col("enmo"), 60, "enmo", False),
    ).select("series_id", *FEATURE_NAMES)
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z", time_zone="UTC"),
                pl.col("anglez").alias("anglez_original"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("anglez_original"),
                    pl.col("enmo"),
                    pl.col("timestamp"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            this_series_df = add_feature(this_series_df)

            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)


if __name__ == "__main__":
    main()