from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl

from liq.datasets.config import FeatureSchema, SequenceConfig
from liq.datasets.windowing import WindowBuilder


def _sample_df(rows: int) -> pl.DataFrame:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    return pl.DataFrame({
        "timestamp": [ts0 + timedelta(hours=i) for i in range(rows)],
        "a": list(range(rows)),
        "b": [i * 10 for i in range(rows)],
    })


def test_ssl_window_shapes() -> None:
    cfg = SequenceConfig(context_bars=4, max_horizon_bars=2, patch_size=2)
    schema = FeatureSchema(features=["a", "b"])
    builder = WindowBuilder(cfg, schema)
    df = _sample_df(10)
    windows = builder.build_ssl_windows(df, stride_bars=2)
    assert windows.shape == (3, 6, 2)
    assert np.all(windows[0, 0] == np.array([0, 0]))


def test_supervised_alignment() -> None:
    cfg = SequenceConfig(context_bars=4, max_horizon_bars=2, patch_size=2)
    schema = FeatureSchema(features=["a", "b"])
    builder = WindowBuilder(cfg, schema)
    df = _sample_df(10)
    y_dir = list(range(10))
    y_ret = [float(i) for i in range(10)]
    X, y_d, y_r = builder.build_supervised_windows(df, y_dir, y_ret, stride_bars=2)
    assert X.shape == (3, 4, 2)
    # First window labels should align to end-1 index (context_bars - 1)
    assert y_d[0] == y_dir[3]
    assert y_r[0] == y_ret[3]


def test_windowing_invalid_stride() -> None:
    cfg = SequenceConfig(context_bars=4, max_horizon_bars=2, patch_size=2)
    schema = FeatureSchema(features=["a", "b"])
    builder = WindowBuilder(cfg, schema)
    df = _sample_df(10)
    try:
        builder.build_ssl_windows(df, stride_bars=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for stride_bars=0")


def test_windowing_too_short() -> None:
    cfg = SequenceConfig(context_bars=4, max_horizon_bars=2, patch_size=2)
    schema = FeatureSchema(features=["a", "b"])
    builder = WindowBuilder(cfg, schema)
    df = _sample_df(3)
    windows = builder.build_ssl_windows(df, stride_bars=1)
    assert windows.shape[0] == 0
    X, y_d, y_r = builder.build_supervised_windows(df, [0, 1, 2], [0.0, 1.0, 2.0], 1)
    assert X.shape[0] == 0
    assert y_d.shape[0] == 0
    assert y_r.shape[0] == 0


def test_multiview_length_mismatch() -> None:
    cfg = SequenceConfig(context_bars=4, max_horizon_bars=2, patch_size=2)
    schema = FeatureSchema(features=["a", "b"])
    builder = WindowBuilder(cfg, schema)
    df_a = _sample_df(5)
    df_b = _sample_df(6)
    try:
        builder.build_ssl_multiview(df_a, df_b, stride_bars=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched view lengths")


def test_multiview_timestamp_mismatch() -> None:
    cfg = SequenceConfig(context_bars=4, max_horizon_bars=2, patch_size=2)
    schema = FeatureSchema(features=["a", "b"])
    builder = WindowBuilder(cfg, schema)
    df_a = _sample_df(5)
    df_b = _sample_df(5).with_columns(
        (pl.col("timestamp") + pl.duration(hours=1)).alias("timestamp")
    )
    try:
        builder.build_ssl_multiview(df_a, df_b, stride_bars=1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched timestamps")
