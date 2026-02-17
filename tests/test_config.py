
import polars as pl
import pytest

from liq.datasets.config import FeatureSchema, SequenceConfig


def test_sequence_config_derivations() -> None:
    cfg = SequenceConfig(context_bars=8, max_horizon_bars=4, patch_size=2)
    assert cfg.x_full_bars == 12
    assert cfg.context_patches == 4
    assert cfg.max_horizon_patches == 2
    assert cfg.x_full_patches == 6


def test_sequence_config_invalid_patch() -> None:
    with pytest.raises(ValueError):
        SequenceConfig(context_bars=7, max_horizon_bars=4, patch_size=2)


def test_sequence_config_negative_values() -> None:
    with pytest.raises(ValueError):
        SequenceConfig(context_bars=-1, max_horizon_bars=4, patch_size=2)


def test_feature_schema_apply_df() -> None:
    schema = FeatureSchema(features=["a", "b", "c"])
    df = pl.DataFrame({"b": [1.0], "a": [2.0]})
    out = schema.apply_df(df)
    assert out.columns == ["a", "b", "c"]
    assert out["c"][0] == 0.0


def test_feature_schema_validate_df() -> None:
    schema = FeatureSchema(features=["x", "y"])
    df = pl.DataFrame({"x": [1.0], "z": [2.0]})
    result = schema.validate_df(df)
    assert result["missing"] == ["y"]
    assert result["extra"] == ["z"]


def test_feature_schema_duplicate_features() -> None:
    with pytest.raises(ValueError):
        FeatureSchema(features=["a", "a"])
