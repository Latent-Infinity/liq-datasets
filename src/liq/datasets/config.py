"""Dataset contracts: sequence configuration and feature schema."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class SequenceConfig:
    """Sequence and patching configuration for windowed datasets."""

    context_bars: int
    max_horizon_bars: int
    patch_size: int

    def __post_init__(self) -> None:
        if self.context_bars <= 0 or self.max_horizon_bars <= 0 or self.patch_size <= 0:
            raise ValueError("context_bars, max_horizon_bars, patch_size must be positive")
        if self.context_bars % self.patch_size != 0:
            raise ValueError("context_bars must be divisible by patch_size")
        if self.max_horizon_bars % self.patch_size != 0:
            raise ValueError("max_horizon_bars must be divisible by patch_size")
        if self.x_full_bars % self.patch_size != 0:
            raise ValueError("x_full_bars must be divisible by patch_size")

    @property
    def x_full_bars(self) -> int:
        return self.context_bars + self.max_horizon_bars

    @property
    def context_patches(self) -> int:
        return self.context_bars // self.patch_size

    @property
    def max_horizon_patches(self) -> int:
        return self.max_horizon_bars // self.patch_size

    @property
    def x_full_patches(self) -> int:
        return self.x_full_bars // self.patch_size

    def summary(self) -> dict[str, int]:
        return {
            "context_bars": self.context_bars,
            "max_horizon_bars": self.max_horizon_bars,
            "patch_size": self.patch_size,
            "x_full_bars": self.x_full_bars,
            "context_patches": self.context_patches,
            "max_horizon_patches": self.max_horizon_patches,
            "x_full_patches": self.x_full_patches,
        }


@dataclass(frozen=True)
class FeatureSchema:
    """Feature schema with fixed ordering and default fill."""

    features: list[str]
    name: str = "default"

    def __post_init__(self) -> None:
        if not self.features:
            raise ValueError("features must be non-empty")
        if len(set(self.features)) != len(self.features):
            raise ValueError("features must be unique")

    @property
    def input_dim(self) -> int:
        return len(self.features)

    def apply_df(self, df: pl.DataFrame, fill_value: float = 0.0) -> pl.DataFrame:
        """Return DataFrame with schema order and missing columns filled."""
        missing = [feat for feat in self.features if feat not in df.columns]
        if missing:
            df = df.with_columns([pl.lit(fill_value).alias(col) for col in missing])
        df = df.select(self.features)
        return df

    def validate_df(self, df: pl.DataFrame) -> dict[str, list[str]]:
        """Return missing/extra feature names without mutating."""
        missing = [feat for feat in self.features if feat not in df.columns]
        extra = [col for col in df.columns if col not in self.features]
        return {"missing": missing, "extra": extra}
