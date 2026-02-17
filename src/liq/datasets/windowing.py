"""Window building utilities for SSL and supervised datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl

from liq.datasets.config import FeatureSchema, SequenceConfig


@dataclass(frozen=True)
class WindowBuilder:
    """Builds fixed-shape windows for SSL and supervised datasets."""

    seq_config: SequenceConfig
    feature_schema: FeatureSchema

    def _to_matrix(self, df: pl.DataFrame) -> np.ndarray:
        aligned = self.feature_schema.apply_df(df)
        return aligned.to_numpy()

    def build_ssl_windows(self, df: pl.DataFrame, stride_bars: int) -> np.ndarray:
        if stride_bars <= 0:
            raise ValueError("stride_bars must be positive")
        matrix = self._to_matrix(df)
        total = matrix.shape[0]
        window = self.seq_config.x_full_bars
        if total < window:
            return np.empty((0, window, self.feature_schema.input_dim))

        windows = []
        for start in range(0, total - window + 1, stride_bars):
            windows.append(matrix[start : start + window])
        if not windows:
            return np.empty((0, window, self.feature_schema.input_dim))
        return np.stack(windows)

    def build_supervised_windows(
        self,
        df: pl.DataFrame,
        y_dir: Iterable[int],
        y_ret: Iterable[float],
        stride_bars: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if stride_bars <= 0:
            raise ValueError("stride_bars must be positive")
        matrix = self._to_matrix(df)
        y_dir_arr = np.asarray(list(y_dir))
        y_ret_arr = np.asarray(list(y_ret))
        if y_dir_arr.shape[0] != matrix.shape[0] or y_ret_arr.shape[0] != matrix.shape[0]:
            raise ValueError("y_dir and y_ret must align with feature rows")

        total = matrix.shape[0]
        context = self.seq_config.context_bars
        horizon = self.seq_config.max_horizon_bars
        if total < context + horizon:
            return (
                np.empty((0, context, self.feature_schema.input_dim)),
                np.empty((0,), dtype=y_dir_arr.dtype),
                np.empty((0,), dtype=y_ret_arr.dtype),
            )

        windows = []
        labels_dir = []
        labels_ret = []
        max_start = total - (context + horizon) + 1
        for start in range(0, max_start, stride_bars):
            end = start + context
            label_idx = end - 1
            windows.append(matrix[start:end])
            labels_dir.append(y_dir_arr[label_idx])
            labels_ret.append(y_ret_arr[label_idx])

        X = np.stack(windows) if windows else np.empty((0, context, self.feature_schema.input_dim))
        return X, np.asarray(labels_dir), np.asarray(labels_ret)

    def build_ssl_multiview(
        self,
        view_a: pl.DataFrame,
        view_b: pl.DataFrame,
        stride_bars: int,
    ) -> dict[str, np.ndarray]:
        if "timestamp" in view_a.columns and "timestamp" in view_b.columns and view_a["timestamp"].to_list() != view_b["timestamp"].to_list():
            raise ValueError("view_a and view_b timestamps must be aligned")
        return {
            "view_a": self.build_ssl_windows(view_a, stride_bars),
            "view_b": self.build_ssl_windows(view_b, stride_bars),
        }
