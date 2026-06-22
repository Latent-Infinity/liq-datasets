"""Combinatorial purged cross-validation paths for mean-reversion labels."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import combinations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from liq.datasets.mean_reversion.labels import TripleBarrierLabel


class CPCVConfig(BaseModel):
    """Configuration for combinatorial purged cross-validation path generation."""

    model_config = ConfigDict(frozen=True)

    n_splits: int = Field(gt=1)
    n_test_splits: int = Field(gt=0)
    embargo_bars: int = Field(ge=0)

    @model_validator(mode="after")
    def _valid_test_split_count(self) -> CPCVConfig:
        if self.n_test_splits >= self.n_splits:
            raise ValueError("n_test_splits must be smaller than n_splits")
        return self


class CPCVPath(BaseModel):
    """One purged train/test path over label indices."""

    model_config = ConfigDict(frozen=True)

    path_id: str
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    test_folds: tuple[int, ...]


def embargo_width(L_vol: int, L_base: int, H: int) -> int:
    """Return the union embargo width for scanner lookbacks and label horizon."""
    if L_vol <= 0 or L_base <= 0 or H <= 0:
        raise ValueError("L_vol, L_base, and H must be positive")
    return max(L_vol, L_base, H)


def _fold_bounds(label_count: int, n_splits: int) -> tuple[tuple[int, int], ...]:
    base, remainder = divmod(label_count, n_splits)
    bounds: list[tuple[int, int]] = []
    start = 0
    for fold in range(n_splits):
        width = base + (1 if fold < remainder else 0)
        stop = start + width
        bounds.append((start, stop))
        start = stop
    return tuple(bounds)


def _fold_indices(bounds: Sequence[tuple[int, int]], folds: Sequence[int]) -> tuple[int, ...]:
    indices: list[int] = []
    for fold in folds:
        start, stop = bounds[fold]
        indices.extend(range(start, stop))
    return tuple(indices)


def _train_indices(
    label_count: int, test_indices: Sequence[int], embargo_bars: int
) -> tuple[int, ...]:
    test_set = set(test_indices)
    train: list[int] = []
    for index in range(label_count):
        if index in test_set:
            continue
        if any(abs(index - test_index) <= embargo_bars for test_index in test_set):
            continue
        train.append(index)
    return tuple(train)


def generate_paths(labels: Sequence[TripleBarrierLabel], config: CPCVConfig) -> Iterator[CPCVPath]:
    """Yield every unique purged train/test fold combination in deterministic order."""
    label_count = len(labels)
    if label_count < config.n_splits:
        raise ValueError("labels must contain at least n_splits rows")
    bounds = _fold_bounds(label_count, config.n_splits)
    for test_folds in combinations(range(config.n_splits), config.n_test_splits):
        test_indices = _fold_indices(bounds, test_folds)
        yield CPCVPath(
            path_id="cpcv:" + ",".join(str(fold) for fold in test_folds),
            train_indices=_train_indices(label_count, test_indices, config.embargo_bars),
            test_indices=test_indices,
            test_folds=tuple(test_folds),
        )


__all__ = ["CPCVConfig", "CPCVPath", "embargo_width", "generate_paths"]
