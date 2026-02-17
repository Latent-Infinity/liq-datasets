"""Holdout splitting with embargo and audit logs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class HoldoutConfig:
    train_end_ts: datetime
    dev_end_ts: datetime
    lockbox_end_ts: datetime
    label_lookahead_bars: int
    embargo_bars: int


@dataclass(frozen=True)
class HoldoutSplit:
    train: slice
    dev: slice
    lockbox: slice


class HoldoutManager:
    """Create train/dev/lockbox splits with embargo and lookahead guards."""

    def __init__(self, timestamps: Sequence[datetime], config: HoldoutConfig) -> None:
        if not timestamps:
            raise ValueError("timestamps must be non-empty")
        self.timestamps = list(timestamps)
        self.config = config
        if not (config.train_end_ts < config.dev_end_ts < config.lockbox_end_ts):
            raise ValueError("train_end_ts < dev_end_ts < lockbox_end_ts must hold")
        if config.label_lookahead_bars < 0 or config.embargo_bars < 0:
            raise ValueError("label_lookahead_bars and embargo_bars must be non-negative")
        if any(self.timestamps[i] > self.timestamps[i + 1] for i in range(len(self.timestamps) - 1)):
            raise ValueError("timestamps must be sorted ascending")

    def _find_end_index(self, ts: datetime) -> int:
        candidates = [i for i, t in enumerate(self.timestamps) if t <= ts]
        if not candidates:
            raise ValueError("No timestamps available at or before requested boundary")
        return max(candidates)

    def split(self) -> HoldoutSplit:
        train_end = self._find_end_index(self.config.train_end_ts)
        dev_end = self._find_end_index(self.config.dev_end_ts)
        lockbox_end = self._find_end_index(self.config.lockbox_end_ts)

        lookahead = self.config.label_lookahead_bars
        embargo = self.config.embargo_bars

        train_end = train_end - lookahead
        dev_end = dev_end - lookahead
        lockbox_end = lockbox_end - lookahead

        if train_end <= 0 or dev_end <= train_end or lockbox_end <= dev_end:
            raise ValueError("Holdout boundaries invalid after lookahead adjustment")

        dev_start = train_end + embargo
        lockbox_start = dev_end + embargo

        if dev_start >= dev_end or lockbox_start >= lockbox_end:
            raise ValueError("Holdout boundaries invalid after embargo adjustment")

        return HoldoutSplit(
            train=slice(0, train_end + 1),
            dev=slice(dev_start, dev_end + 1),
            lockbox=slice(lockbox_start, lockbox_end + 1),
        )

    def get_audit_log(self) -> dict[str, object]:
        split = self.split()
        return {
            "train": {"start": split.train.start, "end": split.train.stop},
            "dev": {"start": split.dev.start, "end": split.dev.stop},
            "lockbox": {"start": split.lockbox.start, "end": split.lockbox.stop},
            "label_lookahead_bars": self.config.label_lookahead_bars,
            "embargo_bars": self.config.embargo_bars,
        }
