"""Walk-forward split definitions and generators."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from collections.abc import Sequence


SplitBoundary = slice | tuple[datetime, datetime]
_SLICE_ID_PREFIX = "time_window"


def _validate_timestamps(timestamps: Sequence[datetime]) -> None:
    if not timestamps:
        raise ValueError("timestamps must be non-empty")
    if any(t0 > t1 for t0, t1 in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("timestamps must be sorted ascending")


def _is_aware(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


def _check_timezone(reference: datetime, candidate: datetime, *, field: str) -> None:
    if not _is_aware(candidate):
        raise ValueError(f"{field} timestamps must be timezone-aware")
    if not _is_aware(reference):
        raise ValueError("timestamps must be timezone-aware")
    if candidate.tzinfo is not reference.tzinfo:
        raise ValueError("timestamp timezones must match index tz")


def _safe_to_slice(
    boundary: tuple[datetime, datetime],
    index: Sequence[datetime],
) -> slice:
    _check_timezone(index[0], boundary[0], field="start")
    _check_timezone(index[0], boundary[1], field="end")
    if boundary[0] >= boundary[1]:
        raise ValueError("tuple boundaries must satisfy start < end")

    start = bisect_left(index, boundary[0])
    stop = bisect_left(index, boundary[1])
    if start >= stop:
        raise ValueError("resolved boundary range is empty")
    return slice(start, stop)


@dataclass(frozen=True)
class WalkForwardSplit:
    """Canonical walk-forward split with optional lockbox.

    Accepts both integer ``slice`` boundaries (provider-ready) and datetime
    tuple boundaries (user-facing). `dev` is alias for `validate`.
    """

    train: SplitBoundary
    validate: SplitBoundary
    test: SplitBoundary
    lockbox: SplitBoundary | None = None
    slice_id: str = "time_window:auto"
    embargo_bars: int = 0

    def __post_init__(self) -> None:
        if self.embargo_bars < 0:
            raise ValueError("embargo_bars must be non-negative")
        for label, boundary in (
            ("train", self.train),
            ("validate", self.validate),
            ("test", self.test),
        ):
            if not isinstance(boundary, (slice, tuple)):
                raise TypeError(f"{label} boundary must be slice or tuple[datetime, datetime]")
            if isinstance(boundary, tuple) and len(boundary) != 2:
                raise ValueError(f"{label} tuple boundary requires start/end datetimes")
            if isinstance(boundary, tuple) and (
                not isinstance(boundary[0], datetime) or not isinstance(boundary[1], datetime)
            ):
                raise TypeError(f"{label} tuple boundary requires start/end datetimes")

        if self.lockbox is not None and not isinstance(self.lockbox, (slice, tuple)):
            raise TypeError("lockbox boundary must be None or slice or tuple[datetime, datetime]")
        if isinstance(self.lockbox, tuple) and len(self.lockbox) != 2:
            raise ValueError("lockbox tuple boundary requires start/end datetimes")
        if isinstance(self.lockbox, tuple) and (
            not isinstance(self.lockbox[0], datetime) or not isinstance(self.lockbox[1], datetime)
        ):
            raise TypeError("lockbox tuple boundary requires start/end datetimes")

    @property
    def dev(self) -> SplitBoundary:
        """Backwards-compatible alias for validate."""
        return self.validate

    def to_bar_slices(self, index: Sequence[datetime]) -> "WalkForwardSplit":
        """Convert datetime boundaries to integer slices against an index.

        Datetime boundaries use left-closed, right-open semantics:
        ``[start, end)``.
        """
        if not index:
            raise ValueError("index must be non-empty")
        if any(t0 > t1 for t0, t1 in zip(index, index[1:], strict=False)):
            raise ValueError("index must be sorted ascending")
        resolved = {}
        for name, boundary in (
            ("train", self.train),
            ("validate", self.validate),
            ("test", self.test),
        ):
            resolved[name] = _to_slice(boundary, index, name=name)

        resolved_lockbox: slice | None = None
        if self.lockbox is not None:
            resolved_lockbox = _to_slice(self.lockbox, index, name="lockbox")

        return WalkForwardSplit(
            train=resolved["train"],
            validate=resolved["validate"],
            test=resolved["test"],
            lockbox=resolved_lockbox,
            slice_id=self.slice_id,
            embargo_bars=self.embargo_bars,
        )


def _to_slice(
    boundary: SplitBoundary,
    index: Sequence[datetime],
    *,
    name: str,
) -> slice:
    if isinstance(boundary, slice):
        start = 0 if boundary.start is None else int(boundary.start)
        stop = len(index) if boundary.stop is None else int(boundary.stop)
        if start < 0 or stop < 0:
            raise ValueError(f"{name} slice index must be non-negative")
        return slice(start, min(stop, len(index)))

    if len(boundary) != 2:  # type: ignore[arg-type]
        raise ValueError(f"{name} boundary tuple requires start/end datetimes")
    return _safe_to_slice(boundary, index)


def generate_walk_forward_splits(
    timestamps: Sequence[datetime],
    *,
    train_window: int,
    validate_window: int,
    test_window: int,
    step_size: int,
    embargo_bars: int = 0,
    label_lookahead_bars: int = 0,
) -> list[WalkForwardSplit]:
    """Generate canonical walk-forward triple splits over the full timestamp list."""
    if train_window <= 0 or validate_window <= 0 or test_window <= 0:
        raise ValueError("window sizes must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if embargo_bars < 0 or label_lookahead_bars < 0:
        raise ValueError("embargo_bars and label_lookahead_bars must be non-negative")

    _validate_timestamps(timestamps)
    n = len(timestamps)
    splits: list[WalkForwardSplit] = []
    start = 0
    split_idx = 0
    while True:
        raw_train_start = start
        raw_train_stop = raw_train_start + train_window
        raw_validate_start = raw_train_stop + embargo_bars
        raw_validate_stop = raw_validate_start + validate_window
        raw_test_start = raw_validate_stop + embargo_bars
        raw_test_stop = raw_test_start + test_window

        if raw_test_stop > n:
            break

        train_stop = raw_train_stop - label_lookahead_bars
        validate_stop = raw_validate_stop - label_lookahead_bars
        test_stop = raw_test_stop - label_lookahead_bars

        if train_stop <= raw_train_start:
            raise ValueError("lookahead too large for train window")
        if validate_stop <= raw_train_stop:
            raise ValueError("lookahead too large for validate window")
        if test_stop <= raw_validate_stop:
            raise ValueError("lookahead too large for test window")

        validate_start = train_stop + embargo_bars
        test_start = validate_stop + embargo_bars

        split = WalkForwardSplit(
            train=slice(raw_train_start, train_stop),
            validate=slice(validate_start, validate_stop),
            test=slice(test_start, test_stop),
            slice_id=(
                f"{_SLICE_ID_PREFIX}:{split_idx}"
                f":start={raw_train_start}:end={raw_train_start + train_window}"
            ),
            embargo_bars=embargo_bars,
        )
        splits.append(split)

        split_idx += 1
        start += step_size
        if start >= n:
            break

    if not splits:
        raise ValueError("walk-forward parameters produced no complete splits")
    return splits
