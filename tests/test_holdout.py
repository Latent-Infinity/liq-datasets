from datetime import UTC, datetime, timedelta

import pytest

from liq.datasets.holdout import HoldoutConfig, HoldoutManager


def test_holdout_splits_with_embargo() -> None:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [ts0 + timedelta(hours=i) for i in range(30)]
    cfg = HoldoutConfig(
        train_end_ts=timestamps[19],
        dev_end_ts=timestamps[24],
        lockbox_end_ts=timestamps[29],
        label_lookahead_bars=2,
        embargo_bars=1,
    )
    manager = HoldoutManager(timestamps, cfg)
    split = manager.split()
    # train end adjusted for lookahead
    assert split.train.stop == 18
    # dev starts after embargo
    assert split.dev.start == split.train.stop
    assert split.dev.stop == 23
    assert split.lockbox.start == split.dev.stop


def test_holdout_invalid_order() -> None:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [ts0 + timedelta(hours=i) for i in range(5)]
    cfg = HoldoutConfig(
        train_end_ts=timestamps[3],
        dev_end_ts=timestamps[2],
        lockbox_end_ts=timestamps[4],
        label_lookahead_bars=0,
        embargo_bars=0,
    )
    with pytest.raises(ValueError):
        HoldoutManager(timestamps, cfg)


def test_holdout_empty_timestamps() -> None:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    cfg = HoldoutConfig(
        train_end_ts=ts0,
        dev_end_ts=ts0,
        lockbox_end_ts=ts0,
        label_lookahead_bars=0,
        embargo_bars=0,
    )
    with pytest.raises(ValueError):
        HoldoutManager([], cfg)


def test_holdout_negative_bars() -> None:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [ts0 + timedelta(hours=i) for i in range(5)]
    cfg = HoldoutConfig(
        train_end_ts=timestamps[1],
        dev_end_ts=timestamps[2],
        lockbox_end_ts=timestamps[3],
        label_lookahead_bars=-1,
        embargo_bars=0,
    )
    with pytest.raises(ValueError):
        HoldoutManager(timestamps, cfg)


def test_holdout_invalid_after_lookahead() -> None:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [ts0 + timedelta(hours=i) for i in range(6)]
    cfg = HoldoutConfig(
        train_end_ts=timestamps[2],
        dev_end_ts=timestamps[3],
        lockbox_end_ts=timestamps[5],
        label_lookahead_bars=3,
        embargo_bars=0,
    )
    manager = HoldoutManager(timestamps, cfg)
    with pytest.raises(ValueError):
        manager.split()


def test_holdout_unsorted_timestamps() -> None:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [ts0 + timedelta(hours=1), ts0]
    cfg = HoldoutConfig(
        train_end_ts=timestamps[1],
        dev_end_ts=timestamps[0] + timedelta(hours=1),
        lockbox_end_ts=timestamps[0] + timedelta(hours=2),
        label_lookahead_bars=0,
        embargo_bars=0,
    )
    with pytest.raises(ValueError):
        HoldoutManager(timestamps, cfg)


def test_holdout_boundary_before_first_timestamp() -> None:
    ts0 = datetime(2024, 1, 1, tzinfo=UTC)
    timestamps = [ts0 + timedelta(hours=i) for i in range(5)]
    cfg = HoldoutConfig(
        train_end_ts=ts0 - timedelta(hours=1),
        dev_end_ts=timestamps[2],
        lockbox_end_ts=timestamps[4],
        label_lookahead_bars=0,
        embargo_bars=0,
    )
    manager = HoldoutManager(timestamps, cfg)
    with pytest.raises(ValueError):
        manager.split()
