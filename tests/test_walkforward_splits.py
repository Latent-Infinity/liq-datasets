from datetime import UTC, datetime, timedelta, timezone

import pytest

from liq.datasets.walk_forward import WalkForwardSplit, generate_walk_forward_splits, _to_slice


def _hours(start: datetime, count: int) -> list[datetime]:
    return [start + timedelta(hours=i) for i in range(count)]


def test_generate_walk_forward_splits_basic() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 31)
    splits = generate_walk_forward_splits(
        timestamps,
        train_window=5,
        validate_window=4,
        test_window=3,
        step_size=5,
    )

    assert len(splits) == 4
    assert splits[0].train == slice(0, 5)
    assert splits[0].validate == slice(5, 9)
    assert splits[0].test == slice(9, 12)
    assert splits[1].train == slice(5, 10)
    assert splits[1].validate == slice(10, 14)
    assert splits[1].test == slice(14, 17)
    assert splits[0].slice_id.startswith("time_window:")

    for split in splits:
        assert split.train.stop <= split.validate.start
        assert split.validate.stop <= split.test.start
        assert split.validate.start == split.train.stop + split.embargo_bars
        assert split.test.start == split.validate.stop + split.embargo_bars


def test_generate_walk_forward_splits_with_lookahead() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 32)
    splits = generate_walk_forward_splits(
        timestamps,
        train_window=5,
        validate_window=4,
        test_window=3,
        step_size=5,
        label_lookahead_bars=1,
    )

    assert len(splits) == 5
    assert splits[0].train == slice(0, 4)
    assert splits[0].validate == slice(4, 8)
    assert splits[0].test == slice(8, 11)


def test_generate_walk_forward_splits_with_embargo() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 20)
    splits = generate_walk_forward_splits(
        timestamps,
        train_window=4,
        validate_window=3,
        test_window=2,
        step_size=4,
        embargo_bars=1,
    )

    assert len(splits) == 3
    assert splits[0].train == slice(0, 4)
    assert splits[0].validate == slice(5, 8)
    assert splits[0].test == slice(9, 11)
    assert splits[1].train == slice(4, 8)
    assert splits[1].validate == slice(9, 12)
    assert splits[1].test == slice(13, 15)
    assert splits[2].train == slice(8, 12)
    assert splits[2].validate == slice(13, 16)
    assert splits[2].test == slice(17, 19)

    for split in splits:
        assert split.validate.start == split.train.stop + split.embargo_bars
        assert split.test.start == split.validate.stop + split.embargo_bars


def test_generate_walk_forward_splits_rejects_invalid_windows() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    with pytest.raises(ValueError):
        generate_walk_forward_splits(
            timestamps,
            train_window=4,
            validate_window=4,
            test_window=4,
            step_size=5,
            label_lookahead_bars=3,
        )


def test_generate_walk_forward_splits_rejects_no_complete_windows() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 6)
    with pytest.raises(ValueError, match="walk-forward parameters produced no complete splits"):
        generate_walk_forward_splits(
            timestamps,
            train_window=4,
            validate_window=4,
            test_window=4,
            step_size=4,
            embargo_bars=1,
        )


def test_generate_walk_forward_splits_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="timestamps must be non-empty"):
        generate_walk_forward_splits(
            [],
            train_window=1,
            validate_window=1,
            test_window=1,
            step_size=1,
        )

    with pytest.raises(ValueError, match="timestamps must be sorted ascending"):
        generate_walk_forward_splits(
            _hours(datetime(2024, 1, 1, tzinfo=UTC), 5)[::-1],
            train_window=1,
            validate_window=1,
            test_window=1,
            step_size=1,
        )

    with pytest.raises(ValueError):
        generate_walk_forward_splits(
            _hours(datetime(2024, 1, 1, tzinfo=UTC), 10),
            train_window=0,
            validate_window=1,
            test_window=1,
            step_size=1,
        )


def test_generate_walk_forward_splits_slice_id_is_deterministic() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 31)
    first = generate_walk_forward_splits(
        timestamps,
        train_window=5,
        validate_window=4,
        test_window=3,
        step_size=5,
    )
    second = generate_walk_forward_splits(
        timestamps,
        train_window=5,
        validate_window=4,
        test_window=3,
        step_size=5,
    )

    assert [split.slice_id for split in first] == [split.slice_id for split in second]


def test_walk_forward_split_to_bar_slices_with_slices() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 20)
    split = WalkForwardSplit(
        train=slice(1, 4),
        validate=slice(5, 8),
        test=slice(9, 11),
        lockbox=slice(13, 15),
        slice_id="time_window:manual",
        embargo_bars=1,
    )
    bars = split.to_bar_slices(timestamps)

    assert bars.train == split.train
    assert bars.validate == split.validate
    assert bars.test == split.test
    assert bars.lockbox == split.lockbox
    assert bars.slice_id == split.slice_id


def test_walk_forward_split_to_bar_slices_with_tuples() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 20)
    split = WalkForwardSplit(
        train=(timestamps[1], timestamps[4]),
        validate=(timestamps[4], timestamps[7]),
        test=(timestamps[9], timestamps[12]),
        lockbox=(timestamps[13], timestamps[17]),
        slice_id="time_window:0:1:4",
        embargo_bars=1,
    )
    bars = split.to_bar_slices(timestamps)

    assert bars.train == slice(1, 4)
    assert bars.validate == slice(4, 7)
    assert bars.test == slice(9, 12)
    assert bars.lockbox == slice(13, 17)


def test_walk_forward_split_to_bar_slices_clamps_out_of_range_timestamps() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    split = WalkForwardSplit(
        train=(timestamps[0] - timedelta(hours=2), timestamps[8] + timedelta(hours=2)),
        validate=(timestamps[2], timestamps[4]),
        test=(timestamps[5], timestamps[7]),
    )
    bars = split.to_bar_slices(timestamps)
    assert bars.train == slice(0, 10)
    assert bars.validate == slice(2, 4)
    assert bars.test == slice(5, 7)


def test_walk_forward_split_to_bar_slices_rejects_empty_index() -> None:
    split = WalkForwardSplit(train=slice(0, 1), validate=slice(1, 2), test=slice(2, 3))
    with pytest.raises(ValueError, match="index must be non-empty"):
        split.to_bar_slices([])


def test_walk_forward_split_to_bar_slices_rejects_empty_datetime_range() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    split = WalkForwardSplit(
        train=(timestamps[4], timestamps[4]),
        validate=(timestamps[1], timestamps[4]),
        test=(timestamps[4], timestamps[8]),
    )
    with pytest.raises(ValueError):
        split.to_bar_slices(timestamps)


def test_walk_forward_split_to_bar_slices_rejects_empty_bisected_range() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    split = WalkForwardSplit(
        train=(timestamps[1] + timedelta(minutes=30), timestamps[1] + timedelta(minutes=45)),
        validate=(timestamps[1], timestamps[2]),
        test=(timestamps[2], timestamps[4]),
    )
    with pytest.raises(ValueError, match="resolved boundary range is empty"):
        split.to_bar_slices(timestamps)


def test_walk_forward_split_to_bar_slices_rejects_naive_reference_timestamps() -> None:
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(10)]
    split = WalkForwardSplit(
        train=(datetime(2024, 1, 1, tzinfo=UTC), timestamps[4]),
        validate=(timestamps[1], timestamps[4]),
        test=(timestamps[4], timestamps[8]),
    )
    with pytest.raises(ValueError, match="timestamps must be timezone-aware"):
        split.to_bar_slices(timestamps)


def test_walk_forward_split_to_bar_slices_rejects_negative_indices() -> None:
    split = WalkForwardSplit(
        train=slice(-1, 1),
        validate=slice(1, 3),
        test=slice(3, 4),
    )
    with pytest.raises(ValueError, match="slice index must be non-negative"):
        split.to_bar_slices(_hours(datetime(2024, 1, 1, tzinfo=UTC), 10))


def test_walk_forward_split_rejects_invalid_split_boundaries() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    with pytest.raises(TypeError, match="boundary must be slice or tuple"):
        WalkForwardSplit(
            train=0,  # type: ignore[arg-type]
            validate=slice(1, 2),
            test=slice(2, 3),
        )

    with pytest.raises(ValueError, match="requires start/end datetimes"):
        WalkForwardSplit(
            train=(timestamps[0],),
            validate=slice(1, 2),
            test=slice(2, 3),
        )

    with pytest.raises(TypeError, match="requires start/end datetimes"):
        WalkForwardSplit(
            train=("x", timestamps[1]),
            validate=slice(1, 2),
            test=slice(2, 3),
        )

    with pytest.raises(ValueError, match="embargo_bars must be non-negative"):
        WalkForwardSplit(
            train=slice(0, 1),
            validate=slice(1, 2),
            test=slice(2, 3),
            embargo_bars=-1,
        )

    with pytest.raises(TypeError, match="lockbox boundary must be None or slice or tuple"):
        WalkForwardSplit(
            train=slice(0, 1),
            validate=slice(1, 2),
            test=slice(2, 3),
            lockbox=1,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="lockbox tuple boundary requires start/end datetimes"):
        WalkForwardSplit(
            train=slice(0, 1),
            validate=slice(1, 2),
            test=slice(2, 3),
            lockbox=(timestamps[0],),
        )

    with pytest.raises(TypeError, match="lockbox tuple boundary requires start/end datetimes"):
        WalkForwardSplit(
            train=slice(0, 1),
            validate=slice(1, 2),
            test=slice(2, 3),
            lockbox=("x", timestamps[1]),
        )


def test_to_slice_rejects_bad_tuple_boundary() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    with pytest.raises(ValueError, match="boundary tuple requires start/end datetimes"):
        _to_slice((timestamps[0], timestamps[1], timestamps[2]), timestamps, name="unit-test")


def test_generate_walk_forward_splits_rejects_step_and_lookahead_constraints() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 20)
    with pytest.raises(ValueError, match="step_size must be positive"):
        generate_walk_forward_splits(
            timestamps,
            train_window=1,
            validate_window=1,
            test_window=1,
            step_size=0,
        )

    with pytest.raises(ValueError, match="non-negative"):
        generate_walk_forward_splits(
            timestamps,
            train_window=1,
            validate_window=1,
            test_window=1,
            step_size=1,
            embargo_bars=-1,
        )

    with pytest.raises(ValueError, match="lookahead too large for train window"):
        generate_walk_forward_splits(
            timestamps,
            train_window=3,
            validate_window=2,
            test_window=2,
            step_size=2,
            label_lookahead_bars=3,
        )

    with pytest.raises(ValueError, match="lookahead too large for validate window"):
        generate_walk_forward_splits(
            timestamps,
            train_window=3,
            validate_window=2,
            test_window=2,
            step_size=2,
            label_lookahead_bars=2,
        )

    with pytest.raises(ValueError, match="lookahead too large for test window"):
        generate_walk_forward_splits(
            timestamps,
            train_window=3,
            validate_window=3,
            test_window=1,
            step_size=5,
            label_lookahead_bars=1,
        )


def test_generate_walk_forward_splits_can_break_on_step_exceeding_series() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 5)
    splits = generate_walk_forward_splits(
        timestamps,
        train_window=1,
        validate_window=1,
        test_window=1,
        step_size=5,
    )
    assert len(splits) == 1


def test_walk_forward_split_to_bar_slices_rejects_unsorted_index() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)[::-1]
    split = WalkForwardSplit(train=slice(0, 1), validate=slice(1, 2), test=slice(2, 3))
    with pytest.raises(ValueError, match="index must be sorted ascending"):
        split.to_bar_slices(timestamps)


def test_walk_forward_split_to_bar_slices_tz_mismatch() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    split = WalkForwardSplit(
        train=(datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=-4))), timestamps[4]),
        validate=(timestamps[1], timestamps[4]),
        test=(timestamps[4], timestamps[8]),
    )
    with pytest.raises(ValueError):
        split.to_bar_slices(timestamps)


def test_walk_forward_split_to_bar_slices_rejects_naive_datetimes() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    split = WalkForwardSplit(
        train=(datetime(2024, 1, 1), timestamps[4]),
        validate=(timestamps[1], timestamps[4]),
        test=(timestamps[4], timestamps[8]),
    )
    with pytest.raises(ValueError):
        split.to_bar_slices(timestamps)


def test_walk_forward_split_dev_alias() -> None:
    split = WalkForwardSplit(
        train=slice(0, 1),
        validate=slice(1, 2),
        test=slice(2, 3),
    )
    assert split.dev == split.validate


def test_generate_walk_forward_splits_with_embargo_creates_expected_gaps() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 25)
    splits = generate_walk_forward_splits(
        timestamps,
        train_window=4,
        validate_window=3,
        test_window=2,
        step_size=4,
        embargo_bars=2,
    )

    assert len(splits) == 4
    assert splits[0].train == slice(0, 4)
    assert splits[0].validate == slice(6, 9)
    assert splits[0].test == slice(11, 13)
    assert splits[1].train == slice(4, 8)
    assert splits[1].validate == slice(10, 13)
    assert splits[1].test == slice(15, 17)
    assert splits[2].train == slice(8, 12)
    assert splits[2].validate == slice(14, 17)
    assert splits[2].test == slice(19, 21)
    assert splits[3].train == slice(12, 16)
    assert splits[3].validate == slice(18, 21)
    assert splits[3].test == slice(23, 25)

    for split in splits:
        assert split.validate.start == split.train.stop + split.embargo_bars
        assert split.test.start == split.validate.stop + split.embargo_bars


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"step_size": 0}, "step_size must be positive"),
        ({"embargo_bars": -1}, "non-negative"),
        ({"label_lookahead_bars": 2}, "lookahead too large for train window"),
        (
            {"train_window": 3, "label_lookahead_bars": 2},
            "lookahead too large for validate window",
        ),
        (
            {"train_window": 3, "validate_window": 3, "test_window": 1, "label_lookahead_bars": 1},
            "lookahead too large for test window",
        ),
    ],
)
def test_generate_walk_forward_splits_error_paths(kwargs: dict[str, int], match: str) -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    params = dict(
        train_window=2,
        validate_window=2,
        test_window=2,
        step_size=2,
    )
    params.update(kwargs)
    with pytest.raises(ValueError, match=match):
        generate_walk_forward_splits(timestamps, **params)


def test_to_slice_uses_left_closed_right_open_endpoints() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    assert _to_slice((timestamps[1], timestamps[4]), timestamps, name="unit-test") == slice(1, 4)


def test_to_slice_right_boundary_excludes_matching_timestamp() -> None:
    timestamps = _hours(datetime(2024, 1, 1, tzinfo=UTC), 10)
    split = WalkForwardSplit(
        train=(timestamps[0], timestamps[4]),
        validate=(timestamps[4], timestamps[6]),
        test=(timestamps[6], timestamps[9]),
    )
    bars = split.to_bar_slices(timestamps)

    assert bars.train == slice(0, 4)
    assert bars.validate == slice(4, 6)
    assert bars.test == slice(6, 9)
