"""Tests for ``liq.datasets.breakout_following.labels``.

The breakout-following workstream observes the same triple-barrier outcome
as mean-reversion but interprets the directions inversely:

* mean-reversion `reversion` → breakout `negative` (loss)
* mean-reversion `continuation` → breakout `positive` (win)
* mean-reversion `timeout` → breakout `timeout` (flat)

This is a pure relabelling — no new label construction logic. The
``FollowLabel`` model mirrors ``TripleBarrierLabel``'s shape and carries
the inverted ``outcome`` field.
"""

from __future__ import annotations

import pytest

from liq.datasets.breakout_following.labels import FollowLabel, flip_label_for_follow
from liq.datasets.mean_reversion.labels import TripleBarrierLabel

# ----- FollowLabel shape ----- #


def test_follow_label_is_frozen_pydantic() -> None:
    label = FollowLabel(
        anchor_event_id="evt-1",
        outcome="positive",
        first_touch_index=42,
        barrier_touched="continuation",
    )
    assert label.anchor_event_id == "evt-1"
    assert label.outcome == "positive"
    assert label.first_touch_index == 42
    assert label.barrier_touched == "continuation"
    # frozen → mutation refused
    with pytest.raises((TypeError, ValueError)):
        label.outcome = "negative"  # type: ignore[misc]


@pytest.mark.parametrize("outcome", ["positive", "negative", "timeout"])
def test_follow_label_outcome_accepts_canonical_values(outcome: str) -> None:
    label = FollowLabel(
        anchor_event_id="evt",
        outcome=outcome,  # type: ignore[arg-type]
        first_touch_index=None,
        barrier_touched="timeout",
    )
    assert label.outcome == outcome


def test_follow_label_rejects_unknown_outcome() -> None:
    with pytest.raises(ValueError):
        FollowLabel(
            anchor_event_id="evt",
            outcome="something_else",  # type: ignore[arg-type]
            first_touch_index=None,
            barrier_touched="timeout",
        )


# ----- flip_label_for_follow truth table ----- #


@pytest.mark.parametrize(
    ("reversion_outcome", "barrier_touched", "expected_follow_outcome"),
    [
        ("positive", "reversion", "negative"),
        ("negative", "continuation", "positive"),
        ("timeout", "timeout", "timeout"),
    ],
)
def test_flip_label_for_follow_inverts_canonical_outcomes(
    reversion_outcome: str,
    barrier_touched: str,
    expected_follow_outcome: str,
) -> None:
    """The canonical 3-row truth table."""
    mr_label = TripleBarrierLabel(
        anchor_event_id="evt-truth-table",
        outcome=reversion_outcome,  # type: ignore[arg-type]
        first_touch_index=7,
        barrier_touched=barrier_touched,  # type: ignore[arg-type]
    )
    follow = flip_label_for_follow(mr_label)
    assert isinstance(follow, FollowLabel)
    assert follow.outcome == expected_follow_outcome


def test_flip_preserves_anchor_event_id() -> None:
    mr_label = TripleBarrierLabel(
        anchor_event_id="evt-preserve",
        outcome="positive",
        first_touch_index=3,
        barrier_touched="reversion",
    )
    follow = flip_label_for_follow(mr_label)
    assert follow.anchor_event_id == "evt-preserve"


def test_flip_preserves_first_touch_index_and_barrier_touched() -> None:
    """The barrier-touched + first-touch fields are observation, not interpretation;
    they survive the flip unchanged."""
    mr_label = TripleBarrierLabel(
        anchor_event_id="evt",
        outcome="negative",
        first_touch_index=11,
        barrier_touched="continuation",
    )
    follow = flip_label_for_follow(mr_label)
    assert follow.first_touch_index == 11
    assert follow.barrier_touched == "continuation"


def test_flip_handles_timeout_first_touch_index_none() -> None:
    mr_label = TripleBarrierLabel(
        anchor_event_id="evt",
        outcome="timeout",
        first_touch_index=None,
        barrier_touched="timeout",
    )
    follow = flip_label_for_follow(mr_label)
    assert follow.first_touch_index is None
    assert follow.barrier_touched == "timeout"
    assert follow.outcome == "timeout"


def test_flip_is_idempotent_through_round_trip_on_outcome_field() -> None:
    """Applying flip twice returns to the original outcome (positive ↔ negative)."""
    mr_label = TripleBarrierLabel(
        anchor_event_id="evt",
        outcome="positive",
        first_touch_index=5,
        barrier_touched="reversion",
    )
    flipped_once = flip_label_for_follow(mr_label)
    # round-trip: reconstruct a TripleBarrierLabel with the flipped outcome
    mr_again = TripleBarrierLabel(
        anchor_event_id=flipped_once.anchor_event_id,
        outcome=flipped_once.outcome,
        first_touch_index=flipped_once.first_touch_index,
        barrier_touched=flipped_once.barrier_touched,
    )
    flipped_twice = flip_label_for_follow(mr_again)
    assert flipped_twice.outcome == mr_label.outcome
