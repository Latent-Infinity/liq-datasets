"""Breakout-following label semantics scaffold.

This module pre-registers the future label contracts:

* ``FollowLabel`` — frozen Pydantic with
  ``(anchor_event_id, outcome, first_touch_index, barrier_touched)`` where
  ``outcome ∈ {positive, negative, timeout}`` reflects the BREAKOUT trade
  direction.
* ``flip_label_for_follow(label) -> FollowLabel`` — pure function that
  takes a ``TripleBarrierLabel`` from the inherited builder and emits a
  ``FollowLabel`` with the inverted-direction interpretation.

Until those contracts are implemented, importing these names raises
``NotImplementedError`` so xfail tests track the planned surface explicitly.
"""

from __future__ import annotations


def __getattr__(name: str) -> object:
    if name in ("FollowLabel", "flip_label_for_follow"):
        msg = (
            f"liq.datasets.breakout_following.labels.{name} "
            "concrete contract is not implemented yet."
        )
        raise NotImplementedError(msg)
    raise AttributeError(
        f"module 'liq.datasets.breakout_following.labels' has no attribute {name!r}"
    )
