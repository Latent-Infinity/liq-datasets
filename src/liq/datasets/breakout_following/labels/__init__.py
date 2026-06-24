"""Breakout-following label semantics.

The breakout-following workstream observes the same triple-barrier outcome
as ``liq.datasets.mean_reversion.labels`` and reinterprets the directions
inversely:

==================================  =========================  =======================
Triple-barrier outcome (observed)   Mean-reversion label        Breakout label
==================================  =========================  =======================
``reversion_first``                 positive (+1)              negative (-1)
``continuation_first``              negative (-1)              positive (+1)
``timeout_no_touch``                timeout (0)                timeout (0)
==================================  =========================  =======================

The pure-function :func:`flip_label_for_follow` performs this relabelling.
No new label-construction logic is introduced — only the sign-flip.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from liq.datasets.mean_reversion.labels import (
    BarrierTouched,
    LabelOutcome,
    TripleBarrierLabel,
)

# The mean-reversion -> breakout outcome remap.
_FOLLOW_OUTCOME_FROM_MR: dict[LabelOutcome, LabelOutcome] = {
    "positive": "negative",
    "negative": "positive",
    "timeout": "timeout",
}


class FollowLabel(BaseModel):
    """One breakout-following outcome linked to a source anchor.

    Mirrors :class:`liq.datasets.mean_reversion.labels.TripleBarrierLabel`'s
    shape so it can drop into the inherited CPCV + DSR + ablation harnesses
    unchanged. The ``outcome`` field carries the breakout-direction
    interpretation; ``barrier_touched`` carries the (direction-neutral)
    observation, identical to the source label.
    """

    model_config = ConfigDict(frozen=True)

    anchor_event_id: str
    outcome: LabelOutcome
    first_touch_index: int | None
    barrier_touched: BarrierTouched


def flip_label_for_follow(label: TripleBarrierLabel) -> FollowLabel:
    """Reinterpret a mean-reversion triple-barrier label for the breakout direction.

    Pure function — no anchor or fixture access. The triple-barrier
    *observation* (``barrier_touched`` + ``first_touch_index``) is preserved
    verbatim; only the ``outcome`` is remapped.
    """
    return FollowLabel(
        anchor_event_id=label.anchor_event_id,
        outcome=_FOLLOW_OUTCOME_FROM_MR[label.outcome],
        first_touch_index=label.first_touch_index,
        barrier_touched=label.barrier_touched,
    )


__all__ = ["FollowLabel", "flip_label_for_follow"]
