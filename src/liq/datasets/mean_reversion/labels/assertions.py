"""Runtime assertions for mean-reversion label validation."""

from __future__ import annotations

from collections.abc import Sequence

from liq.datasets.mean_reversion.labels import TripleBarrierLabel
from liq.datasets.mean_reversion.labels.cpcv import CPCVPath, embargo_width


class EmbargoViolation(RuntimeError):
    """Raised when a CPCV path leaks train labels into the embargo window."""


def assert_cpcv_embargo(
    paths: Sequence[CPCVPath],
    labels: Sequence[TripleBarrierLabel],
    *,
    L_vol: int,
    L_base: int,
    H: int,
) -> None:
    """Assert every train index is outside the union embargo around test indices."""
    width = embargo_width(L_vol, L_base, H)
    label_count = len(labels)
    for path in paths:
        for index in (*path.train_indices, *path.test_indices):
            if index < 0 or index >= label_count:
                raise EmbargoViolation(f"{path.path_id}: label index {index} outside label range")
        for train_index in path.train_indices:
            for test_index in path.test_indices:
                if abs(train_index - test_index) <= width:
                    raise EmbargoViolation(
                        f"{path.path_id}: train index {train_index} within {width} bars of test index {test_index}"
                    )


__all__ = ["EmbargoViolation", "assert_cpcv_embargo"]
