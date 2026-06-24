"""Xfail stubs for breakout-following label contracts.

Each xfail row in the ContractsFreeze manifest has a corresponding xfail
test here.
"""

from __future__ import annotations

import importlib

import pytest


def test_namespace_imports() -> None:
    """The namespace itself must be importable."""
    module = importlib.import_module("liq.datasets.breakout_following.labels")
    assert module is not None


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason="FollowLabel concrete Pydantic contract is not implemented yet",
    strict=True,
)
def test_follow_label_concrete_class_resolves() -> None:
    from liq.datasets.breakout_following.labels import FollowLabel  # noqa: F401


@pytest.mark.xfail(
    raises=NotImplementedError,
    reason="flip_label_for_follow concrete function is not implemented yet",
    strict=True,
)
def test_flip_label_for_follow_function_resolves() -> None:
    from liq.datasets.breakout_following.labels import flip_label_for_follow  # noqa: F401


def test_unknown_attribute_raises_attribute_error() -> None:
    """Non-stub attribute access must raise AttributeError (not NotImplementedError)."""
    from liq.datasets.breakout_following import labels

    with pytest.raises(AttributeError):
        _ = labels.SomeUnrelatedAttribute  # type: ignore[attr-defined]
