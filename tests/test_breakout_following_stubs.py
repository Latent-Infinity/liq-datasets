"""Contract-import + refusal sanity tests for breakout-following labels.

These tests pin the namespace surface promised by ContractsFreeze v1 and
the `AttributeError`-vs-`NotImplementedError` refusal contract. Concrete
surfaces are asserted as passing imports here.
"""

from __future__ import annotations

import importlib

import pytest


def test_namespace_imports() -> None:
    """The namespace itself must be importable."""
    module = importlib.import_module("liq.datasets.breakout_following.labels")
    assert module is not None


def test_follow_label_concrete_class_resolves() -> None:
    """``FollowLabel`` is a concrete Pydantic model (ContractsFreeze row 1)."""
    from liq.datasets.breakout_following.labels import FollowLabel

    assert FollowLabel is not None
    # frozen Pydantic instances have a ``model_config`` attribute
    assert hasattr(FollowLabel, "model_config")


def test_flip_label_for_follow_function_resolves() -> None:
    """``flip_label_for_follow`` is callable (ContractsFreeze row 1)."""
    from liq.datasets.breakout_following.labels import flip_label_for_follow

    assert callable(flip_label_for_follow)


def test_unknown_attribute_raises_attribute_error() -> None:
    """Non-stub attribute access must raise ``AttributeError``."""
    from liq.datasets.breakout_following import labels

    with pytest.raises(AttributeError):
        _ = labels.SomeUnrelatedAttribute  # type: ignore[attr-defined]
