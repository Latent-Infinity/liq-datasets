from __future__ import annotations

from importlib import import_module

import pytest


@pytest.mark.xfail(
    strict=True,
    reason="mean-reversion anchor DTO contract lands with label implementation",
)
def test_mean_reversion_anchor_dto_contract_imports() -> None:
    module_exports = vars(import_module("liq.datasets.mean_reversion.labels.dto"))

    assert module_exports["MeanReversionAnchorDTO"] is not None


@pytest.mark.xfail(
    strict=True,
    reason="mean-reversion triple-barrier config lands with label implementation",
)
def test_triple_barrier_config_contract_imports() -> None:
    module_exports = vars(import_module("liq.datasets.mean_reversion.labels"))

    assert module_exports["TripleBarrierConfig"] is not None


@pytest.mark.xfail(
    strict=True,
    reason="mean-reversion triple-barrier label lands with label implementation",
)
def test_triple_barrier_label_contract_imports() -> None:
    module_exports = vars(import_module("liq.datasets.mean_reversion.labels"))

    assert module_exports["TripleBarrierLabel"] is not None


@pytest.mark.xfail(
    strict=True,
    reason="mean-reversion label builder lands with label implementation",
)
def test_build_labels_contract_imports() -> None:
    module_exports = vars(import_module("liq.datasets.mean_reversion.labels"))

    assert callable(module_exports["build_labels"])
