from __future__ import annotations

from importlib import import_module


def test_mean_reversion_anchor_dto_contract_imports() -> None:
    module_exports = vars(import_module("liq.datasets.mean_reversion.labels.dto"))

    assert module_exports["MeanReversionAnchorDTO"] is not None


def test_triple_barrier_config_contract_imports() -> None:
    module_exports = vars(import_module("liq.datasets.mean_reversion.labels"))

    assert module_exports["TripleBarrierConfig"] is not None


def test_triple_barrier_label_contract_imports() -> None:
    module_exports = vars(import_module("liq.datasets.mean_reversion.labels"))

    assert module_exports["TripleBarrierLabel"] is not None


def test_build_labels_contract_imports() -> None:
    module_exports = vars(import_module("liq.datasets.mean_reversion.labels"))

    assert callable(module_exports["build_labels"])
