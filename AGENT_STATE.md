# AGENT_STATE.md — liq-datasets

State tracked per implementation plan that touches this repo.

## Plan: `mean-reversion-swing-scan-impl-plan` (v0.3.1)

| Work package | Capability | Status | Evidence |
| --- | --- | --- | --- |
| F0 | Mean-reversion labels namespace under construction; DTO/config/label/builder import contracts pinned by strict xfail tests; `pyarrow` and `pydantic` dependencies declared for future parquet schema and DTO work | ready-for-review | `src/liq/datasets/mean_reversion/labels/__init__.py`; `tests/test_mean_reversion_label_contract_imports.py`; `pyproject.toml` |
| F3 | Mean-reversion anchor DTO mirror; triple-barrier config and label contracts; first-touch range-unit label builder from persisted anchor and bar parquet; schema-parity and import-boundary gates; F0 xfails flipped | ready-for-review | `src/liq/datasets/mean_reversion/labels/`; `tests/test_mean_reversion_labels.py`; `tests/test_mean_reversion_label_contract_imports.py` |
| F3H | Combinatorial purged cross-validation path generation; union embargo width `max(L_vol, L_base, H)`; runtime embargo assertion; deterministic path IDs; real-fixture label embargo gate; vulture clean | ready-for-review | `src/liq/datasets/mean_reversion/labels/{cpcv,assertions}.py`; `tests/test_mean_reversion_cpcv.py` |
