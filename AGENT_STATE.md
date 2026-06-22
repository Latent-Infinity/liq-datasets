# AGENT_STATE.md — liq-datasets

State tracked per implementation plan that touches this repo.

## Plan: `mean-reversion-swing-scan-impl-plan` (v0.3)

| Work package | Capability | Status | Evidence |
| --- | --- | --- | --- |
| F0 | Mean-reversion labels namespace under construction; DTO/config/label/builder import contracts pinned by strict xfail tests; `pyarrow` and `pydantic` dependencies declared for future parquet schema and DTO work | ready-for-review | `src/liq/datasets/mean_reversion/labels/__init__.py`; `tests/test_mean_reversion_label_contract_imports.py`; `pyproject.toml` |
