# AGENT_STATE.md — liq-datasets

State tracked per implementation plan that touches this repo.

## Plan: `mean-reversion-swing-scan-impl-plan` (v0.3.1)

| Work package | Capability | Status | Evidence |
| --- | --- | --- | --- |
| F0 | Mean-reversion labels namespace under construction; DTO/config/label/builder import contracts pinned by strict xfail tests; `pyarrow` and `pydantic` dependencies declared for future parquet schema and DTO work | ready-for-review | `src/liq/datasets/mean_reversion/labels/__init__.py`; `tests/test_mean_reversion_label_contract_imports.py`; `pyproject.toml` |
| F3 | Mean-reversion anchor DTO mirror; triple-barrier config and label contracts; first-touch range-unit label builder from persisted anchor and bar parquet; schema-parity and import-boundary gates; F0 xfails flipped | ready-for-review | `src/liq/datasets/mean_reversion/labels/`; `tests/test_mean_reversion_labels.py`; `tests/test_mean_reversion_label_contract_imports.py` |
| F3H | Combinatorial purged cross-validation path generation; union embargo width `max(L_vol, L_base, H)`; runtime embargo assertion; deterministic path IDs; real-fixture label embargo gate; vulture clean | ready-for-review | `src/liq/datasets/mean_reversion/labels/{cpcv,assertions}.py`; `tests/test_mean_reversion_cpcv.py` |

## Plan: `breakout-following-strategy-impl-plan` (v0.2.8)

| Work package | Capability | Status | Evidence |
| --- | --- | --- | --- |
| B0 | Breakout-following labels namespace scaffold landed; unknown attributes keep the module-level `AttributeError` refusal contract. | ready-for-review | `src/liq/datasets/breakout_following/labels/__init__.py`; `tests/test_breakout_following_stubs.py` |
| B2 | `FollowLabel` and `flip_label_for_follow` are implemented by inverting inherited triple-barrier labels into the breakout-following direction. Truth-table and import-contract tests cover continuation/reversion/timeout values, string coercion, unknown-label rejection, and inherited import boundaries; targeted label coverage is 100 %. | ready-for-review | `src/liq/datasets/breakout_following/labels/__init__.py`; `tests/test_breakout_following_labels.py`; `tests/test_breakout_following_stubs.py` |
