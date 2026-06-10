# Phase 0 Legacy Freeze

Frozen on: 2026-06-11

The archive was created before Phase 1 cleanup and contains the old source tree,
configuration, README, and research documents.

| Artifact | SHA-256 |
|---|---|
| `LLM4RL_phase0_legacy_20260611.zip` | `9B0516FC6F91643CEF7A6C1F6B4B715AEF7746E047B2D48C496B6DA47B9547D1` |
| `MAPS_original_latex_source.zip` | `5569A8B09D1168B7BBF797BD427DF709AE9C9B9DAD0F89E2ACBC25BAF88EFC3E` |
| `config_phase0.yaml` | `A8355139A37F4DEBE8A54D289462D0540201B1FAE4B93FA0B8231F0F28378529` |

Removed from the active tree after freezing:

- commented and non-runnable legacy experiment scripts;
- incomplete MAAC prototype not included in the Phase 1 baseline matrix;
- `utils/simulate_metrics.py`, which generated synthetic curves;
- Python bytecode caches and generated report HTML.

The archived results and the new model-version-2 results must never be mixed.
