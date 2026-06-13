# Phase 0 Legacy Freeze

Frozen on: 2026-06-11

The archive was created before Phase 1 cleanup and contains the old source tree,
configuration, README, and research documents.

| Artifact | SHA-256 |
|---|---|
| `LLM4RL_phase0_legacy_20260611.zip` | `C37468CFFBF5ACB93B27F64581DB5DF97985E878C7A03AED58C17C3BA9D2A9F1` |
| `MAPS_original_latex_source.zip` | `5569A8B09D1168B7BBF797BD427DF709AE9C9B9DAD0F89E2ACBC25BAF88EFC3E` |
| `config_phase0.yaml` | `21E9D6DA3CBF30A57D64721B0D56BEBC8846693715DE10E303D9EE3DA509F73A` |

Security maintenance on 2026-06-13 replaced plaintext legacy API credentials
with redacted markers in the standalone configuration and the archived
configuration/client copies. No experiment logic was changed. The checksums
above refer to the sanitized archive.

Removed from the active tree after freezing:

- commented and non-runnable legacy experiment scripts;
- incomplete MAAC prototype not included in the Phase 1 baseline matrix;
- `utils/simulate_metrics.py`, which generated synthetic curves;
- Python bytecode caches and generated report HTML.

The archived results and the new model-version-2 results must never be mixed.
