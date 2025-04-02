# Phase 6 — Final verification and cleanup (2026-11-08)

## Verification scope and provenance

This verification ran against a **reconstructed working copy** assembled from the previously supplied Phase 4 project archive, Phase 5 source update, Phase 2–4 analysis artifact archive, Phase 5 **separately supplied** prediction/log archive, and the Phase 6 analysis updates. It is not a remote audit of the user's Windows directory or Google Drive. Earlier "missing Phase 5 predictions/history" statements describe the *initial 154-file lightweight archive only*; these inputs were later provided in `phase6-required-analysis-files.zip` and are now present and verified.

## Component status

| Component | Final status and evidence |
|---|---|
| Phase 6 generated artifacts | All explicit `artifacts/phase_6/` references in the previously supplied Phase 6 Markdown documents resolve to existing files. Required tabular outputs and plots regenerated. |
| Phase 5 validation/test metrics | **16/16** recomputed from supplied per-message predictions and matched to saved metric JSON (8 validation + 8 test). |
| Phase 5 model alignment | Four architectures aligned exactly on game ID, message ID, absolute index and target for both splits and tasks. |
| BERT probability analysis | Distribution summaries, histograms, PR-based validation-only threshold selection and test application regenerated. Original default-threshold scores unchanged. |
| Training-history inspection | **8/8** `artifacts/phase_5/logs/*/*/epochs.jsonl` files available; loss and epoch curves derived without training. |
| Error and game analysis | Phase 5 FP/FN, disagreements versus hierarchical baseline, and per-game metrics regenerated from predictions; Phase 2–4 per-game outputs regenerated. |
| Phase 5 Power/BERT ablation | Macro F1, minority-class F1 and interaction differences regenerated using actual prediction metrics. |
| Historical benchmark comparison | Five Phase 2–4 selected models plus four Phase 5 variants have **identical held-out message-ID sets and targets within each task**. Metric definitions checked using saved predictions. |
| Original dataset | Exactly `train.jsonl`, `validation.jsonl`, `test.jsonl` under `data/`; byte-for-byte unchanged versus original Phase 4 archive; SHA-256 hashes saved. |
| Phase 4 code cleanup | Removed only the second, byte-identical `target_preserving_text()` definition; all context tests pass. |
| Repository export hygiene | Cache directories and compiled files removed from the **assembled working tree and update export**; `.gitignore` covers caches. Existing research artifacts preserved. |
| CPU automated tests | **71/71 passed** in assembled working copy (Python environment supplied with CPU PyTorch). |
| Model inference parity | Previously reported by the user for eight Phase 5 trained models. **Not rerun in this verification**, since the complete pretrained checkpoints were intentionally not part of the small analysis export. |
| GPU training | **Not performed** during Phase 6 final verification. Prior training was performed by the user in Colab. |

## Observed Phase 5 test results

| Model | Actual-lie Macro F1 | Actual lie F1 | Suspected-lie Macro F1 | Suspected lie F1 |
|---|---:|---:|---:|---:|
| Hierarchical LSTM | 0.5263 | 0.1094 | 0.5093 | 0.0698 |
| + Power | **0.5795** | **0.2406** | **0.5336** | **0.1326** |
| + BERT | 0.4771 | 0.0000 | 0.4828 | 0.0000 |
| + BERT + Power | 0.4771 | 0.0000 | 0.4828 | 0.0000 |

**Interpretation:** Power improves *observed test performance* in this pair of independently trained, non-BERT hierarchical models. The BERT variants predict only the negative class at the original threshold. Validation-selected thresholds are reported as separate exploratory analyses. Neither feature effects nor threshold tuning is supported as statistically significant by a two-game test split. Causes of BERT behavior remain hypotheses, not established bugs.

## Remaining gaps and caveats

- Phase 5 prediction export does not contain original message text, sender/receiver names, season or message-level score differences. Rich linguistic error examples and direct power-feature sensitivity therefore require a separately supplied, carefully reviewed data join or checkpoint-level experiment.
- BERT gradient flow and internal representation diagnostics were **not executed**; scalar epoch histories cannot prove a particular failure mechanism.
- Full BERT checkpoint weights were **not included** in the small Phase 6 analysis export. This final pass does not independently rerun model inference or GPU training.
- Direct original-paper reproduction claims remain conditional on architecture and preprocessing equivalence, which has not been established completely.
- Only two test games; per-message counts are not independent samples. No statistical significance claim is made.

## Reproducing from the assembled current project

```powershell
$env:PYTHONPATH = "src"
python -m diplomacy.analysis.phase6_analysis --root . --output artifacts/phase_6
python -m diplomacy.analysis.phase6_phase5_deep_dive --root . --output artifacts/phase_6
python -m pytest tests/ -q
```

These commands need the source, YAML configurations, original three JSONL files, Phase 2–4 prediction artifacts and all Phase 5 prediction CSVs, metric JSONs, epoch logs and training summaries. They **do not** require saved model weights. The analysis is reproducible from the assembled working copy with those inputs, but not from the documentation-only ZIP or source-only GitHub checkout (where `artifacts/` is ignored).

Full structured verification: `artifacts/phase_6/audit/final_verification.json`. Historical population alignment: `artifacts/phase_6/benchmarking/cross_phase_message_alignment.csv`.
