# Rigorous rerun: seed 20260817 interim results

These are checkpoint-selection results on the frozen 32-item GSM8K **training-validation** split. They are not results from the untouched 120-item test manifest.

| Condition | Selected candidate | Validation accuracy |
|---|---:|---:|
| Base | round 0 | 0/32 (0.000) |
| Gold SFT reference | epoch 1 | 10/32 (0.3125) |
| Verifier-reward GRPO | round 5 | 7/32 (0.21875) |
| LOPD-lite | round 0 / Base | 0/32 (0.000) |

The learned-skill condition produced 15 cards from verified-successful train-only rollouts, but it has not yet been evaluated on the untouched test set. The cards are retained for audit and are qualitatively repetitive and overly example-specific despite quantity redaction.

## Interpretation

- Gold SFT performed best, but it uses official GSM8K rationales and answers and is a supervised reference rather than a self-involving condition.
- GRPO improved from 0/32 at round 0 to 7/32 at round 5 without supervised answer-token targets.
- LOPD-lite did not improve validation accuracy; the preregistered earliest-tie rule therefore retained Base.
- Base scoring 0/32 under the strict final-line rule indicates that output-format compliance is a substantial part of this interim metric.
- PyTorch reported that an MPS backward kernel lacks a deterministic implementation. Seeds were fixed, but bitwise reproducibility on this hardware is not guaranteed.

No method comparison should be treated as final until the untouched test manifest is evaluated.
