# Real benchmark evaluation

This run evaluates fixed random subsets of public test sets: 30 GSM8K problems, 10 HumanEval functions, and 30 ARC-Challenge questions. It is a subset evaluation, not an official full-benchmark score.

| Condition | Overall | GSM8K | HumanEval | ARC-Challenge |
|---|---:|---:|---:|---:|
| Baseline | 0.486 | 0.367 | 0.000 | 0.767 |
| SFT | 0.429 | 0.267 | 0.000 | 0.733 |
| GRPO | 0.457 | 0.333 | 0.000 | 0.733 |
| External skill | 0.243 | 0.100 | 0.100 | 0.433 |

## Interpretation

On this real-data transfer evaluation, the frozen base model was strongest overall. The SFT and GRPO adapters were trained on the earlier toy task distribution, so their lower scores indicate negative cross-domain transfer rather than that SFT or GRPO are intrinsically harmful. GRPO retained more GSM8K performance than SFT. The external skill prompt substantially hurt ARC and GSM8K, although it solved one HumanEval task; the generic skill instructions were not a reliable substitute for task-specific prompting.

The generation budget was 256 tokens for GSM8K, 384 for HumanEval, and 64 for ARC. HumanEval candidates were executed against the dataset-provided tests. Gold answers were used only inside verifiers, never as generated training targets.
