# Domain-matched GSM8K comparison

All conditions use the same fixed held-out manifest. Training uses only GSM8K train rows; HumanEval and ARC are transfer checks.

| Condition | Overall | GSM8K | HumanEval | ARC-Challenge |
|---|---:|---:|---:|---:|
| baseline | 0.486 | 0.367 | 0.000 | 0.767 |
| skill | 0.243 | 0.100 | 0.100 | 0.433 |
| sft_gsm8k | 0.429 | 0.267 | 0.000 | 0.733 |
| grpo_gsm8k | 0.443 | 0.333 | 0.000 | 0.700 |
| lopd_lite_gsm8k | 0.443 | 0.333 | 0.000 | 0.700 |

LOPD-lite is evaluated student-only: no experience retrieval or latent context is available during evaluation.
Evaluation wall time: 2882.3s.
Gold answers are used only inside benchmark verifiers.
