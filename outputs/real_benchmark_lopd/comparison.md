# Real benchmark comparison including LOPD-lite

Same fixed subset as `outputs/real_benchmark/manifest.json`; this is not an official full-set score.

| Condition | Overall | GSM8K | HumanEval | ARC-Challenge |
|---|---:|---:|---:|---:|
| baseline | 0.486 | 0.367 | 0.000 | 0.767 |
| sft | 0.429 | 0.267 | 0.000 | 0.733 |
| grpo | 0.457 | 0.333 | 0.000 | 0.733 |
| skill | 0.243 | 0.100 | 0.100 | 0.433 |
| lopd_lite | 0.343 | 0.267 | 0.000 | 0.533 |

LOPD-lite was trained on the earlier synthetic training distribution; therefore this is a transfer evaluation, not a domain-matched reproduction of the paper's EnvScaler/TACO setup.
Evaluation wall time: 567.5s.
Gold answers are used only by the existing benchmark verifiers.
