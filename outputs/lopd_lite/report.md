# LOPD-lite experiment

- Successful base-model experiences: **14**
- Latent tokens: **8**; retrieved experiences: **2**
- Final student adapter: `outputs/lopd_lite/adapters/round_5`
- Total training wall time: **583.6s**

- Round 1: mean verifier reward **0.000**
- Round 2: mean verifier reward **0.125**
- Round 3: mean verifier reward **0.125**
- Round 4: mean verifier reward **0.375**
- Round 5: mean verifier reward **0.125**

The base model is the frozen privileged teacher; only the student LoRA and latent composer are optimized. The verifier answer is never inserted into the student prompt or teacher context as a label.

## Evaluation

| Split | LOPD-lite | Baseline | SFT | GRPO | Skill retry |
|---|---:|---:|---:|---:|---:|
| Synthetic held-out (40) | 0.250 | 0.150 | 0.325 | 0.300 | 0.175 |
| Real subset (70) | 0.343 | 0.486 | 0.429 | 0.457 | 0.243 |

The real subset is a transfer evaluation: this run trained on the earlier synthetic distribution, not on GSM8K, HumanEval, or ARC training data.
