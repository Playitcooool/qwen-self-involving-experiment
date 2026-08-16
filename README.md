# Skill-based vs training-based self-involving improvement

This repository runs a reproducible comparison on `Qwen3.5-0.8B`:

1. **Baseline:** frozen model, no persistent memory.
2. **Skill-based:** frozen model plus externally stored, retrieved skill cards.
3. **Training-based:** offline LoRA updates from self-generated correction traces.
4. **LOPD-lite:** reward-weighted latent on-policy self-distillation with a frozen privileged teacher.

This is explicitly **not test-time training (TTT)**. All LoRA updates happen in discrete offline rounds, and the resulting adapter is frozen during evaluation.

## Run

```bash
uv sync
uv run python experiment.py \
  --model-path /Volumes/Samsung/lmstudio/lmstudio-community/Qwen:Qwen3.5-0.8B \
  --rounds 5 \
  --train-per-family 4 \
  --test-per-family 10 \
  --seed 42
```

Results are written to `outputs/`, including JSONL predictions, CSV metrics, and `report.md`.

The completed run reached 0.150 baseline accuracy and 0.325 training-based accuracy at round 5 (0.350 at round 4). The original skill-card interface scored 0.000; a deduplicated silent-card retry scored 0.175 in every round. See `outputs/report.md` for details.

A separate reward-only GRPO-style run is also included. It reached 0.300 at round 5 from the same 0.150 baseline, using group-relative verifier rewards rather than answer-token SFT labels. Its artifacts are in `outputs/grpo/`.

The LOPD-lite condition is implemented in `lopd_lite.py`. It builds a frozen bank of successful base-model rollouts, retrieves similar experiences, compresses them into learned latent tokens, and trains a LoRA student to match a frozen privileged teacher on the student's own rollout prefixes. The verifier reward is used for the privileged-margin signal; the student is evaluated without the latent context. The implementation is intentionally smaller than the paper's full QFormer system so it fits the local 0.8B model:

```bash
uv run python lopd_lite.py \
  --model-path /Volumes/Samsung/lmstudio/lmstudio-community/Qwen:Qwen3.5-0.8B \
  --rounds 5 --train-per-family 2 --bank-per-family 12 \
  --bank-rollouts-per-task 3 --max-new-tokens 64 \
  --output outputs/lopd_lite
```

On the exact prior synthetic held-out split, LOPD-lite scored 0.250 (baseline 0.150, SFT 0.325, GRPO 0.300, corrected skill retry 0.175). On the same fixed real-data subset, it scored 0.343 overall (GSM8K 0.267, HumanEval 0.000, ARC-Challenge 0.533). See `outputs/lopd_lite/synthetic_eval/comparison.md` and `outputs/real_benchmark_lopd/comparison.md`.

The multi-type held-out suite in `outputs/benchmark_suite/` covers word-problem math, rule reasoning, and executable Python functions. On this suite, baseline/SFT/GRPO each scored 0.333, while the external-skill condition scored 0.361. This small suite is diagnostic rather than definitive: math was unsolved by every condition and the code slice did not separate methods.

The real-data subset evaluation in `outputs/real_benchmark/` uses public [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval), and [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) test data. Results are reported as a fixed subset evaluation, not official full-set scores: baseline 0.486, SFT 0.429, GRPO 0.457, external skill 0.243, and LOPD-lite 0.343 overall. LOPD-lite, SFT, and GRPO were trained on the earlier synthetic distribution, so these are transfer results rather than domain-matched training results.

## Design

The benchmark has four exact-match task families: arithmetic, list filtering, string transformation, and constraint-following JSON. Each round uses fresh exploration tasks. Both improvement conditions see the same failed attempts, verifier feedback, and self-analysis prompt; only the storage mechanism differs.

## Reproducibility

- Same checkpoint, tokenizer, decoding settings, task batches, and verifiers across conditions.
- Hidden test tasks are generated from held-out seeds and are never used for updates.
- The training condition uses offline LoRA SFT; no gradients are computed during evaluation.
- The experiment records the model path, package lock, seed, hardware, and command line.
