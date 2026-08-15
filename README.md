# Skill-based vs training-based self-involving improvement

This repository runs a reproducible comparison on `Qwen3.5-0.8B`:

1. **Baseline:** frozen model, no persistent memory.
2. **Skill-based:** frozen model plus externally stored, retrieved skill cards.
3. **Training-based:** offline LoRA updates from self-generated correction traces.

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

The multi-type held-out suite in `outputs/benchmark_suite/` covers word-problem math, rule reasoning, and executable Python functions. On this suite, baseline/SFT/GRPO each scored 0.333, while the external-skill condition scored 0.361. This small suite is diagnostic rather than definitive: math was unsolved by every condition and the code slice did not separate methods.

## Design

The benchmark has four exact-match task families: arithmetic, list filtering, string transformation, and constraint-following JSON. Each round uses fresh exploration tasks. Both improvement conditions see the same failed attempts, verifier feedback, and self-analysis prompt; only the storage mechanism differs.

## Reproducibility

- Same checkpoint, tokenizer, decoding settings, task batches, and verifiers across conditions.
- Hidden test tasks are generated from held-out seeds and are never used for updates.
- The training condition uses offline LoRA SFT; no gradients are computed during evaluation.
- The experiment records the model path, package lock, seed, hardware, and command line.
