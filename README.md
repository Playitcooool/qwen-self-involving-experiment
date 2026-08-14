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

The completed run reached 0.150 baseline accuracy, 0.325 training-based accuracy at round 5 (0.350 at round 4), and 0.000 for the tested skill-card interface. See `outputs/report.md` for the caveat that the skill condition copied injected procedures instead of applying them reliably.

## Design

The benchmark has four exact-match task families: arithmetic, list filtering, string transformation, and constraint-following JSON. Each round uses fresh exploration tasks. Both improvement conditions see the same failed attempts, verifier feedback, and self-analysis prompt; only the storage mechanism differs.

## Reproducibility

- Same checkpoint, tokenizer, decoding settings, task batches, and verifiers across conditions.
- Hidden test tasks are generated from held-out seeds and are never used for updates.
- The training condition uses offline LoRA SFT; no gradients are computed during evaluation.
- The experiment records the model path, package lock, seed, hardware, and command line.
