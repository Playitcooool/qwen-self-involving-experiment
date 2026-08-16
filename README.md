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

## Domain-matched GSM8K run

The completed real-data run in `outputs/gsm8k_lopd/`, `outputs/gsm8k_domain/`, and `outputs/real_benchmark_gsm8k/` uses the official GSM8K `main/train` split (7,473 rows) for training. The split is leakage-safe: 64 problems are reserved for successful experience-bank collection, 40 disjoint problems are used for five on-policy update rounds, and 32 disjoint problems are validation-only. The existing 70-task held-out manifest—30 GSM8K, 10 HumanEval, and 30 ARC-Challenge tasks—is never loaded by the training scripts.

The run was executed with:

```bash
uv run python gsm8k_lopd.py \
  --model-path /Volumes/Samsung/lmstudio/lmstudio-community/Qwen:Qwen3.5-0.8B \
  --output outputs/gsm8k_lopd \
  --rounds 5 --bank-tasks 64 --tasks-per-round 8 \
  --validation-tasks 32 --bank-rollouts-per-task 3 \
  --max-new-tokens 256 --seed 20260816

uv run python gsm8k_baselines.py \
  --model-path /Volumes/Samsung/lmstudio/lmstudio-community/Qwen:Qwen3.5-0.8B \
  --data-manifest outputs/gsm8k_lopd/data_manifest.json \
  --output outputs/gsm8k_domain --rounds 5 \
  --tasks-per-round 8 --max-new-tokens 256 --seed 20260816
```

Only 12 of the 64 bank tasks produced successful base-model trajectories. The final student was evaluated without retrieval or latent context. On the same fixed held-out subset:

| Condition | Overall | GSM8K | HumanEval | ARC-Challenge |
|---|---:|---:|---:|---:|
| Base | 0.486 | 0.367 | 0.000 | 0.767 |
| Static skill | 0.243 | 0.100 | 0.100 | 0.433 |
| Domain SFT | 0.400 | 0.233 | 0.000 | 0.700 |
| Domain GRPO | 0.414 | 0.300 | 0.000 | 0.667 |
| Domain LOPD-lite | 0.400 | **0.333** | 0.000 | 0.600 |

The important domain-matched comparison is GSM8K: LOPD-lite beat matched GRPO (0.333 vs. 0.300) and SFT (0.333 vs. 0.233), but did not beat the frozen base model (0.333 vs. 0.367). The overall score is not a pure math score because HumanEval and ARC are transfer checks. This is evidence that the method can use real successful math trajectories, not evidence that LOPD-lite is always better; the small bank and falling on-policy reward make this a pilot result. The implementation remains the lightweight LOPD-lite variant, not an exact reproduction of the paper's unreleased full QFormer training code.

## Design

The benchmark has four exact-match task families: arithmetic, list filtering, string transformation, and constraint-following JSON. Each round uses fresh exploration tasks. Both improvement conditions see the same failed attempts, verifier feedback, and self-analysis prompt; only the storage mechanism differs.

## Reproducibility

- Same checkpoint, tokenizer, decoding settings, task batches, and verifiers across conditions.
- Hidden test tasks are generated from held-out seeds and are never used for updates.
- The training condition uses offline LoRA SFT; no gradients are computed during evaluation.
- The experiment records the model path, package lock, seed, hardware, and command line.
