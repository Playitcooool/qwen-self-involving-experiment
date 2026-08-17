# Preregistered rigorous rerun protocol

Status: frozen before any model is evaluated on the new test manifest.

## Research question

For the local `Qwen3.5-0.8B` checkpoint, compare inference-time learned skill
retrieval with offline parameter updates. This is not test-time training: every
artifact is frozen before test prompts are loaded.

## Conditions

1. `base`: unchanged local checkpoint.
2. `learned_skill`: top-3 procedure cards retrieved from cards distilled from
   verified-successful base-model GSM8K *training* rollouts. Cards contain no
   benchmark reference answer and parameters are unchanged.
3. `gold_sft_reference`: LoRA trained on official GSM8K rationales and answers.
   This is supervised and is not described as self-involving.
4. `grpo_verifier`: LoRA trained only from group-relative exact-match rewards.
5. `lopd_lite`: reward-gated latent on-policy self-distillation, student-only
   at inference.

## Data roles and selection

- GSM8K train rows are partitioned once into bank, update, and validation roles.
- Checkpoint candidates include round/epoch 0 (Base) for every trained method.
- All methods use the same validation rows. Highest GSM8K exact match wins;
  ties select the earliest checkpoint, including round 0.
- The prior 70-item suite is development data and is excluded from the new test.
- New untouched test: 60 GSM8K, 40 ARC-Challenge, and 20 HumanEval items.
- GSM8K exact match is the sole primary endpoint. ARC and HumanEval are
  separately reported secondary transfer endpoints; no micro-averaged overall
  score is used as a primary claim.

## Repetition and inference

- Training seeds: 20260817, 20260818, 20260819.
- Deterministic greedy test decoding and identical generation limits per task.
- Report item-level paired outcomes, exact McNemar tests, Wilson intervals, and
  training-seed mean and standard deviation. No superiority claim is made when
  uncertainty includes no improvement.

## Leakage and reproducibility controls

- Manifests store source row indices and content hashes, not local sequence IDs.
- Cached source files and model files are SHA-256 fingerprinted.
- Package versions, git revision, commands, seeds, and device are recorded.
- The test manifest is generated and committed before any test inference.
- Test answers may be read only by the verifier after a prediction is complete.
- HumanEval execution is a convenience signal and is not a secure sandbox;
  only trusted public benchmark tests are executed.

