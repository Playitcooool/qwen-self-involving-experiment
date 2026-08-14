# Experiment report

## Protocol

The local `Qwen3.5-0.8B` checkpoint was evaluated on 40 hidden exact-answer tasks per round: 10 arithmetic, 10 filtering, 10 string transformation, and 10 constrained-JSON tasks. Five offline self-involving rounds were run. The baseline is frozen. The skill condition stores self-generated procedures externally and prepends retrieved family-specific cards. The training condition uses the same failed attempts and verifier answers for offline LoRA SFT, then freezes the adapter during evaluation. No evaluation-time gradients or test-time training were used.

## Learning curve

| Round | Baseline | Skill-based | Training-based |
|---:|---:|---:|---:|
| 0 | 0.150 | — | — |
| 1 | — | 0.000 | 0.300 |
| 2 | — | 0.000 | 0.275 |
| 3 | — | 0.000 | 0.325 |
| 4 | — | 0.000 | 0.350 |
| 5 | — | 0.000 | 0.325 |

## Interpretation

In this implementation, offline training improved exact-match performance from 0.150 to 0.325 at round 5, peaking at 0.350 in round 4. The external-skill condition scored 0.000 throughout because the 0.8B model copied or expanded the injected skill text rather than reliably applying it; this is a negative result for this skill-card interface, not evidence that all external skill-memory designs fail.

The training gain was concentrated in constrained JSON and arithmetic. Filtering and string transformation remained unsolved. Because the benchmark is small and synthetic, these results should be treated as an implementation-level comparison and not as a general capability claim.

This report is generated from 440 prediction records and five saved LoRA adapters. The complete per-example data are in `predictions.jsonl`.
