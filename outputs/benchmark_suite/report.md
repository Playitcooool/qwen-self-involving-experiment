# Multi-type benchmark

The held-out suite contains 36 tasks: 12 word-problem math, 12 multi-step rule reasoning, and 12 executable Python-function tasks.

| Condition | Overall | Math | Logic | Code |
|---|---:|---:|---:|---:|
| Baseline | 0.333 | 0.000 | 0.750 | 0.250 |
| SFT | 0.333 | 0.000 | 0.750 | 0.250 |
| GRPO | 0.333 | 0.000 | 0.750 | 0.250 |
| External skill | 0.361 | 0.000 | 0.833 | 0.250 |

The external skill condition was best overall because it solved one additional logic task. SFT and GRPO showed no measurable transfer on this particular held-out suite. Math was too difficult for all conditions, while the small code slice did not distinguish the methods; larger, established benchmarks such as GSM8K and HumanEval should be the next validation step.
