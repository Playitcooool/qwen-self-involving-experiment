# Synthetic held-out comparison including LOPD-lite

This uses the exact 40-task synthetic held-out split from the prior experiment.

| Condition | Accuracy |
|---|---:|
| baseline | 0.150 |
| sft_round_5 | 0.325 |
| grpo_round_5 | 0.300 |
| skill_retry | 0.175 |
| lopd_lite | 0.250 |

LOPD-lite uses the same five-round synthetic training distribution as the earlier comparison, with a separate successful-rollout bank.
The verifier answer is used only to score the generated trajectory.
