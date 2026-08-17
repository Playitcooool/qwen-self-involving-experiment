# Domain-matched GSM8K baselines

- Update tasks: **40**
- SFT uses the official GSM8K rationale followed by the final numeric answer as the target.
- GRPO uses group-relative verifier rewards, with a small sampled reference-retention penalty.

- SFT adapter: `outputs/gsm8k_domain_fixed_g4/sft/adapter`
- GRPO adapter: `outputs/gsm8k_domain_fixed_g4/grpo/adapter`
