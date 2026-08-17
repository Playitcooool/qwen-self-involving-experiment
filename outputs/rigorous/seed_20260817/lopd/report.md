# Domain-matched GSM8K LOPD-lite experiment

- Successful base-model experiences: **15**
- GSM8K update tasks: **40**
- GSM8K validation-only tasks: **32**
- Latent tokens: **8**; retrieved experiences: **2**
- Final student adapter: `outputs/rigorous/seed_20260817/lopd/adapters/round_5`
- Selected adapter: `None` (round **0**, validation accuracy **0.000**)
- Total training wall time: **4875.5s**

- Round 0: validation accuracy **0.000**
- Round 1: mean verifier reward **0.000**, validation accuracy **0.000**
- Round 2: mean verifier reward **0.250**, validation accuracy **0.000**
- Round 3: mean verifier reward **0.000**, validation accuracy **0.000**
- Round 4: mean verifier reward **0.000**, validation accuracy **0.000**
- Round 5: mean verifier reward **0.000**, validation accuracy **0.000**

The GSM8K train split was partitioned before training. The bank contains only successful model-generated trajectories, and the final student is evaluated without retrieval or latent context. Gold answers are used only inside the verifier.
