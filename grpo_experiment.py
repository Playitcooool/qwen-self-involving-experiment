"""Minimal verifier-reward GRPO-style experiment.

Unlike the saved SFT condition, this script never supplies the correct answer
as a training target. It samples a group of completions, scores each completion
with the exact-match verifier, and updates LoRA parameters using within-group
relative rewards.
"""
from __future__ import annotations

import argparse, csv, gc, json, random, time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment import dataset, normalize, evaluate, Runner


def device_name():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class GRPOPolicy:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 64,
        reference_kl_weight: float = 0.05,
        clip_epsilon: float = 0.2,
        update_epochs: int = 2,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = device_name()
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True).to(self.device)
        self.model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        self.max_new_tokens = int(max_new_tokens)
        self.reference_kl_weight = float(reference_kl_weight)
        self.clip_epsilon = float(clip_epsilon)
        self.update_epochs = int(update_epochs)
        # Keep dropout disabled for both rollout and scoring. Gradients still
        # flow in eval mode, and every likelihood then refers to one policy.
        self.model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def sample_group(self, prompt: str, group_size: int, temperature: float):
        inputs = self.tokenizer(prompt.rstrip() + "\n", return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=temperature,
                top_p=0.95, num_return_sequences=group_size,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        p_len = inputs.input_ids.shape[1]
        samples = []
        for sequence in out:
            ids = sequence[p_len:].tolist()
            if self.tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(self.tokenizer.eos_token_id) + 1]
            while ids and ids[-1] == self.tokenizer.pad_token_id:
                ids.pop()
            if not ids:
                ids = [int(self.tokenizer.eos_token_id)]
            samples.append({
                "token_ids": ids,
                "text": self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
            })
        return inputs.input_ids[0].tolist(), samples

    def generate_greedy(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt.rstrip() + "\n", return_tensors="pt").to(self.device)
        self.model.eval()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    def completion_logprobs(self, prompt_ids: list[int], completion_ids: list[int]):
        """Score the exact generated tokens without decode/re-tokenize drift."""
        full = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=self.device)
        p_len = len(prompt_ids)
        self.model.eval()
        logits = self.model(full).logits[:, :-1, :]
        target = full[:, 1:]
        start = max(p_len - 1, 0)
        student_logits = logits[:, start:, :]
        target = target[:, start:]
        student_logp = (
            student_logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            - torch.logsumexp(student_logits, dim=-1)
        )
        with torch.no_grad(), self.model.disable_adapter():
            reference_logits = self.model(full).logits[:, :-1, :][:, start:, :]
            reference_logp = (
                reference_logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                - torch.logsumexp(reference_logits, dim=-1)
            )
        # Non-negative per-token k3 estimator. Under policy samples its
        # expectation estimates KL(policy || reference).
        log_ratio_inverse = reference_logp - student_logp
        reference_kl = torch.exp(log_ratio_inverse) - log_ratio_inverse - 1.0
        return student_logp.squeeze(0), reference_kl.squeeze(0)

    def update_task(self, task, group_size: int, temperature: float):
        prompt_ids, samples = self.sample_group(task.prompt, group_size, temperature)
        rewards = torch.tensor([float(normalize(sample["text"], task.family) == normalize(task.answer, task.family)) for sample in samples], device=self.device)
        advantages = rewards - rewards.mean()
        if float(advantages.std(unbiased=False)) > 0:
            advantages = advantages / (advantages.std(unbiased=False) + 1e-8)
        if float(advantages.abs().sum()) == 0.0:
            return {"mean_reward": float(rewards.mean()), "reference_kl": 0.0, "updated": False, "completions": samples}
        with torch.no_grad():
            old_logps = [self.completion_logprobs(prompt_ids, sample["token_ids"])[0].detach() for sample in samples]
        losses = []
        mean_kls = []
        for _ in range(self.update_epochs):
            objectives = []
            kls = []
            for advantage, old_logp, sample in zip(advantages, old_logps, samples):
                current_logp, reference_kl = self.completion_logprobs(prompt_ids, sample["token_ids"])
                ratio = torch.exp((current_logp - old_logp).clamp(-20.0, 20.0))
                clipped = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                objectives.append(-torch.minimum(ratio * advantage.detach(), clipped * advantage.detach()).mean())
                kls.append(reference_kl.mean())
            policy_loss = torch.stack(objectives).mean()
            reference_kl = torch.stack(kls).mean()
            loss = policy_loss + self.reference_kl_weight * reference_kl
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            losses.append(float(loss.detach()))
            mean_kls.append(float(reference_kl.detach()))
        return {
            "mean_reward": float(rewards.mean()),
            "reference_kl": sum(mean_kls) / len(mean_kls),
            "updated": True,
            "loss": sum(losses) / len(losses),
            "completions": samples,
        }

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True); ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--train-per-family", type=int, default=2); ap.add_argument("--test-per-family", type=int, default=10)
    ap.add_argument("--group-size", type=int, default=4); ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--output", default="outputs/grpo")
    a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True); (out / "adapters").mkdir(exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(a), "torch": torch.__version__, "device": device_name(), "objective": "relative verifier reward; no answer-token labels"}, indent=2))
    test = dataset("test", a.test_per_family, a.seed + 10000)
    policy = GRPOPolicy(a.model_path)
    metrics, predictions = [], []
    for rnd in range(a.rounds + 1):
        if rnd == 0:
            runner = Runner(a.model_path)
            acc, rows = evaluate(runner, test, None, "grpo_baseline", 0)
            metrics.append({"round": 0, "condition": "grpo_baseline", "accuracy": acc})
            predictions.extend(rows)
            del runner
        else:
            explore = dataset(f"grpo-r{rnd}", a.train_per_family, a.seed + rnd)
            start = time.time(); updates = 0; rewards = []
            for task in explore:
                result = policy.update_task(task, a.group_size, a.temperature)
                rewards.append(result["mean_reward"]); updates += int(result["updated"])
            adapter = out / "adapters" / f"round_{rnd}"
            policy.save(adapter)
            del policy.model
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()
            policy = GRPOPolicy(a.model_path)
            policy.model.load_adapter(adapter, adapter_name="default")
            runner = Runner(a.model_path, str(adapter))
            acc, rows = evaluate(runner, test, None, "grpo", rnd)
            metrics.append({"round": rnd, "condition": "grpo", "accuracy": acc, "mean_train_reward": sum(rewards) / len(rewards), "updated_tasks": updates, "seconds": time.time() - start})
            predictions.extend(rows)
            del runner
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for m in metrics for k in m})); w.writeheader(); w.writerows(metrics)
    with (out / "predictions.jsonl").open("w") as f:
        for row in predictions: f.write(json.dumps(row) + "\n")
    (out / "report.md").write_text("# GRPO-style verifier-reward experiment\n\n" + "\n".join(f"- Round {m['round']} **{m['condition']}**: {m['accuracy']:.3f}" for m in metrics) + "\n\nThe policy was updated with group-relative verifier rewards. Correct answers were used only by the evaluator to assign rewards, never as supervised target tokens. The earlier SFT results remain in `outputs/`.\n")


if __name__ == "__main__": main()
