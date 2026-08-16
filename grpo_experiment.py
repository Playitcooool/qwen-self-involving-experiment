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
    def __init__(self, model_path: str, max_new_tokens: int = 64):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = device_name()
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True).to(self.device)
        self.model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
        self.max_new_tokens = int(max_new_tokens)
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

    def sample_group(self, prompt: str, group_size: int, temperature: float):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=temperature,
                top_p=0.95, num_return_sequences=group_size,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        p_len = inputs.input_ids.shape[1]
        return [self.tokenizer.decode(seq[p_len:], skip_special_tokens=True).strip() for seq in out]

    def completion_logprob(self, prompt: str, completion: str):
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        full = self.tokenizer(prompt + completion, return_tensors="pt").input_ids.to(self.device)
        p_len = prompt_ids.shape[1]
        logits = self.model(full).logits[:, :-1, :]
        logp = torch.log_softmax(logits, dim=-1)
        target = full[:, 1:]
        start = max(p_len - 1, 0)
        return logp[:, start:, :].gather(-1, target[:, start:].unsqueeze(-1)).squeeze(-1).sum()

    def update_task(self, task, group_size: int, temperature: float):
        completions = self.sample_group(task.prompt, group_size, temperature)
        rewards = torch.tensor([float(normalize(c, task.family) == normalize(task.answer, task.family)) for c in completions], device=self.device)
        advantages = rewards - rewards.mean()
        if float(advantages.abs().sum()) == 0.0:
            return {"mean_reward": float(rewards.mean()), "updated": False}
        logps = torch.stack([self.completion_logprob(task.prompt, c) for c in completions])
        loss = -(advantages.detach() * logps).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {"mean_reward": float(rewards.mean()), "updated": True, "loss": float(loss.detach())}

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
