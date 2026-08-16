"""A small, reproducible implementation of Latent On-Policy Self-Distillation.

This is deliberately a LOPD-lite implementation for the local Qwen3.5-0.8B
checkpoint.  It keeps the paper's important asymmetry:

* the student rolls out without privileged context;
* a frozen copy of the same base model receives learnable latent context;
* the student matches the privileged teacher on the exact rollout prefixes;
* a verifier reward keeps the learned context useful;
* only the LoRA student is kept for inference.

The implementation uses a lightweight learned-query composer over frozen input
embeddings instead of the paper's full QFormer encoder.  The driver constructs
the experience bank from successful base-model rollouts on its training tasks
(synthetic or domain-matched real data), and it never stores the reference answer
as a model input.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment import Task, dataset, normalize


def device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def model_hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    return int(getattr(text_config or config, "hidden_size"))


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def prompt_text(prompt: str) -> str:
    """Use the same explicit separator for generation and teacher/student loss."""
    return prompt.rstrip() + "\n"


def encode_text(tokenizer, text: str, max_length: int | None = None) -> list[int]:
    kwargs: dict[str, Any] = {"add_special_tokens": True}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})
    return list(tokenizer(text, **kwargs).input_ids)


def encode_completion(tokenizer, text: str, max_length: int | None = None) -> list[int]:
    kwargs: dict[str, Any] = {"add_special_tokens": False}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max_length})
    ids = list(tokenizer(text, **kwargs).input_ids)
    if not ids:
        ids = [tokenizer.eos_token_id]
    elif tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    return ids


def completion_slices(prompt_len: int, completion_len: int, latent_len: int) -> tuple[slice, slice]:
    """Return causal-logit slices for student and teacher completion tokens."""
    student_start = max(prompt_len - 1, 0)
    teacher_start = latent_len + max(prompt_len - 1, 0)
    return (
        slice(student_start, student_start + completion_len),
        slice(teacher_start, teacher_start + completion_len),
    )


def reverse_kl_topk(student_logits: torch.Tensor, teacher_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Reverse KL on teacher top-k plus one aggregate tail bucket.

    The teacher support is selected without gradient, while teacher values remain
    attached to the graph so gradients can reach the latent composer.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    k = min(int(top_k), int(teacher_logits.shape[-1]))
    student_logp = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    with torch.no_grad():
        indices = torch.topk(teacher_logp, k=k, dim=-1).indices
    student_top = student_logp.gather(-1, indices)
    teacher_top = teacher_logp.gather(-1, indices)
    student_prob = student_top.exp()
    teacher_prob = teacher_top.exp()
    student_tail = (1.0 - student_prob.sum(dim=-1)).clamp_min(1e-8)
    teacher_tail = (1.0 - teacher_prob.sum(dim=-1)).clamp_min(1e-8)
    top_term = (student_prob * (student_top - teacher_top)).sum(dim=-1)
    tail_term = student_tail * (student_tail.log() - teacher_tail.log())
    return top_term + tail_term


class LatentComposer(torch.nn.Module):
    """Small learned-query latent composer used by LOPD-lite.

    The paper uses a QFormer over backbone hidden states.  This compact variant
    uses frozen input embeddings as the memory and learned cross-attention
    queries, which is practical on the 0.8B local checkpoint while preserving
    a differentiable latent context.
    """

    def __init__(self, hidden_size: int, latent_tokens: int = 8):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_tokens = int(latent_tokens)
        scale = 1.0 / math.sqrt(hidden_size)
        self.queries = torch.nn.Parameter(torch.randn(latent_tokens, hidden_size) * scale)
        self.key = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.output = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        if memory.ndim == 2:
            memory = memory.unsqueeze(0)
        if memory.ndim != 3 or memory.shape[-1] != self.hidden_size:
            raise ValueError(f"expected memory [batch, tokens, {self.hidden_size}], got {tuple(memory.shape)}")
        queries = self.queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        keys = self.key(memory)
        values = self.value(memory)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.hidden_size)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, values)
        return self.norm(queries + self.output(attended))


@dataclass
class Experience:
    task_id: str
    family: str
    prompt: str
    trajectory: str
    reward: float = 1.0


class ExperienceBank:
    """Frozen successful-rollout bank with deterministic cosine retrieval."""

    def __init__(self):
        self.items: list[Experience] = []
        self.vectors: list[torch.Tensor] = []

    def add(self, item: Experience, vector: torch.Tensor) -> None:
        self.items.append(item)
        self.vectors.append(vector.detach().float().cpu())

    def __len__(self) -> int:
        return len(self.items)

    def retrieve(self, query: torch.Tensor, top_k: int, exclude_task_id: str | None = None) -> list[Experience]:
        candidates = [
            (idx, item)
            for idx, item in enumerate(self.items)
            if exclude_task_id is None or item.task_id != exclude_task_id
        ]
        if not candidates:
            # Never fall back to the excluded task. A one-item bank therefore
            # correctly reports that no privileged context is available rather
            # than leaking the current trajectory back into its own teacher.
            return []
        query = query.detach().float().cpu()
        query = query / query.norm().clamp_min(1e-8)
        scored = []
        for idx, item in candidates:
            vec = self.vectors[idx]
            vec = vec / vec.norm().clamp_min(1e-8)
            scored.append((float(torch.dot(query, vec)), idx, item))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [item for _, _, item in scored[: max(1, int(top_k))]]

    def write_jsonl(self, path: Path) -> None:
        with path.open("w") as handle:
            for item in self.items:
                # Do not serialize the verifier answer.  The trajectory is the
                # model-generated experience supplied to the privileged teacher.
                handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


class LOPDTrainer:
    def __init__(self, model_path: str, args: argparse.Namespace):
        self.model_path = model_path
        self.args = args
        self.device = device_name()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", trust_remote_code=True
        ).to(self.device)
        base.config.use_cache = False
        self.model: PeftModel = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        self.model.train()
        hidden_size = model_hidden_size(self.model)
        self.composer = LatentComposer(hidden_size, args.latent_tokens).to(self.device)
        self.student_params = [p for p in self.model.parameters() if p.requires_grad]
        self.composer_params = list(self.composer.parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.student_params, "lr": args.student_lr},
                {"params": self.composer_params, "lr": args.composer_lr},
            ],
            weight_decay=args.weight_decay,
        )
        self.bank = ExperienceBank()
        self.anchor_cache: dict[str, torch.Tensor] = {}
        self.beta = 0.0
        self._embedding = self.model.get_input_embeddings()

    def _text_vector(self, text: str) -> torch.Tensor:
        ids = torch.tensor([encode_text(self.tokenizer, text, self.args.encoder_max_tokens)], device=self.device)
        with torch.no_grad():
            emb = self._embedding(ids).float()
        return emb.mean(dim=1).squeeze(0)

    def _memory(self, task_prompt: str, experiences: Iterable[Experience]) -> torch.Tensor:
        joined = "\n\n--- PAST EXPERIENCE ---\n\n".join(
            f"Task: {item.prompt}\nTrajectory:\n{item.trajectory}" for item in experiences
        )
        text = f"Current task:\n{task_prompt}\n\n{joined}"
        ids = torch.tensor([encode_text(self.tokenizer, text, self.args.encoder_max_tokens)], device=self.device)
        with torch.no_grad():
            return self._embedding(ids).float()

    def _compose(self, task: Task, *, detach: bool = False) -> torch.Tensor:
        query = self._text_vector(task.prompt)
        experiences = self.bank.retrieve(query, self.args.retrieve_top_k, exclude_task_id=task.task_id)
        if not experiences:
            raise RuntimeError("LOPD requires at least one successful experience in the bank")
        latents = self.composer(self._memory(task.prompt, experiences))
        if detach:
            latents = latents.detach()
        return latents

    def generate(
        self,
        prompt: str,
        *,
        sample: bool,
        temperature: float | None = None,
        use_teacher: bool = False,
    ) -> str:
        was_training = self.model.training
        self.model.eval()
        text = prompt_text(prompt)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(self.device)
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.args.max_new_tokens,
            "do_sample": sample,
            "top_p": self.args.top_p,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if sample:
            kwargs["temperature"] = temperature or self.args.temperature
        adapter_context = self.model.disable_adapter() if use_teacher else nullcontext()
        with adapter_context, torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        if was_training:
            self.model.train()
        return self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

    def collect_experience(self, tasks: list[Task]) -> ExperienceBank:
        """Collect only successful base-model rollouts for the frozen bank."""
        seen: set[tuple[str, str]] = set()
        for task in tasks:
            for _ in range(self.args.bank_rollouts_per_task):
                pred = self.generate(task.prompt, sample=True, use_teacher=True)
                if normalize(pred, task.family) != normalize(task.answer, task.family):
                    continue
                key = (task.task_id, pred)
                if key in seen:
                    continue
                seen.add(key)
                item = Experience(
                    task_id=task.task_id,
                    family=task.family,
                    prompt=task.prompt,
                    trajectory=pred,
                )
                self.bank.add(item, self._text_vector(f"{task.prompt}\n{pred}"))
                if len(self.bank) >= self.args.max_bank:
                    return self.bank
        return self.bank

    def _sequence(self, task_prompt: str, completion: str) -> tuple[torch.Tensor, int, int]:
        pids = encode_text(self.tokenizer, prompt_text(task_prompt), self.args.max_sequence_tokens)
        remaining = max(1, self.args.max_sequence_tokens - len(pids))
        cids = encode_completion(self.tokenizer, completion, remaining)
        ids = torch.tensor([pids + cids], dtype=torch.long, device=self.device)
        return ids, len(pids), len(cids)

    def _forward_pair(
        self, task: Task, completion: str, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        full_ids, prompt_len, completion_len = self._sequence(task.prompt, completion)
        attention = torch.ones_like(full_ids)
        self.model.train()
        student = self.model(
            input_ids=full_ids,
            attention_mask=attention,
            use_cache=False,
            return_dict=True,
        ).logits
        token_embeds = self._embedding(full_ids)
        latent_embeds = latents.to(dtype=token_embeds.dtype)
        teacher_inputs = torch.cat([latent_embeds, token_embeds], dim=1)
        teacher_attention = torch.ones(
            (1, teacher_inputs.shape[1]), dtype=attention.dtype, device=self.device
        )
        was_training = self.model.training
        self.model.eval()
        try:
            with self.model.disable_adapter():
                teacher = self.model(
                    inputs_embeds=teacher_inputs,
                    attention_mask=teacher_attention,
                    use_cache=False,
                    return_dict=True,
                ).logits
        finally:
            if was_training:
                self.model.train()
        student_slice, teacher_slice = completion_slices(
            prompt_len, completion_len, latents.shape[1]
        )
        target = full_ids[:, prompt_len : prompt_len + completion_len]
        return (
            student[:, student_slice, :],
            teacher[:, teacher_slice, :],
            target,
            latents,
        )

    def _anchor(self, task: Task, current: torch.Tensor) -> torch.Tensor:
        if task.task_id not in self.anchor_cache:
            self.anchor_cache[task.task_id] = current.detach().clone()
        return self.anchor_cache[task.task_id].to(current.device)

    def cold_start(self) -> list[float]:
        """Initialize the composer with successful trajectories only."""
        if not len(self.bank):
            raise RuntimeError("cannot cold-start without successful experiences")
        cold_optimizer = torch.optim.AdamW(
            self.composer.parameters(), lr=self.args.cold_start_lr
        )
        losses: list[float] = []
        for item in self.bank.items[: self.args.cold_start_steps]:
            task = Task(item.task_id, item.family, item.prompt, "", "")
            latents = self._compose(task)
            # The trajectory itself is generated by the model; no gold answer is
            # passed to the teacher or composer.
            student_logits, teacher_logits, target, _ = self._forward_pair(
                task, item.trajectory, latents
            )
            del student_logits
            teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
            nll = -teacher_logp.gather(-1, target.unsqueeze(-1)).mean()
            cold_optimizer.zero_grad(set_to_none=True)
            nll.backward()
            torch.nn.utils.clip_grad_norm_(self.composer.parameters(), self.args.max_grad_norm)
            cold_optimizer.step()
            losses.append(float(nll.detach()))
        return losses

    def update_rollout(self, task: Task, completion: str, reward: float) -> dict[str, Any]:
        latents = self._compose(task)
        anchor = self._anchor(task, latents)
        student_logits, teacher_logits, target, _ = self._forward_pair(task, completion, latents)
        kl = reverse_kl_topk(student_logits, teacher_logits, self.args.top_k)
        kl_mean = kl.mean()
        student_logp = torch.log_softmax(student_logits.float(), dim=-1)
        teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
        sampled_student = student_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        sampled_teacher = teacher_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        delta = sampled_teacher - sampled_student.detach()
        advantage = 2.0 * float(reward) - 1.0
        privilege = (advantage * delta).mean()
        anchor_loss = (latents - anchor).pow(2).mean()
        loss = kl_mean + self.beta * (self.args.margin - privilege) + self.args.anchor_weight * anchor_loss
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        student_grad = torch.nn.utils.clip_grad_norm_(self.student_params, self.args.max_grad_norm)
        composer_grad = torch.nn.utils.clip_grad_norm_(self.composer_params, self.args.max_grad_norm)
        self.optimizer.step()
        self.beta = max(0.0, self.beta + self.args.dual_step * (self.args.margin - float(privilege.detach())))
        return {
            "task_id": task.task_id,
            "family": task.family,
            "reward": float(reward),
            "advantage": advantage,
            "loss": float(loss.detach()),
            "kl": float(kl_mean.detach()),
            "privilege": float(privilege.detach()),
            "anchor_loss": float(anchor_loss.detach()),
            "beta": self.beta,
            "student_grad_norm": float(student_grad),
            "composer_grad_norm": float(composer_grad),
            "updated": True,
        }

    def save(self, out: Path, args: argparse.Namespace) -> Path:
        adapter = out / "adapters" / f"round_{args.rounds}"
        adapter.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(adapter)
        self.tokenizer.save_pretrained(adapter)
        composer_dir = out / "composer"
        composer_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.composer.state_dict(), composer_dir / "composer.pt")
        (composer_dir / "config.json").write_text(
            json.dumps(
                {
                    "hidden_size": self.composer.hidden_size,
                    "latent_tokens": self.composer.latent_tokens,
                    "retrieve_top_k": args.retrieve_top_k,
                    "teacher": "frozen base model with adapters disabled",
                },
                indent=2,
            )
        )
        (out / "final_checkpoint.json").write_text(
            json.dumps({"adapter": str(adapter), "composer": str(composer_dir / "composer.pt")}, indent=2)
        )
        return adapter


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="outputs/lopd_lite")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--train-per-family", type=int, default=2)
    parser.add_argument("--bank-per-family", type=int, default=12)
    parser.add_argument("--bank-rollouts-per-task", type=int, default=3)
    parser.add_argument("--max-bank", type=int, default=64)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--retrieve-top-k", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-sequence-tokens", type=int, default=256)
    parser.add_argument("--encoder-max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--composer-lr", type=float, default=5e-4)
    parser.add_argument("--cold-start-lr", type=float, default=5e-4)
    parser.add_argument("--cold-start-steps", type=int, default=16)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--dual-step", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "torch": torch.__version__,
                "device": device_name(),
                "git_revision": git_revision(),
                "method": "LOPD-lite: reward-weighted latent on-policy self-distillation",
                "reward": "1 iff exact verifier normalization matches; 0 otherwise",
                "gold_answers_as_model_input": False,
            },
            indent=2,
        )
    )
    trainer = LOPDTrainer(args.model_path, args)
    bank_tasks = dataset("lopd-bank", args.bank_per_family, args.seed + 5000)
    train_tasks = dataset("lopd-train", args.train_per_family, args.seed)
    start = time.time()
    bank = trainer.collect_experience(bank_tasks)
    bank.write_jsonl(out / "experience_bank.jsonl")
    if not len(bank):
        raise RuntimeError(
            "The base model produced no verified successful rollouts. Increase --bank-per-family or --bank-rollouts-per-task."
        )
    cold_losses = trainer.cold_start()
    (out / "cold_start.json").write_text(json.dumps({"bank_size": len(bank), "losses": cold_losses}, indent=2))
    log_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    for round_no in range(1, args.rounds + 1):
        round_tasks = dataset(f"lopd-r{round_no}", args.train_per_family, args.seed + round_no)
        rewards: list[float] = []
        update_start = time.time()
        for task in round_tasks:
            completion = trainer.generate(task.prompt, sample=True)
            reward = float(normalize(completion, task.family) == normalize(task.answer, task.family))
            try:
                row = trainer.update_rollout(task, completion, reward)
            except RuntimeError as exc:
                row = {
                    "task_id": task.task_id,
                    "family": task.family,
                    "reward": reward,
                    "updated": False,
                    "error": str(exc),
                }
            row.update({"round": round_no, "completion": completion})
            log_rows.append(row)
            rewards.append(reward)
        round_rows.append(
            {
                "round": round_no,
                "condition": "lopd_lite",
                "mean_train_reward": sum(rewards) / len(rewards),
                "bank_size": len(bank),
                "beta": trainer.beta,
                "seconds": time.time() - update_start,
            }
        )
        adapter = out / "adapters" / f"round_{round_no}"
        adapter.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(adapter)
        trainer.tokenizer.save_pretrained(adapter)
    write_metrics(out / "metrics.csv", round_rows)
    with (out / "training_log.jsonl").open("w") as handle:
        for row in log_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    final_adapter = trainer.save(out, args)
    (out / "report.md").write_text(
        "# LOPD-lite experiment\n\n"
        f"- Successful base-model experiences: **{len(bank)}**\n"
        f"- Latent tokens: **{args.latent_tokens}**; retrieved experiences: **{args.retrieve_top_k}**\n"
        f"- Final student adapter: `{final_adapter}`\n"
        f"- Total training wall time: **{time.time() - start:.1f}s**\n\n"
        + "\n".join(
            f"- Round {row['round']}: mean verifier reward **{row['mean_train_reward']:.3f}**"
            for row in round_rows
        )
        + "\n\n"
        "The base model is the frozen privileged teacher; only the student LoRA and "
        "latent composer are optimized. The verifier answer is never inserted into "
        "the student prompt or teacher context as a label.\n"
    )
    del trainer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
