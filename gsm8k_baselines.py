"""Train domain-matched GSM8K SFT and verifier-reward GRPO baselines."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import time
import shutil
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from experiment import Runner, normalize
from gsm8k_data import (
    GSM8KDataSplits,
    build_gsm8k_splits,
    gsm8k_solution,
    load_gsm8k_splits_from_manifest,
    load_gsm8k_rows,
    write_manifest,
)
from grpo_experiment import GRPOPolicy, device_name
from reproducibility import git_revision, package_versions, seed_everything


class GSM8KSFTDataset(torch.utils.data.Dataset):
    """Fixed-length prompt/answer examples with loss masked on the prompt."""

    def __init__(self, tokenizer, tasks, max_length: int, targets: dict[str, str]):
        self.rows = []
        pad_id = tokenizer.pad_token_id
        for task in tasks:
            prompt_ids = tokenizer(task.prompt + "\n", add_special_tokens=True).input_ids
            answer_ids = tokenizer(targets[task.task_id], add_special_tokens=False).input_ids
            if tokenizer.eos_token_id is not None:
                answer_ids.append(int(tokenizer.eos_token_id))
            ids = (prompt_ids + answer_ids)[:max_length]
            labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
            padding = max_length - len(ids)
            self.rows.append(
                {
                    "input_ids": torch.tensor(ids + [pad_id] * padding, dtype=torch.long),
                    "attention_mask": torch.tensor([1] * len(ids) + [0] * padding, dtype=torch.long),
                    "labels": torch.tensor(labels + [-100] * padding, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def train_sft(model_path: str, tasks, output: Path, args: argparse.Namespace) -> dict[str, object]:
    if not args.validation_tasks_data:
        raise ValueError("SFT checkpoint selection requires at least one validation task")
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = device_name()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", trust_remote_code=True
    ).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    rows = load_gsm8k_rows("train")
    targets = {
        task.task_id: gsm8k_solution(rows[int(task.task_id.rsplit(":", 1)[-1])]["answer"])
        for task in tasks
    }
    dataset = GSM8KSFTDataset(tokenizer, tasks, args.max_sequence_tokens, targets)
    training_args = TrainingArguments(
        output_dir=str(output / "trainer"),
        per_device_train_batch_size=args.sft_batch_size,
        gradient_accumulation_steps=args.sft_grad_accumulation,
        num_train_epochs=args.sft_epochs,
        learning_rate=args.sft_lr,
        save_strategy="epoch",
        save_total_limit=None,
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    final_adapter = output / "final_adapter"
    final_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    checkpoints = sorted((output / "trainer").glob("checkpoint-*"), key=lambda path: int(path.name.rsplit("-", 1)[-1]))
    del trainer, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    candidates: list[tuple[int, Path | None]] = [(0, None)] + [(index, path) for index, path in enumerate(checkpoints, start=1)]
    validation_rows = []
    best_index = 0
    best_accuracy = -1.0
    best_path: Path | None = None
    for index, candidate in candidates:
        runner = Runner(model_path, str(candidate) if candidate else None)
        hits = 0
        for task in args.validation_tasks_data:
            prediction = runner.generate(task.prompt, max_new_tokens=args.max_new_tokens)
            correct = normalize(prediction, task.family) == normalize(task.answer, task.family)
            hits += int(correct)
            validation_rows.append({"checkpoint": index, "task_id": task.task_id, "correct": correct, "prediction": prediction})
        accuracy = hits / len(args.validation_tasks_data)
        if accuracy > best_accuracy:
            best_accuracy, best_index, best_path = accuracy, index, candidate
        del runner
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    with (output / "validation_predictions.jsonl").open("w") as handle:
        for row in validation_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    adapter = output / "adapter"
    if best_path is not None:
        shutil.copytree(best_path, adapter, dirs_exist_ok=True)
    result = {
        "condition": "gold_sft_reference",
        "tasks": len(tasks),
        "epochs": args.sft_epochs,
        "seconds": time.time() - started,
        "adapter": str(adapter) if best_path is not None else None,
        "final_adapter": str(final_adapter),
        "selected_checkpoint": best_index,
        "validation_accuracy": best_accuracy,
        "uses_gold_labels": True,
        "target": "official GSM8K rationale followed by final numeric answer",
    }
    (output / "final_checkpoint.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def train_grpo(model_path: str, data: GSM8KDataSplits, output: Path, args: argparse.Namespace):
    if not data.validation:
        raise ValueError("GRPO checkpoint selection requires at least one validation task")
    started = time.time()
    policy = GRPOPolicy(
        model_path,
        max_new_tokens=args.max_new_tokens,
        reference_kl_weight=args.reference_kl_weight,
        clip_epsilon=args.clip_epsilon,
        update_epochs=args.grpo_update_epochs,
    )
    round_rows = []
    validation_rows = []
    best_round = 0
    best_accuracy = -1.0
    best_adapter: str | None = None
    def validate(round_no: int) -> float:
        hits = 0
        for task in data.validation:
            prediction = policy.generate_greedy(task.prompt)
            correct = normalize(prediction, task.family) == normalize(task.answer, task.family)
            hits += int(correct)
            validation_rows.append({"round": round_no, "task_id": task.task_id, "correct": correct, "prediction": prediction})
        return hits / len(data.validation)
    best_accuracy = validate(0)
    round_rows.append({"round": 0, "condition": "grpo_verifier", "validation_accuracy": best_accuracy, "updated_tasks": 0})
    for round_no, tasks in enumerate(data.update_rounds, start=1):
        update_start = time.time()
        rewards = []
        reference_kls = []
        updated = 0
        for task in tasks:
            result = policy.update_task(task, args.group_size, args.temperature)
            rewards.append(result["mean_reward"])
            reference_kls.append(result.get("reference_kl", 0.0))
            updated += int(result["updated"])
        checkpoint = output / "adapters" / f"round_{round_no}"
        policy.save(checkpoint)
        validation_accuracy = validate(round_no)
        round_rows.append(
            {
                "round": round_no,
                "condition": "grpo_verifier",
                "mean_train_reward": sum(rewards) / len(rewards),
                "mean_reference_kl": sum(reference_kls) / len(reference_kls),
                "updated_tasks": updated,
                "seconds": time.time() - update_start,
                "adapter": str(checkpoint),
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_round = round_no
            best_adapter = str(checkpoint)
    final_adapter = output / "final_adapter"
    policy.save(final_adapter)
    adapter = output / "adapter"
    if best_adapter is not None:
        shutil.copytree(best_adapter, adapter, dirs_exist_ok=True)
    result = {
        "condition": "grpo_verifier",
        "tasks": len(data.update_tasks),
        "rounds": len(data.update_rounds),
        "seconds": time.time() - started,
        "adapter": str(adapter) if best_adapter is not None else None,
        "final_adapter": str(final_adapter),
        "selected_round": best_round,
        "validation_accuracy": best_accuracy,
        "uses_gold_labels": False,
        "reward": "group-relative exact GSM8K verifier reward",
    }
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in round_rows for key in row}))
        writer.writeheader()
        writer.writerows(round_rows)
    with (output / "validation_predictions.jsonl").open("w") as handle:
        for row in validation_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output / "final_checkpoint.json").write_text(json.dumps(result, indent=2) + "\n")
    del policy
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-manifest", default="outputs/gsm8k_lopd/data_manifest.json")
    parser.add_argument("--output", default="outputs/gsm8k_domain")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--bank-tasks", type=int, default=64)
    parser.add_argument("--tasks-per-round", type=int, default=8)
    parser.add_argument("--validation-tasks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--reference-kl-weight", type=float, default=0.05)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--grpo-update-epochs", type=int, default=2)
    parser.add_argument("--max-sequence-tokens", type=int, default=384)
    parser.add_argument("--sft-epochs", type=float, default=2.0)
    parser.add_argument("--sft-lr", type=float, default=2e-4)
    parser.add_argument("--sft-batch-size", type=int, default=2)
    parser.add_argument("--sft-grad-accumulation", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.data_manifest)
    if manifest_path.exists():
        data = load_gsm8k_splits_from_manifest(manifest_path)
    else:
        data = build_gsm8k_splits(
            bank_tasks=args.bank_tasks,
            rounds=args.rounds,
            tasks_per_round=args.tasks_per_round,
            validation_tasks=args.validation_tasks,
            seed=args.seed,
        )
        write_manifest(data, manifest_path)
    write_manifest(data, out / "data_manifest.json")
    (out / "config.json").write_text(json.dumps({**vars(args), "validation_tasks_data": None, "git_revision": git_revision(), "packages": package_versions(), "device": device_name(), "deterministic_algorithms": True}, default=str, indent=2) + "\n")
    args.validation_tasks_data = data.validation
    sft = train_sft(args.model_path, data.update_tasks, out / "sft", args)
    grpo = train_grpo(args.model_path, data, out / "grpo", args)
    (out / "summary.json").write_text(json.dumps({"sft": sft, "grpo": grpo}, indent=2) + "\n")
    (out / "report.md").write_text(
        "# Domain-matched GSM8K baselines\n\n"
        f"- Update tasks: **{len(data.update_tasks)}**\n"
        "- SFT uses the official GSM8K rationale followed by the final numeric answer as the target.\n"
        "- GRPO uses group-relative verifier rewards, with a small sampled reference-retention penalty.\n\n"
        f"- SFT adapter: `{sft['adapter']}`\n"
        f"- GRPO adapter: `{grpo['adapter']}`\n"
    )


if __name__ == "__main__":
    main()
