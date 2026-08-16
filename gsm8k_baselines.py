"""Train domain-matched GSM8K SFT and verifier-reward GRPO baselines."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from experiment import normalize
from gsm8k_data import (
    GSM8KDataSplits,
    build_gsm8k_splits,
    gsm8k_solution,
    load_gsm8k_splits_from_manifest,
    load_gsm8k_rows,
    write_manifest,
)
from grpo_experiment import GRPOPolicy, device_name


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
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
    trainer.train()
    adapter = output / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    result = {
        "condition": "sft_gsm8k",
        "tasks": len(tasks),
        "epochs": args.sft_epochs,
        "seconds": time.time() - started,
        "adapter": str(adapter),
        "uses_gold_labels": True,
        "target": "official GSM8K rationale followed by final numeric answer",
    }
    del trainer, model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


def train_grpo(model_path: str, data: GSM8KDataSplits, output: Path, args: argparse.Namespace):
    started = time.time()
    policy = GRPOPolicy(
        model_path,
        max_new_tokens=args.max_new_tokens,
        reference_kl_weight=args.reference_kl_weight,
    )
    round_rows = []
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
        round_rows.append(
            {
                "round": round_no,
                "condition": "grpo_gsm8k",
                "mean_train_reward": sum(rewards) / len(rewards),
                "mean_reference_kl": sum(reference_kls) / len(reference_kls),
                "updated_tasks": updated,
                "seconds": time.time() - update_start,
                "adapter": str(checkpoint),
            }
        )
    adapter = output / "adapter"
    policy.save(adapter)
    result = {
        "condition": "grpo_gsm8k",
        "tasks": len(data.update_tasks),
        "rounds": len(data.update_rounds),
        "seconds": time.time() - started,
        "adapter": str(adapter),
        "uses_gold_labels": False,
        "reward": "group-relative exact GSM8K verifier reward",
    }
    with (output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in round_rows for key in row}))
        writer.writeheader()
        writer.writerows(round_rows)
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)
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
