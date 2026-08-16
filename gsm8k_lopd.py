"""Run LOPD-lite with real GSM8K training data and a held-out test manifest.

The GSM8K train split is partitioned into three disjoint roles:

* successful-rollout bank construction;
* on-policy student/composer updates;
* validation-only bookkeeping.

The existing GSM8K test manifest is never loaded by this training script.  The
gold answer is used only by the verifier through ``Task.answer`` and is never
inserted into a prompt or serialized in the experience bank.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import shutil
import time
from pathlib import Path

import torch

from gsm8k_data import build_gsm8k_splits, load_gsm8k_splits_from_manifest, write_manifest
from lopd_lite import LOPDTrainer, device_name, git_revision
from experiment import normalize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="outputs/gsm8k_lopd")
    parser.add_argument("--data-manifest", default="")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--bank-tasks", type=int, default=64)
    parser.add_argument("--tasks-per-round", type=int, default=8)
    parser.add_argument("--validation-tasks", type=int, default=32)
    parser.add_argument("--bank-rollouts-per-task", type=int, default=3)
    parser.add_argument("--max-bank", type=int, default=96)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--retrieve-top-k", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-sequence-tokens", type=int, default=384)
    parser.add_argument("--encoder-max-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--student-lr", type=float, default=1e-4)
    parser.add_argument("--composer-lr", type=float, default=5e-4)
    parser.add_argument("--cold-start-lr", type=float, default=5e-4)
    parser.add_argument("--cold-start-steps", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--dual-step", type=float, default=0.5)
    parser.add_argument("--anchor-weight", type=float, default=0.1)
    parser.add_argument("--reference-kl-weight", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    return parser


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.data_manifest) if args.data_manifest else None
    if manifest_path and manifest_path.exists():
        data = load_gsm8k_splits_from_manifest(manifest_path)
    else:
        data = build_gsm8k_splits(
            bank_tasks=args.bank_tasks,
            rounds=args.rounds,
            tasks_per_round=args.tasks_per_round,
            validation_tasks=args.validation_tasks,
            seed=args.seed,
        )
    write_manifest(data, out / "data_manifest.json")
    (out / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "torch": torch.__version__,
                "device": device_name(),
                "git_revision": git_revision(),
                "method": "LOPD-lite: domain-matched GSM8K reward-weighted latent on-policy self-distillation",
                "training_dataset": data.source,
                "gold_answers_as_model_input": False,
                "gold_answers_used_by": "exact GSM8K verifier only",
                "test_manifest_used_during_training": None,
            },
            indent=2,
        )
    )

    trainer = LOPDTrainer(args.model_path, args)
    started = time.time()
    bank = trainer.collect_experience(data.bank)
    bank.write_jsonl(out / "experience_bank.jsonl")
    if not len(bank):
        raise RuntimeError(
            "The base model produced no verified GSM8K rollouts. Increase --bank-tasks or --bank-rollouts-per-task."
        )
    cold_losses = trainer.cold_start()
    (out / "cold_start.json").write_text(
        json.dumps({"bank_size": len(bank), "losses": cold_losses}, indent=2)
    )

    log_rows: list[dict[str, object]] = []
    round_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    best_round: int | None = None
    best_validation_accuracy = -1.0
    for round_no, round_tasks in enumerate(data.update_rounds, start=1):
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
                "condition": "lopd_lite_gsm8k",
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
        validation_hits = 0
        for task in data.validation:
            prediction = trainer.generate(task.prompt, sample=False)
            correct = normalize(prediction, task.family) == normalize(task.answer, task.family)
            validation_hits += int(correct)
            validation_rows.append(
                {
                    "round": round_no,
                    "task_id": task.task_id,
                    "prediction": prediction,
                    "correct": correct,
                }
            )
        validation_accuracy = validation_hits / len(data.validation) if data.validation else 0.0
        round_rows[-1]["validation_accuracy"] = validation_accuracy
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_round = round_no

    if best_round is None:
        if not data.update_rounds:
            raise RuntimeError("cannot select a LOPD checkpoint without training rounds")
        best_round = len(data.update_rounds)
        best_validation_accuracy = 0.0

    write_metrics(out / "metrics.csv", round_rows)
    with (out / "training_log.jsonl").open("w") as handle:
        for row in log_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "validation_predictions.jsonl").open("w") as handle:
        for row in validation_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    final_adapter = trainer.save(out, args)
    best_adapter = out / "adapters" / "best"
    shutil.copytree(
        out / "adapters" / f"round_{best_round}",
        best_adapter,
        dirs_exist_ok=True,
    )
    (out / "final_checkpoint.json").write_text(
        json.dumps(
            {
                "adapter": str(best_adapter),
                "final_adapter": str(final_adapter),
                "selected_round": best_round,
                "validation_accuracy": best_validation_accuracy,
                "composer": str(out / "composer" / "composer.pt"),
            },
            indent=2,
        )
        + "\n"
    )
    (out / "report.md").write_text(
        "# Domain-matched GSM8K LOPD-lite experiment\n\n"
        f"- Successful base-model experiences: **{len(bank)}**\n"
        f"- GSM8K update tasks: **{len(data.update_tasks)}**\n"
        f"- GSM8K validation-only tasks: **{len(data.validation)}**\n"
        f"- Latent tokens: **{args.latent_tokens}**; retrieved experiences: **{args.retrieve_top_k}**\n"
        f"- Final student adapter: `{final_adapter}`\n"
        f"- Selected adapter: `{best_adapter}` (round **{best_round}**, validation accuracy **{best_validation_accuracy:.3f}**)\n"
        f"- Total training wall time: **{time.time() - started:.1f}s**\n\n"
        + "\n".join(
            f"- Round {row['round']}: mean verifier reward **{row['mean_train_reward']:.3f}**, validation accuracy **{row['validation_accuracy']:.3f}**"
            for row in round_rows
        )
        + "\n\n"
        "The GSM8K train split was partitioned before training. The bank contains "
        "only successful model-generated trajectories, and the final student is "
        "evaluated without retrieval or latent context. Gold answers are used only "
        "inside the verifier.\n"
    )
    del trainer
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
