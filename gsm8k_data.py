"""Leakage-safe GSM8K train data utilities for the real-data experiment."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from experiment import Task
from real_benchmark import load_parquet


GSM8K_REPO = "openai/gsm8k"
GSM8K_CONFIG = "main"


def gsm8k_answer(raw_answer: str) -> str:
    """Extract the benchmark's final answer without retaining its derivation."""
    if "####" not in raw_answer:
        raise ValueError("GSM8K row has no #### final-answer delimiter")
    answer = raw_answer.rsplit("####", 1)[-1].strip().replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", answer)
    return match.group(0) if match else answer.rstrip(".!?")


def gsm8k_prompt(question: str) -> str:
    return (
        "Solve this grade-school math problem. Show concise reasoning, then write "
        "the final numeric answer on the last line.\n\n" + question
    )


def row_to_task(row: dict[str, Any], row_index: int, split: str) -> Task:
    return Task(
        task_id=f"gsm8k:{split}:{row_index}",
        family="gsm8k",
        prompt=gsm8k_prompt(str(row["question"])),
        answer=gsm8k_answer(str(row["answer"])),
        skill=(
            "Translate the story into quantities, solve step by step, check the "
            "arithmetic, and put only the final number on the last line."
        ),
    )


def load_gsm8k_rows(split: str = "train") -> list[dict[str, Any]]:
    path = f"{GSM8K_CONFIG}/{split}-00000-of-00001.parquet"
    return load_parquet(GSM8K_REPO, path)


def tasks_from_indices(
    rows: Sequence[dict[str, Any]], indices: Sequence[int], split: str = "train"
) -> list[Task]:
    return [row_to_task(rows[index], index, split) for index in indices]


def tasks_from_ids(rows: Sequence[dict[str, Any]], task_ids: Sequence[str]) -> list[Task]:
    tasks: list[Task] = []
    for task_id in task_ids:
        prefix, split, raw_index = task_id.split(":", 2)
        if prefix != "gsm8k" or split != "train":
            raise ValueError(f"unsupported GSM8K task ID: {task_id}")
        index = int(raw_index)
        if index < 0 or index >= len(rows):
            raise ValueError(f"GSM8K task index out of range: {task_id}")
        tasks.append(row_to_task(rows[index], index, split))
    return tasks


@dataclass
class GSM8KDataSplits:
    bank: list[Task]
    update_rounds: list[list[Task]]
    validation: list[Task]
    source: dict[str, Any]

    @property
    def update_tasks(self) -> list[Task]:
        return [task for group in self.update_rounds for task in group]

    def manifest(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "bank_task_ids": [task.task_id for task in self.bank],
            "update_task_ids_by_round": [
                [task.task_id for task in group] for group in self.update_rounds
            ],
            "validation_task_ids": [task.task_id for task in self.validation],
            "overlap_checks": {
                "bank_update": sorted(
                    {task.task_id for task in self.bank}
                    & {task.task_id for task in self.update_tasks}
                ),
                "bank_validation": sorted(
                    {task.task_id for task in self.bank}
                    & {task.task_id for task in self.validation}
                ),
                "update_validation": sorted(
                    {task.task_id for task in self.update_tasks}
                    & {task.task_id for task in self.validation}
                ),
            },
        }


def build_gsm8k_splits(
    *,
    bank_tasks: int,
    rounds: int,
    tasks_per_round: int,
    validation_tasks: int,
    seed: int,
) -> GSM8KDataSplits:
    if min(bank_tasks, rounds, tasks_per_round, validation_tasks) < 0:
        raise ValueError("split sizes must be non-negative")
    rows = load_gsm8k_rows("train")
    needed = bank_tasks + rounds * tasks_per_round + validation_tasks
    if needed > len(rows):
        raise ValueError(f"requested {needed} GSM8K train rows, only {len(rows)} available")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    cursor = 0
    bank_indices = order[cursor : cursor + bank_tasks]
    cursor += bank_tasks
    update_rounds: list[list[Task]] = []
    update_indices_by_round: list[list[int]] = []
    for _ in range(rounds):
        indices = order[cursor : cursor + tasks_per_round]
        cursor += tasks_per_round
        update_indices_by_round.append(indices)
    validation_indices = order[cursor : cursor + validation_tasks]
    bank = tasks_from_indices(rows, bank_indices)
    for indices in update_indices_by_round:
        update_rounds.append(tasks_from_indices(rows, indices))
    validation = tasks_from_indices(rows, validation_indices)
    data = GSM8KDataSplits(
        bank=bank,
        update_rounds=update_rounds,
        validation=validation,
        source={
            "dataset": f"{GSM8K_REPO}/{GSM8K_CONFIG}",
            "split": "train",
            "row_count": len(rows),
            "seed": seed,
            "requested_bank_tasks": bank_tasks,
            "requested_update_tasks": rounds * tasks_per_round,
            "requested_validation_tasks": validation_tasks,
        },
    )
    manifest = data.manifest()
    if any(manifest["overlap_checks"].values()):
        raise RuntimeError("GSM8K split construction produced overlapping task IDs")
    return data


def load_gsm8k_splits_from_manifest(path: Path) -> GSM8KDataSplits:
    manifest = json.loads(path.read_text())
    rows = load_gsm8k_rows("train")
    data = GSM8KDataSplits(
        bank=tasks_from_ids(rows, manifest["bank_task_ids"]),
        update_rounds=[tasks_from_ids(rows, ids) for ids in manifest["update_task_ids_by_round"]],
        validation=tasks_from_ids(rows, manifest["validation_task_ids"]),
        source=manifest["source"],
    )
    actual = data.manifest()
    if actual["bank_task_ids"] != manifest["bank_task_ids"]:
        raise RuntimeError("GSM8K bank task IDs do not reproduce the saved manifest")
    if actual["update_task_ids_by_round"] != manifest["update_task_ids_by_round"]:
        raise RuntimeError("GSM8K update task IDs do not reproduce the saved manifest")
    if actual["validation_task_ids"] != manifest["validation_task_ids"]:
        raise RuntimeError("GSM8K validation task IDs do not reproduce the saved manifest")
    if any(actual["overlap_checks"].values()):
        raise RuntimeError("saved GSM8K manifest contains overlapping task IDs")
    return data


def write_manifest(data: GSM8KDataSplits, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data.manifest(), indent=2) + "\n")
