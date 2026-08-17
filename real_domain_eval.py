"""Evaluate domain-matched GSM8K adapters on the fixed held-out suite."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from pathlib import Path

import torch

from experiment import Runner
from real_benchmark import arc_tasks, evaluate_condition, gsm_tasks, humaneval_tasks


def load_tasks(manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    counts = manifest["counts"]
    seed = int(manifest["seed"])
    tasks = (
        gsm_tasks(int(counts["gsm8k"]), seed)
        + humaneval_tasks(int(counts["humaneval"]), seed + 1)
        + arc_tasks(int(counts["arc_challenge"]), seed + 2)
    )
    if {row["task_id"] for row in tasks} != set(manifest["task_ids"]):
        raise RuntimeError("held-out task reconstruction does not match its manifest")
    return manifest, tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", default="outputs/real_benchmark/manifest.json")
    parser.add_argument("--output", default="outputs/real_benchmark_gsm8k")
    parser.add_argument("--sft-adapter", default="outputs/gsm8k_domain/sft/adapter")
    parser.add_argument("--grpo-adapter", default="outputs/gsm8k_domain/grpo/adapter")
    parser.add_argument("--lopd-adapter", default="outputs/gsm8k_lopd/adapters/best")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest, tasks = load_tasks(Path(args.manifest))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    conditions = [("baseline", None), ("skill", None)]
    for name, value in (
        ("sft_gsm8k", args.sft_adapter),
        ("grpo_gsm8k", args.grpo_adapter),
        ("lopd_lite_gsm8k", args.lopd_adapter),
    ):
        if not value or not Path(value).exists():
            raise FileNotFoundError(
                f"required {name} adapter does not exist: {value}. "
                "Run the matched training jobs before evaluation."
            )
        conditions.append((name, value))
    all_rows = []
    metrics = []
    started = time.time()
    for name, adapter in conditions:
        runner = Runner(args.model_path, adapter)
        rows, seconds = evaluate_condition(name, runner, tasks)
        all_rows.extend(rows)
        metrics.append(
            {
                "condition": name,
                "benchmark": "all",
                "accuracy": sum(row["correct"] for row in rows) / len(rows),
                "seconds": seconds,
            }
        )
        for benchmark in sorted({row["benchmark"] for row in rows}):
            subset = [row for row in rows if row["benchmark"] == benchmark]
            metrics.append(
                {
                    "condition": name,
                    "benchmark": benchmark,
                    "accuracy": sum(row["correct"] for row in subset) / len(subset),
                    "seconds": seconds,
                }
            )
        del runner
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    with (out / "predictions.jsonl").open("w") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "benchmark", "accuracy", "seconds"])
        writer.writeheader()
        writer.writerows(metrics)
    by_condition: dict[str, dict[str, float]] = {}
    for row in metrics:
        by_condition.setdefault(row["condition"], {})[row["benchmark"]] = float(row["accuracy"])
    order = ["baseline", "skill", "sft_gsm8k", "grpo_gsm8k", "lopd_lite_gsm8k"]
    lines = [
        "# Domain-matched GSM8K comparison",
        "",
        "All conditions use the same fixed held-out manifest. Training uses only GSM8K train rows; HumanEval and ARC are transfer checks.",
        "",
        "| Condition | Overall | GSM8K | HumanEval | ARC-Challenge |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in order:
        if condition not in by_condition:
            continue
        values = by_condition[condition]
        lines.append(
            f"| {condition} | {values.get('all', 0.0):.3f} | {values.get('gsm8k', 0.0):.3f} | "
            f"{values.get('humaneval', 0.0):.3f} | {values.get('arc_challenge', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "LOPD-lite is evaluated student-only: no experience retrieval or latent context is available during evaluation.",
            f"Evaluation wall time: {time.time() - started:.1f}s.",
            "Gold answers are used only inside benchmark verifiers.",
        ]
    )
    (out / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
