"""Evaluate a LOPD-lite student on the exact existing real-benchmark subset."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from real_benchmark import arc_tasks, evaluate_condition, gsm_tasks, humaneval_tasks
from experiment import Runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--manifest", default="outputs/real_benchmark/manifest.json")
    parser.add_argument("--output", default="outputs/real_benchmark_lopd")
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    counts = manifest["counts"]
    seed = int(manifest["seed"])
    tasks = (
        gsm_tasks(int(counts["gsm8k"]), seed)
        + humaneval_tasks(int(counts["humaneval"]), seed + 1)
        + arc_tasks(int(counts["arc_challenge"]), seed + 2)
    )
    expected_ids = set(manifest["task_ids"])
    actual_ids = {row["task_id"] for row in tasks}
    if actual_ids != expected_ids:
        raise RuntimeError("reconstructed tasks do not match outputs/real_benchmark/manifest.json")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.time()
    runner = Runner(args.model_path, args.adapter)
    rows, seconds = evaluate_condition("lopd_lite", runner, tasks)
    metrics = [
        {
            "condition": "lopd_lite",
            "benchmark": "all",
            "accuracy": sum(row["correct"] for row in rows) / len(rows),
            "seconds": seconds,
        }
    ]
    for benchmark in sorted({row["benchmark"] for row in rows}):
        subset = [row for row in rows if row["benchmark"] == benchmark]
        metrics.append(
            {
                "condition": "lopd_lite",
                "benchmark": benchmark,
                "accuracy": sum(row["correct"] for row in subset) / len(subset),
                "seconds": seconds,
            }
        )
    with (out / "predictions.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "benchmark", "accuracy", "seconds"])
        writer.writeheader()
        writer.writerows(metrics)
    previous = []
    previous_path = Path("outputs/real_benchmark/metrics.csv")
    if previous_path.exists():
        with previous_path.open() as handle:
            previous = list(csv.DictReader(handle))
    all_metrics = previous + [{**row, "accuracy": str(row["accuracy"])} for row in metrics]
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "benchmark", "accuracy", "seconds"])
        writer.writeheader()
        writer.writerows(all_metrics)
    by_condition: dict[str, dict[str, float]] = {}
    for row in all_metrics:
        by_condition.setdefault(row["condition"], {})[row["benchmark"]] = float(row["accuracy"])
    order = ["baseline", "sft", "grpo", "skill", "lopd_lite"]
    lines = [
        "# Real benchmark comparison including LOPD-lite",
        "",
        "Same fixed subset as `outputs/real_benchmark/manifest.json`; this is not an official full-set score.",
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
            "LOPD-lite was trained on the earlier synthetic training distribution; therefore this is a transfer evaluation, not a domain-matched reproduction of the paper's EnvScaler/TACO setup.",
            f"Evaluation wall time: {time.time() - started:.1f}s.",
            "Gold answers are used only by the existing benchmark verifiers.",
        ]
    )
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    del runner
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
