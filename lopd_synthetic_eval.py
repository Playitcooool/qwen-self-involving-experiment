"""Evaluate LOPD-lite on the exact synthetic held-out split used before."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from experiment import Runner, dataset, evaluate


def read_accuracy(path: Path, predicate) -> float | None:
    if not path.exists():
        return None
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if predicate(row)]
    if not matches:
        return None
    return float(matches[-1]["accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", default="outputs/lopd_lite/synthetic_eval")
    parser.add_argument("--test-per-family", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tasks = dataset("test", args.test_per_family, args.seed + 10000)
    started = time.time()
    runner = Runner(args.model_path, args.adapter)
    accuracy, rows = evaluate(runner, tasks, None, "lopd_lite", 5)
    safe_rows = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    with (out / "predictions.jsonl").open("w") as handle:
        for row in safe_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "accuracy", "seconds"])
        writer.writeheader()
        writer.writerow({"condition": "lopd_lite", "accuracy": accuracy, "seconds": time.time() - started})
    previous = {
        "baseline": read_accuracy(
            Path("outputs/metrics.csv"), lambda row: row["condition"] == "baseline"
        ),
        "sft_round_5": read_accuracy(
            Path("outputs/metrics.csv"), lambda row: row["condition"] == "training" and row["round"] == "5"
        ),
        "grpo_round_5": read_accuracy(
            Path("outputs/grpo/metrics.csv"), lambda row: row["condition"] == "grpo" and row["round"] == "5"
        ),
        "skill_retry": read_accuracy(
            Path("outputs/skill_retry/metrics.csv"), lambda row: row["condition"] == "skill_retry" and row["round"] == "5"
        ),
    }
    previous["lopd_lite"] = accuracy
    lines = [
        "# Synthetic held-out comparison including LOPD-lite",
        "",
        "This uses the exact 40-task synthetic held-out split from the prior experiment.",
        "",
        "| Condition | Accuracy |",
        "|---|---:|",
    ]
    for name in ["baseline", "sft_round_5", "grpo_round_5", "skill_retry", "lopd_lite"]:
        value = previous[name]
        if value is not None:
            lines.append(f"| {name} | {value:.3f} |")
    lines.extend(
        [
            "",
            "LOPD-lite uses the same five-round synthetic training distribution as the earlier comparison, with a separate successful-rollout bank.",
            "The verifier answer is used only to score the generated trajectory.",
        ]
    )
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "accuracy"])
        writer.writeheader()
        for name, value in previous.items():
            if value is not None:
                writer.writerow({"condition": name, "accuracy": value})
    del runner
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
