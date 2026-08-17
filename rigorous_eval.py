from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
from pathlib import Path

import torch

from experiment import Runner
from learned_skill import load_cards, skill_prefix
from real_benchmark import check_arc, check_gsm, check_humaneval
from rigorous_data import build_test_tasks, task_identity
from reproducibility import content_sha256, directory_fingerprint, sha256_file


def wilson(hits: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return centre - radius, centre + radius


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gains, losses) + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def load_frozen_tasks(path: Path, model_path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(path.read_text())
    if manifest.get("status") != "frozen_before_test_inference":
        raise RuntimeError("test manifest is not marked frozen")
    if content_sha256(manifest["tasks"]) != manifest["task_list_sha256"]:
        raise RuntimeError("test task-list hash does not match the frozen manifest")
    for source in manifest["source_files"]:
        source_path = Path(source["path"])
        if source_path.stat().st_size != source["bytes"] or sha256_file(source_path) != source["sha256"]:
            raise RuntimeError(f"dataset source fingerprint mismatch: {source_path}")
    current_model = directory_fingerprint(model_path)
    if current_model["aggregate_sha256"] != manifest["model"]["aggregate_sha256"]:
        raise RuntimeError("local model fingerprint does not match the frozen manifest")
    counts = manifest["counts"]
    tasks = build_test_tasks(counts["gsm8k"], counts["humaneval"], counts["arc_challenge"], manifest["seed"])
    if [task_identity(row) for row in tasks] != manifest["tasks"]:
        raise RuntimeError("source rows no longer reproduce the frozen test manifest")
    return manifest, tasks


def selected_adapter(path: Path) -> str | None:
    metadata = json.loads(path.read_text())
    adapter = metadata.get("adapter")
    if adapter and not Path(adapter).exists():
        raise FileNotFoundError(adapter)
    return adapter


def score(prediction: str, task: dict) -> bool:
    if task["benchmark"] == "gsm8k":
        return check_gsm(prediction, task["gold"])
    if task["benchmark"] == "arc_challenge":
        return check_arc(prediction, task["gold"])
    return check_humaneval(prediction, task)


def evaluate(name: str, seed: int | None, runner: Runner, tasks: list[dict], cards: list[dict] | None = None) -> list[dict]:
    rows = []
    for task in tasks:
        prefix = skill_prefix(cards, task["prompt"]) if cards and task["benchmark"] == "gsm8k" else ""
        limit = 384 if task["benchmark"] == "humaneval" else (256 if task["benchmark"] == "gsm8k" else 64)
        prediction = runner.generate(task["prompt"], prefix, max_new_tokens=limit)
        rows.append({
            "condition": name,
            "training_seed": seed,
            "benchmark": task["benchmark"],
            "task_id": task["task_id"],
            "correct": score(prediction, task),
            "prediction": prediction,
            "prompt_prefix_tokens": len(runner.tokenizer(prefix).input_ids) if prefix else 0,
        })
    return rows


def summarize(rows: list[dict], seeds: list[int]) -> tuple[list[dict], list[dict]]:
    metric_rows = []
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        for seed in sorted({row["training_seed"] for row in condition_rows}, key=lambda value: -1 if value is None else value):
            seeded = [row for row in condition_rows if row["training_seed"] == seed]
            for benchmark in ("gsm8k", "arc_challenge", "humaneval"):
                subset = [row for row in seeded if row["benchmark"] == benchmark]
                hits = sum(row["correct"] for row in subset)
                low, high = wilson(hits, len(subset))
                metric_rows.append({"condition": condition, "training_seed": seed, "benchmark": benchmark, "hits": hits, "n": len(subset), "accuracy": hits / len(subset), "wilson_low": low, "wilson_high": high})
    base = {row["task_id"]: row["correct"] for row in rows if row["condition"] == "base"}
    paired = []
    for condition in sorted({row["condition"] for row in rows} - {"base"}):
        for seed in seeds:
            subset = [row for row in rows if row["condition"] == condition and row["training_seed"] == seed and row["benchmark"] == "gsm8k"]
            gains = sum(bool(row["correct"]) and not base[row["task_id"]] for row in subset)
            losses = sum(not bool(row["correct"]) and base[row["task_id"]] for row in subset)
            paired.append({"condition": condition, "training_seed": seed, "benchmark": "gsm8k", "gains_vs_base": gains, "losses_vs_base": losses, "exact_mcnemar_p": exact_mcnemar(gains, losses)})
    return metric_rows, paired


def aggregate_seed_metrics(metrics: list[dict]) -> list[dict]:
    rows = []
    for condition in sorted({row["condition"] for row in metrics}):
        for benchmark in ("gsm8k", "arc_challenge", "humaneval"):
            values = [row["accuracy"] for row in metrics if row["condition"] == condition and row["benchmark"] == benchmark]
            rows.append({
                "condition": condition,
                "benchmark": benchmark,
                "training_seeds": len(values),
                "mean_accuracy": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min_accuracy": min(values),
                "max_accuracy": max(values),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", default="protocol/fresh_test_manifest.json")
    parser.add_argument("--root", default="outputs/rigorous")
    parser.add_argument("--output", default="outputs/rigorous/evaluation")
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260817, 20260818, 20260819])
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    manifest, tasks = load_frozen_tasks(Path(args.manifest), Path(args.model_path))
    rows = []
    runner = Runner(args.model_path)
    rows.extend(evaluate("base", None, runner, tasks))
    del runner
    for seed in args.seeds:
        seed_root = Path(args.root) / f"seed_{seed}"
        cards = load_cards(seed_root / "learned_skills.json")
        runner = Runner(args.model_path)
        rows.extend(evaluate("learned_skill", seed, runner, tasks, cards))
        del runner
        for condition, metadata in (
            ("gold_sft_reference", seed_root / "domain" / "sft" / "final_checkpoint.json"),
            ("grpo_verifier", seed_root / "domain" / "grpo" / "final_checkpoint.json"),
            ("lopd_lite", seed_root / "lopd" / "final_checkpoint.json"),
        ):
            runner = Runner(args.model_path, selected_adapter(metadata))
            rows.extend(evaluate(condition, seed, runner, tasks))
            del runner
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    metrics, paired = summarize(rows, args.seeds)
    aggregates = aggregate_seed_metrics(metrics)
    with (out / "predictions.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for filename, records in (("metrics.csv", metrics), ("aggregate_metrics.csv", aggregates), ("paired_tests.csv", paired)):
        with (out / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
