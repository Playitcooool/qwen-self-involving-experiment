from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from gsm8k_data import build_gsm8k_splits, write_manifest
from rigorous_data import write_frozen_manifest


SEEDS = (20260817, 20260818, 20260819)


def execute(command: list[str], log: list[list[str]]) -> None:
    log.append(command)
    subprocess.run(command, check=True)


def prepare(model_path: Path) -> None:
    protocol = Path("protocol")
    protocol.mkdir(exist_ok=True)
    train_manifest = protocol / "train_manifest.json"
    if not train_manifest.exists():
        data = build_gsm8k_splits(bank_tasks=64, rounds=5, tasks_per_round=8, validation_tasks=32, seed=20260817)
        write_manifest(data, train_manifest)
    test_manifest = protocol / "fresh_test_manifest.json"
    if not test_manifest.exists():
        write_frozen_manifest(test_manifest, model_path, gsm8k=60, humaneval=20, arc=40, seed=20260817)


def train(model_path: Path, seeds: list[int]) -> None:
    commands: list[list[str]] = []
    for seed in seeds:
        root = Path("outputs/rigorous") / f"seed_{seed}"
        execute([
            sys.executable, "gsm8k_baselines.py", "--model-path", str(model_path),
            "--data-manifest", "protocol/train_manifest.json", "--output", str(root / "domain"),
            "--seed", str(seed), "--rounds", "5", "--tasks-per-round", "8", "--validation-tasks", "32",
            "--group-size", "4", "--grpo-update-epochs", "2", "--max-sequence-tokens", "512",
        ], commands)
        execute([
            sys.executable, "gsm8k_lopd.py", "--model-path", str(model_path),
            "--data-manifest", "protocol/train_manifest.json", "--output", str(root / "lopd"),
            "--seed", str(seed), "--rounds", "5", "--tasks-per-round", "8", "--validation-tasks", "32",
            "--max-sequence-tokens", "512",
        ], commands)
        execute([
            sys.executable, "learned_skill.py", "--model-path", str(model_path),
            "--experience-bank", str(root / "lopd" / "experience_bank.jsonl"),
            "--output", str(root / "learned_skills.json"), "--max-cards", "24",
        ], commands)
    Path("outputs/rigorous").mkdir(parents=True, exist_ok=True)
    (Path("outputs/rigorous") / "training_commands.json").write_text(json.dumps(commands, indent=2) + "\n")


def evaluate(model_path: Path, seeds: list[int]) -> None:
    command = [
        sys.executable, "rigorous_eval.py", "--model-path", str(model_path),
        "--manifest", "protocol/fresh_test_manifest.json", "--root", "outputs/rigorous",
        "--output", "outputs/rigorous/evaluation", "--seeds", *[str(seed) for seed in seeds],
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stage", choices=("prepare", "train", "evaluate", "all"), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    model_path = Path(args.model_path)
    if args.stage in ("prepare", "all"):
        prepare(model_path)
    if args.stage in ("train", "all"):
        train(model_path, args.seeds)
    if args.stage in ("evaluate", "all"):
        evaluate(model_path, args.seeds)


if __name__ == "__main__":
    main()
