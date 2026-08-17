from __future__ import annotations

import argparse
import json
from pathlib import Path

from real_benchmark import DATA_CACHE, arc_tasks, gsm_tasks, humaneval_tasks
from reproducibility import content_sha256, directory_fingerprint, sha256_file


OLD_TEST_SEED = 20260815
NEW_TEST_SEED = 20260817


def task_identity(row: dict) -> dict[str, object]:
    return {
        "benchmark": row["benchmark"],
        "task_id": row["task_id"],
        "source_row_index": row["source_row_index"],
        "content_sha256": row["content_sha256"],
    }


def build_test_tasks(gsm8k: int, humaneval: int, arc: int, seed: int) -> list[dict]:
    old = gsm_tasks(30, OLD_TEST_SEED) + humaneval_tasks(10, OLD_TEST_SEED + 1) + arc_tasks(30, OLD_TEST_SEED + 2)
    excluded = {(row["benchmark"], row["source_row_index"]) for row in old}
    tasks = (
        gsm_tasks(gsm8k, seed, excluded_indices={i for b, i in excluded if b == "gsm8k"})
        + humaneval_tasks(humaneval, seed + 1, excluded_indices={i for b, i in excluded if b == "humaneval"})
        + arc_tasks(arc, seed + 2, excluded_indices={i for b, i in excluded if b == "arc_challenge"})
    )
    if {(row["benchmark"], row["source_row_index"]) for row in tasks} & excluded:
        raise RuntimeError("new test overlaps the prior development suite")
    return tasks


def write_frozen_manifest(path: Path, model_path: Path, gsm8k: int, humaneval: int, arc: int, seed: int) -> None:
    tasks = build_test_tasks(gsm8k, humaneval, arc, seed)
    cache_files = sorted(DATA_CACHE.glob("*.parquet"))
    identities = [task_identity(row) for row in tasks]
    manifest = {
        "protocol": "rigorous_protocol.md",
        "status": "frozen_before_test_inference",
        "seed": seed,
        "prior_development_seed": OLD_TEST_SEED,
        "prior_development_suite_excluded": True,
        "counts": {name: sum(row["benchmark"] == name for row in tasks) for name in ("gsm8k", "humaneval", "arc_challenge")},
        "tasks": identities,
        "task_list_sha256": content_sha256(identities),
        "source_files": [{"path": str(item), "bytes": item.stat().st_size, "sha256": sha256_file(item)} for item in cache_files],
        "model": directory_fingerprint(model_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", default="protocol/fresh_test_manifest.json")
    parser.add_argument("--gsm8k", type=int, default=60)
    parser.add_argument("--humaneval", type=int, default=20)
    parser.add_argument("--arc", type=int, default=40)
    parser.add_argument("--seed", type=int, default=NEW_TEST_SEED)
    args = parser.parse_args()
    write_frozen_manifest(Path(args.output), Path(args.model_path), args.gsm8k, args.humaneval, args.arc, args.seed)


if __name__ == "__main__":
    main()
