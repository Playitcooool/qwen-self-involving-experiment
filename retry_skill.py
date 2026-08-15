"""Retry the external-skill condition with deduplicated, silent skill cards."""
from __future__ import annotations

import csv, json, time
from pathlib import Path

from experiment import Runner, dataset, normalize


def load_skills(root: Path, through_round: int):
    by_family = {}
    for n in range(1, through_round + 1):
        path = root / f"skills_round_{n}.json"
        if not path.exists():
            continue
        for card in json.loads(path.read_text()):
            by_family.setdefault(card["family"], card["skill"])
    return by_family


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--source-output", default="outputs")
    ap.add_argument("--output", default="outputs/skill_retry")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--test-per-family", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    source = Path(a.source_output); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    test = dataset("test", a.test_per_family, a.seed + 10000)
    runner = Runner(a.model_path)
    rows, metrics = [], []
    for rnd in range(1, a.rounds + 1):
        skills = load_skills(source, rnd)
        start = time.time(); hits = 0
        for task in test:
            card = skills.get(task.family, "")
            prompt = (
                "Use the following procedure silently. Do not quote, explain, or repeat it. "
                "Solve the task and output only the requested answer.\n"
                f"PRIVATE PROCEDURE: {card}\n"
            )
            pred = runner.generate(task.prompt, prompt)
            ok = normalize(pred, task.family) == normalize(task.answer, task.family)
            hits += int(ok)
            rows.append({"condition": "skill_retry", "round": rnd, "task_id": task.task_id, "family": task.family, "answer": task.answer, "prediction": pred, "correct": ok})
        metrics.append({"round": rnd, "condition": "skill_retry", "accuracy": hits / len(test), "seconds": time.time() - start})
    with (out / "predictions.jsonl").open("w") as f:
        for row in rows: f.write(json.dumps(row) + "\n")
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round", "condition", "accuracy", "seconds"]); w.writeheader(); w.writerows(metrics)
    (out / "report.md").write_text(
        "# Skill retry\n\n" +
        "This retry deduplicates cards by family and instructs the model not to repeat the procedure. "
        "The base model remains frozen; no training occurs.\n\n" +
        "\n".join(f"- Round {m['round']}: **{m['accuracy']:.3f}**" for m in metrics) + "\n"
    )


if __name__ == "__main__": main()
