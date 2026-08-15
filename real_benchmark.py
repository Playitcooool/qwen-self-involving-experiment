"""Evaluate the saved systems on public GSM8K, HumanEval, and ARC-Challenge subsets."""
from __future__ import annotations

import csv, gc, json, random, re, signal, time, urllib.request
from pathlib import Path

import torch
import pandas as pd
from experiment import Runner


DATA_CACHE = Path("data_cache")


def load_parquet(repo: str, path: str):
    DATA_CACHE.mkdir(exist_ok=True)
    local = DATA_CACHE / (repo.replace("/", "__") + "__" + path.replace("/", "__"))
    if not local.exists():
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}?download=true"
        urllib.request.urlretrieve(url, local)
    return pd.read_parquet(local).to_dict(orient="records")


def sample_rows(ds, n: int, seed: int):
    ids = list(range(len(ds))); random.Random(seed).shuffle(ids)
    return [ds[i] for i in ids[:n]]


def gsm_tasks(n: int, seed: int):
    ds = load_parquet("openai/gsm8k", "main/test-00000-of-00001.parquet")
    out = []
    for row in sample_rows(ds, n, seed):
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"benchmark": "gsm8k", "task_id": f"gsm8k:{len(out)}", "prompt": "Solve this grade-school math problem. Show concise reasoning, then write the final numeric answer on the last line.\n\n" + row["question"], "gold": answer})
    return out


def humaneval_tasks(n: int, seed: int):
    ds = load_parquet("openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet")
    out = []
    for row in sample_rows(ds, n, seed):
        out.append({"benchmark": "humaneval", "task_id": row["task_id"], "prompt": row["prompt"] + "\n\nComplete the function. Output only valid Python code.", "stub_prompt": row["prompt"], "gold": row["entry_point"], "test": row["test"], "entry_point": row["entry_point"]})
    return out


def arc_tasks(n: int, seed: int):
    ds = load_parquet("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet")
    out = []
    for row in sample_rows(ds, n, seed):
        choices = row["choices"]
        rendered = "\n".join(f"{label}. {text}" for label, text in zip(choices["label"], choices["text"]))
        out.append({"benchmark": "arc_challenge", "task_id": f"arc:{len(out)}", "prompt": "Answer this multiple-choice science question. Return only the option letter.\n\nQuestion: " + row["question"] + "\n" + rendered, "gold": row["answerKey"]})
    return out


def last_number(text: str):
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else ""


def check_gsm(pred: str, gold: str):
    return last_number(pred) == gold


def check_arc(pred: str, gold: str):
    letters = re.findall(r"\b([A-E])\b", pred.upper())
    return bool(letters) and letters[-1] == gold.upper()


class Timeout(Exception): pass


def timeout_handler(signum, frame): raise Timeout()


def check_humaneval(pred: str, row: dict):
    match = re.search(r"```(?:python)?[ \t]*\n?(.*?)```", pred, re.S | re.I)
    code = match.group(1) if match else pred
    if "def " not in code: code = row["stub_prompt"] + "\n" + code
    start = code.find("def ")
    if start > 0 and "from " not in code[:start]: code = code[start:]
    try:
        scope = {}
        signal.signal(signal.SIGALRM, timeout_handler); signal.alarm(4)
        exec(code, scope, scope)
        fn = scope[row["entry_point"]]
        exec(row["test"], scope, scope)
        scope["check"](fn)
        signal.alarm(0)
        return True
    except Exception:
        signal.alarm(0)
        return False


SKILLS = {
    "gsm8k": "Translate the story into quantities, solve step by step, check the arithmetic, and put only the final number on the last line.",
    "humaneval": "Follow the exact function signature, implement the simplest correct behavior, handle edge cases, and output executable Python only.",
    "arc_challenge": "Read the question carefully, compare each option against the evidence, and output only the correct option letter.",
}


def evaluate_condition(name: str, runner: Runner, tasks: list[dict]):
    rows = []; start = time.time()
    for row in tasks:
        prefix = ""
        if name == "skill": prefix = "Use this procedure silently; do not repeat or explain it.\nPRIVATE PROCEDURE: " + SKILLS[row["benchmark"]] + "\n"
        limit = 384 if row["benchmark"] == "humaneval" else (256 if row["benchmark"] == "gsm8k" else 64)
        pred = runner.generate(row["prompt"], prefix, max_new_tokens=limit)
        if row["benchmark"] == "gsm8k": ok = check_gsm(pred, row["gold"])
        elif row["benchmark"] == "arc_challenge": ok = check_arc(pred, row["gold"])
        else: ok = check_humaneval(pred, row)
        rows.append({"condition": name, "benchmark": row["benchmark"], "task_id": row["task_id"], "correct": ok, "prediction": pred})
    return rows, time.time() - start


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model-path", required=True); ap.add_argument("--output", default="outputs/real_benchmark"); ap.add_argument("--gsm8k", type=int, default=30); ap.add_argument("--humaneval", type=int, default=10); ap.add_argument("--arc", type=int, default=30); ap.add_argument("--seed", type=int, default=20260815)
    a = ap.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    tasks = gsm_tasks(a.gsm8k, a.seed) + humaneval_tasks(a.humaneval, a.seed + 1) + arc_tasks(a.arc, a.seed + 2)
    (out / "manifest.json").write_text(json.dumps({"seed": a.seed, "counts": {k: sum(t["benchmark"] == k for t in tasks) for k in sorted(set(t["benchmark"] for t in tasks))}, "task_ids": [t["task_id"] for t in tasks]}, indent=2))
    conditions = [("baseline", None), ("sft", Path("outputs/adapters/round_5")), ("grpo", Path("outputs/grpo/adapters/round_5")), ("skill", None)]
    all_rows = []; metrics = []
    for name, adapter in conditions:
        runner = Runner(a.model_path, str(adapter) if adapter else None)
        rows, seconds = evaluate_condition(name, runner, tasks); all_rows.extend(rows)
        metrics.append({"condition": name, "benchmark": "all", "accuracy": sum(r["correct"] for r in rows) / len(rows), "seconds": seconds})
        for bench in sorted(set(t["benchmark"] for t in tasks)):
            sub = [r for r in rows if r["benchmark"] == bench]
            metrics.append({"condition": name, "benchmark": bench, "accuracy": sum(r["correct"] for r in sub) / len(sub), "seconds": seconds})
        del runner; gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
    with (out / "predictions.jsonl").open("w") as f:
        for row in all_rows: f.write(json.dumps(row) + "\n")
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "benchmark", "accuracy", "seconds"]); w.writeheader(); w.writerows(metrics)
    (out / "report.md").write_text("# Real benchmark evaluation\n\n" + "\n".join(f"- **{m['condition']} / {m['benchmark']}**: {m['accuracy']:.3f}" for m in metrics) + "\n\nGold answers were used only inside benchmark verifiers.\n")


if __name__ == "__main__": main()
