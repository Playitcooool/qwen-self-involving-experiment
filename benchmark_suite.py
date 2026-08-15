"""Held-out multi-type benchmark for the saved baseline/SFT/GRPO/skill systems."""
from __future__ import annotations

import ast, csv, gc, json, random, re, time
from dataclasses import dataclass
from pathlib import Path

import torch
from experiment import Runner


@dataclass
class BTask:
    task_id: str
    kind: str
    prompt: str
    answer: str
    tests: list | None = None


def make_math(i: int, seed: int) -> BTask:
    r = random.Random(seed * 1009 + i)
    a, b, groups = r.randint(3, 19), r.randint(2, 12), r.randint(2, 6)
    if i % 3 == 0:
        prompt = f"A library has {a} fiction books and {b} science books. It receives {groups} identical boxes, each containing that many books as the current collection. How many books arrive in total? Return only the integer."
    elif i % 3 == 1:
        prompt = f"A workshop makes {a} blue parts and {b} red parts each morning for {groups} mornings. How many parts are made altogether? Give only the integer."
    else:
        prompt = f"There are {groups} teams. Each team has {a} adults and {b} children. How many people are there? Output only the integer."
    return BTask(f"math-{i}", "math", prompt, str((a + b) * groups))


def make_logic(i: int, seed: int) -> BTask:
    r = random.Random(seed * 2027 + i)
    entity = r.choice(["the fox", "the pilot", "the botanist", "the robot"])
    p1, p2, p3 = r.sample(["quiet", "blue", "patient", "alert", "kind"], 3)
    positive = i % 2 == 0
    facts = [f"{entity} is {p1}.", f"If something is {p1}, then it is {p2}."]
    if positive:
        query = f"{entity} is {p2}."
        answer = "yes"
    else:
        query = f"{entity} is {p3}."
        answer = "no"
    if positive:
        facts.append(f"If something is {p2}, then it is {p3}.")
    prompt = "Decide whether the query follows from the facts and rules. Answer only yes or no.\nFacts and rules:\n- " + "\n- ".join(facts) + f"\nQuery: {query}"
    return BTask(f"logic-{i}", "logic", prompt, answer)


def make_code(i: int, seed: int) -> BTask:
    cases = [
        ("sum_positive", "Return the sum of all positive integers in nums.", "def sum_positive(nums):", [([1, -2, 4], 5), ([-3, 0], 0)]),
        ("count_vowels", "Return the number of vowel letters in text, counting a, e, i, o, u in either case.", "def count_vowels(text):", [("Apple", 2), ("sky", 0)]),
        ("reverse_words", "Return the words in text in reverse order, separated by one space.", "def reverse_words(text):", [("one two three", "three two one"), ("hi", "hi")]),
        ("clamp", "Return x limited to the inclusive range [low, high].", "def clamp(x, low, high):", [(5, 0, 3, 3), (-1, 0, 3, 0)]),
    ]
    name, desc, sig, tests = cases[i % len(cases)]
    prompt = f"Write only a Python function. {desc} The required signature is `{sig}` Do not include markdown or explanation."
    return BTask(f"code-{i}", "code", prompt, name, tests)


def benchmark(n_per_kind: int = 12, seed: int = 31415) -> list[BTask]:
    return ([make_math(i, seed) for i in range(n_per_kind)] +
            [make_logic(i, seed) for i in range(n_per_kind)] +
            [make_code(i, seed) for i in range(n_per_kind)])


def text_answer(pred: str, task: BTask) -> bool:
    if task.kind in ("math", "logic"):
        vals = re.findall(r"-?\d+", pred) if task.kind == "math" else re.findall(r"\b(?:yes|no)\b", pred.lower())
        return bool(vals) and vals[-1].lower() == task.answer.lower()
    return verify_code(pred, task)


def verify_code(pred: str, task: BTask) -> bool:
    code = pred
    match = re.search(r"```(?:python)?\s*(.*?)```", pred, re.S | re.I)
    if match: code = match.group(1)
    start = code.find("def ")
    if start >= 0: code = code[start:]
    try:
        tree = ast.parse(code)
        scope = {}
        exec(compile(tree, "<candidate>", "exec"), {"__builtins__": {"sum": sum, "len": len, "str": str, "max": max, "min": min, "lower": str.lower}}, scope)
        name = task.answer
        fn = scope.get(name)
        if not callable(fn): return False
        if name == "sum_positive": return all(fn(args) == expected for args, expected in task.tests)
        if name == "count_vowels": return all(fn(args) == expected for args, expected in task.tests)
        if name == "reverse_words": return all(fn(args) == expected for args, expected in task.tests)
        if name == "clamp": return all(fn(*args[:-1]) == args[-1] for args in task.tests)
    except Exception:
        return False
    return False


SKILLS = {
    "math": "Translate the story into quantities, identify the operation, calculate step by step, then output only the requested number.",
    "logic": "List the explicit facts, apply rules forward one step at a time, and answer yes only when the query is derivable.",
    "code": "Identify the exact signature and edge cases, write the smallest correct function, and output only executable Python code.",
}


def run_condition(name: str, runner: Runner, tasks: list[BTask], out: Path):
    rows = []; start = time.time()
    for task in tasks:
        prefix = ""
        if name == "skill":
            prefix = "Use this procedure silently. Do not repeat it or explain it.\nPRIVATE PROCEDURE: " + SKILLS[task.kind] + "\n"
        pred = runner.generate(task.prompt, prefix)
        rows.append({"condition": name, "task_id": task.task_id, "kind": task.kind, "prediction": pred, "correct": text_answer(pred, task)})
    return rows, time.time() - start


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--model-path", required=True); ap.add_argument("--output", default="outputs/benchmark_suite"); ap.add_argument("--n-per-kind", type=int, default=12); ap.add_argument("--seed", type=int, default=31415)
    a = ap.parse_args(); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    tasks = benchmark(a.n_per_kind, a.seed)
    task_file = out / "tasks.jsonl"
    task_file.write_text("\n".join(json.dumps(t.__dict__) for t in tasks) + "\n")
    conditions = [("baseline", None), ("sft", Path("outputs/adapters/round_5")), ("grpo", Path("outputs/grpo/adapters/round_5")), ("skill", None)]
    all_rows = []; metrics = []
    for name, adapter in conditions:
        runner = Runner(a.model_path, str(adapter) if adapter else None)
        rows, seconds = run_condition(name, runner, tasks, out); all_rows.extend(rows)
        metrics.append({"condition": name, "accuracy": sum(r["correct"] for r in rows) / len(rows), "seconds": seconds})
        for kind in ["math", "logic", "code"]:
            sub = [r for r in rows if r["kind"] == kind]
            metrics.append({"condition": name + ":" + kind, "accuracy": sum(r["correct"] for r in sub) / len(sub), "seconds": seconds})
        del runner; gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
    with (out / "predictions.jsonl").open("w") as f:
        for row in all_rows: f.write(json.dumps(row) + "\n")
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "accuracy", "seconds"]); w.writeheader(); w.writerows(metrics)
    (out / "report.md").write_text("# Multi-type benchmark\n\n" + "\n".join(f"- **{m['condition']}**: {m['accuracy']:.3f}" for m in metrics) + "\n\nTask types: held-out word-problem math, rule reasoning, and executable Python functions.\n")


if __name__ == "__main__": main()
