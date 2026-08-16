from __future__ import annotations

import argparse, gc, json, os, random, re, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, get_peft_model


FAMILIES = ("arithmetic", "filter", "transform", "json")


@dataclass
class Task:
    task_id: str
    family: str
    prompt: str
    answer: str
    skill: str


def make_task(family: str, i: int, split: str, seed: int) -> Task:
    r = random.Random((seed + 17) * 100000 + i * 31 + FAMILIES.index(family))
    if family == "arithmetic":
        a, b, c = r.randint(2, 19), r.randint(2, 19), r.randint(1, 9)
        answer = str((a + b) * c - b)
        prompt = f"Compute exactly. Return only the integer. ({a} + {b}) * {c} - {b} = ?"
        skill = "For nested arithmetic, compute the parentheses first, then multiply, then subtract."
    elif family == "filter":
        xs = [r.randint(1, 30) for _ in range(7)]
        threshold = r.randint(8, 22)
        answer = ",".join(str(x) for x in xs if x >= threshold)
        prompt = f"From this list {xs}, keep numbers greater than or equal to {threshold}. Return them in original order, comma-separated, with no extra text."
        skill = "For filtering, scan every element once, preserve original order, and emit only qualifying values."
    elif family == "transform":
        words = [r.choice(["amber", "cedar", "linen", "orbit", "pearl", "violet"]) for _ in range(3)]
        answer = "|".join(w[::-1] for w in words)
        prompt = f"Reverse each word and join the results with |. Return only the result. Words: {' '.join(words)}"
        skill = "For per-item transformation, transform each item independently, then join using exactly the requested delimiter."
    else:
        name = r.choice(["Ada", "Lin", "Mika", "Noor"])
        age = r.randint(18, 63)
        city = r.choice(["Osaka", "Lima", "Dakar", "Riga"])
        answer = json.dumps({"name": name, "age": age, "city": city}, separators=(",", ":"))
        prompt = f"Return exactly one compact JSON object with keys name, age, city and these values: name={name}, age={age}, city={city}. No markdown."
        skill = "For constrained JSON, copy the requested keys exactly, use valid JSON double quotes, and omit commentary."
    return Task(f"{split}-{family}-{i}", family, prompt, answer, skill)


def dataset(split: str, n: int, seed: int) -> list[Task]:
    return [make_task(f, i, split, seed) for f in FAMILIES for i in range(n)]


class Runner:
    def __init__(self, model_path: str, adapter: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        base = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True)
        base.to(self.device)
        if adapter:
            base = PeftModel.from_pretrained(base, adapter)
        self.model = base.eval()

    def generate(self, prompt: str, prefix: str = "", max_new_tokens: int = 64) -> str:
        text = (prefix + "\n" if prefix else "") + prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def normalize(x: str, family: str | None = None) -> str:
    x = x.strip().replace("```json", "").replace("```", "").strip()
    if family == "gsm8k":
        # GSM8K answers are checked from the final numeric value.  The
        # reasoning trace is allowed in the rollout, but only its final
        # number is used by the verifier.
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", x)
        return nums[-1].replace(",", "") if nums else ""
    if family == "arithmetic":
        nums = re.findall(r"(?<!\d)-?\d+(?!\d)", x)
        return nums[-1] if nums else ""
    if family == "filter":
        groups = re.findall(r"\d+(?:,\d+)+", x)
        return groups[-1] if groups else ""
    if family == "transform":
        groups = re.findall(r"[a-z]+\|[a-z]+\|[a-z]+", x.lower())
        return groups[-1] if groups else ""
    if family == "json":
        blobs = re.findall(r"\{[^{}]+\}", x)
        for blob in reversed(blobs):
            try:
                obj = json.loads(blob)
                return json.dumps(obj, sort_keys=True, separators=(",", ":"))
            except json.JSONDecodeError:
                pass
        return ""
    return re.sub(r"\s+", "", x)


def evaluate(runner: Runner, tasks: list[Task], prefix_fn=None, condition="baseline", round_no=0, out=None):
    rows, hits = [], 0
    for t in tasks:
        pred = runner.generate(t.prompt, prefix_fn(t) if prefix_fn else "")
        ok = normalize(pred, t.family) == normalize(t.answer, t.family)
        hits += int(ok)
        rows.append({"condition": condition, "round": round_no, "task_id": t.task_id, "family": t.family, "answer": t.answer, "prediction": pred, "correct": ok})
    if out:
        with out.open("a") as f:
            for row in rows: f.write(json.dumps(row) + "\n")
    return hits / len(tasks), rows


def self_involve(runner: Runner, tasks: list[Task], out: Path):
    cards, examples = [], []
    for t in tasks:
        pred = runner.generate(t.prompt)
        if normalize(pred, t.family) == normalize(t.answer, t.family):
            continue
        reflection = runner.generate(f"Task: {t.prompt}\nYour answer: {pred}\nVerifier answer: {t.answer}\nExplain the specific error and give a short reusable procedure.")
        cards.append({"family": t.family, "skill": t.skill, "reflection": reflection})
        examples.append({"prompt": t.prompt, "answer": t.answer})
    out.write_text(json.dumps(cards, indent=2))
    return cards, examples


def train_lora(model_path: str, examples: list[dict[str, str]], out_dir: Path):
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", trust_remote_code=True).to(device)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = [tok(f"User: {e['prompt']}\nAssistant: {e['answer']}", truncation=True, max_length=256) for e in rows]
        def __len__(self): return len(self.rows)
        def __getitem__(self, idx): return self.rows[idx]
    ds = TinyDataset(examples)
    args = TrainingArguments(output_dir=str(out_dir), per_device_train_batch_size=2, gradient_accumulation_steps=2, num_train_epochs=2, learning_rate=2e-4, logging_steps=10, save_strategy="no", report_to=[])
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train(); model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
    del trainer, model
    gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True); ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--train-per-family", type=int, default=8); ap.add_argument("--test-per-family", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--output", default="outputs")
    a = ap.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    out = Path(a.output); out.mkdir(exist_ok=True); (out / "adapters").mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    (out / "config.json").write_text(json.dumps({**vars(a), "torch": torch.__version__, "device": device}, indent=2))
    pred_file = out / "predictions.jsonl"
    test = dataset("test", a.test_per_family, a.seed + 10000)
    skills: list[dict[str, Any]] = []
    adapter = None
    metrics = []
    for rnd in range(a.rounds + 1):
        start = time.time(); runner = Runner(a.model_path, adapter)
        skill_prefix = lambda t: "\n".join(s["skill"] for s in skills if s["family"] == t.family)[-1800:]
        conditions = [("baseline", None)] if rnd == 0 else [("skill", skill_prefix), ("training", None)]
        for condition, fn in conditions:
            acc, rows = evaluate(runner, test, fn, condition, rnd, pred_file)
            metrics.append({"round": rnd, "condition": condition, "accuracy": acc, "seconds": time.time() - start})
        if rnd == a.rounds: break
        explore = dataset(f"r{rnd}", a.train_per_family, a.seed + rnd)
        cards, examples = self_involve(runner, explore, out / f"skills_round_{rnd+1}.json")
        skills.extend(cards)
        if examples:
            adapter_dir = out / "adapters" / f"round_{rnd+1}"
            train_lora(a.model_path, examples, adapter_dir); adapter = str(adapter_dir)
        del runner
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
    import csv
    with (out / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["round", "condition", "accuracy", "seconds"]); w.writeheader(); w.writerows(metrics)
    (out / "report.md").write_text("# Experiment report\n\n" + "\n".join(f"- Round {m['round']} **{m['condition']}**: {m['accuracy']:.3f}" for m in metrics) + "\n\nThis comparison uses offline LoRA rounds; no evaluation-time gradient updates occur.\n")


if __name__ == "__main__": main()
