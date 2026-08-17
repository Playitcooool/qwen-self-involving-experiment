from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

from experiment import Runner
from reproducibility import content_sha256


WORDS = re.compile(r"[a-zA-Z]{3,}")
STOP = {"the", "and", "for", "with", "this", "that", "then", "from", "your", "answer", "task", "solve", "write"}
NUMBER_WORDS = re.compile(r"\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)\b", re.I)


def terms(text: str) -> set[str]:
    return {word.lower() for word in WORDS.findall(text) if word.lower() not in STOP}


def sanitize_procedure(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"[-+]?\d[\d,]*(?:\.\d+)?", "<number>", text)
    text = NUMBER_WORDS.sub("<number>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:600]


def build_cards(model_path: str, experience_path: Path, output: Path, max_cards: int) -> list[dict]:
    experiences = [json.loads(line) for line in experience_path.read_text().splitlines() if line.strip()]
    runner = Runner(model_path)
    cards = []
    seen = set()
    for item in experiences:
        if len(cards) >= max_cards:
            break
        trajectory_lines = [line for line in item["trajectory"].splitlines() if line.strip()]
        # The verifier-approved final line is deliberately removed before the
        # model sees the trajectory, and all remaining quantities are redacted.
        redacted_trajectory = sanitize_procedure("\n".join(trajectory_lines[:-1]))
        redacted_problem = sanitize_procedure(item["prompt"])
        prompt = (
            "A model solved the following training problem correctly. Summarize the reusable solving procedure, "
            "not the specific problem or result. Do not copy names, quantities, or the final answer. Return one concise procedure only.\n\n"
            f"REDACTED TRAINING PROBLEM:\n{redacted_problem}\n\nREDACTED MODEL TRAJECTORY WITHOUT FINAL LINE:\n{redacted_trajectory}"
        )
        procedure = sanitize_procedure(runner.generate(prompt, max_new_tokens=128))
        if not procedure or procedure in seen:
            continue
        seen.add(procedure)
        cards.append({
            "source_task_id": item["task_id"],
            "source_prompt": item["prompt"],
            "source_trajectory_sha256": content_sha256(item["trajectory"]),
            "procedure": procedure,
            "uses_reference_answer": False,
            "final_trajectory_line_removed_before_summary": True,
            "quantities_redacted_before_and_after_summary": True,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"method": "base-model summary of verified-successful train rollouts", "cards": cards}, indent=2) + "\n")
    return cards


def load_cards(path: Path) -> list[dict]:
    return json.loads(path.read_text())["cards"]


def retrieve(cards: list[dict], prompt: str, top_k: int = 3) -> list[dict]:
    query = terms(prompt)
    scored = []
    for index, card in enumerate(cards):
        source = terms(card["source_prompt"])
        score = len(query & source) / math.sqrt(max(1, len(query)) * max(1, len(source)))
        scored.append((score, -index, card))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [card for _, _, card in scored[:top_k]]


def skill_prefix(cards: list[dict], prompt: str, top_k: int = 3) -> str:
    selected = retrieve(cards, prompt, top_k)
    procedures = "\n".join(f"- {card['procedure']}" for card in selected)
    return "Use these train-derived procedures silently. Do not quote them in the answer.\n" + procedures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--experience-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-cards", type=int, default=24)
    args = parser.parse_args()
    build_cards(args.model_path, Path(args.experience_bank), Path(args.output), args.max_cards)


if __name__ == "__main__":
    main()
