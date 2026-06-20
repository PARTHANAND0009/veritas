from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformer_lens import HookedTransformer

from veritas.model import generate

_ST_MODEL_CACHE: dict[str, Any] = {}


@dataclass
class EvalItem:
    prompt: str
    answer: str
    label: int  # 0 = correct, 1 = hallucinated


def load_dataset(path: Path) -> list[EvalItem]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items.append(EvalItem(prompt=obj["prompt"], answer=obj["answer"], label=obj["label"]))
    return items


def _substring_label(generation: str, correct_answers: list[str]) -> int:
    gen_lower = generation.lower()
    return 0 if any(ans.lower() in gen_lower for ans in correct_answers if ans) else 1


def semantic_label(
    generation: str,
    correct_answers: list[str],
    model_name: str = "paraphrase-MiniLM-L6-v2",
) -> tuple[int, str]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore[import]
        from sentence_transformers import util as st_util  # type: ignore[import]

        if model_name not in _ST_MODEL_CACHE:
            _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        st = _ST_MODEL_CACHE[model_name]

        gen_emb = st.encode(generation, convert_to_tensor=True)
        for ans in correct_answers:
            if ans and float(st_util.cos_sim(gen_emb, st.encode(ans, convert_to_tensor=True)).item()) >= 0.65:
                return 0, "semantic"
        return 1, "semantic"

    except ImportError:
        print("Warning: sentence-transformers not available, falling back to substring match.", flush=True)
        return _substring_label(generation, correct_answers), "substring"


def build_truthfulqa_dataset(
    model: HookedTransformer,
    n_items: int = 50,
    output_path: Path = Path("eval/dataset.jsonl"),
    seed: int = 42,
    max_new_tokens: int = 80,
    label_strategy: str = "substring",
) -> list[EvalItem]:
    from datasets import load_dataset as hf_load

    random.seed(seed)
    ds = hf_load("truthfulqa/truthful_qa", "generation", split="validation")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    

    
    existing: set[str] = set()
    items: list[EvalItem] = []
    if output_path.exists() and output_path.stat().st_size > 0:
        items = load_dataset(output_path)
        existing = {it.prompt for it in items}

    indices = random.sample(range(len(ds)), min(n_items, len(ds)))

    with open(output_path, "a") as f:
        for idx in indices:
            row = ds[idx]
            question: str = row["question"]
            correct_answers: list[str] = row["correct_answers"]
            prompt = f"Q: {question}\nA:"

            if prompt in existing:
                continue

            print(f"  [{len(items) + 1}/{n_items}] Q: {question[:60]}", flush=True)
            result = generate(
                model, prompt, max_new_tokens=max_new_tokens, temperature=0.0,
                seed=seed, capture_residuals=False,
            )
            gen_text = "".join(result.generated_strs)

            if label_strategy == "semantic":
                label, strategy = semantic_label(gen_text, correct_answers)
            else:
                label = _substring_label(gen_text, correct_answers)
                strategy = "substring"

            best_answer = correct_answers[0] if correct_answers else ""
            item = EvalItem(prompt=prompt, answer=best_answer, label=label)
            items.append(item)
            existing.add(prompt)

            f.write(json.dumps({"prompt": prompt, "answer": best_answer, "label": label}) + "\n")
            print(f"       label={label} ({strategy})  gen={gen_text[:60]!r}", flush=True)

    return items[:n_items]
