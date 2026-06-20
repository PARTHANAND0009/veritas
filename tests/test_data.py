from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from veritas.data import EvalItem, load_dataset, semantic_label


def test_load_dataset_roundtrip(tmp_path: Path) -> None:
    items_in = [
        EvalItem("Q: A?\nA:", "A answer", 0),
        EvalItem("Q: B?\nA:", "B answer", 1),
        EvalItem("Q: C?\nA:", "C answer", 0),
    ]
    path = tmp_path / "test.jsonl"
    with open(path, "w") as f:
        for it in items_in:
            f.write(json.dumps({"prompt": it.prompt, "answer": it.answer, "label": it.label}) + "\n")

    out = load_dataset(path)
    assert len(out) == 3
    for a, b in zip(items_in, out):
        assert a.prompt == b.prompt and a.label == b.label


def test_semantic_label_correct() -> None:
    pytest.importorskip("sentence_transformers")
    label, strategy = semantic_label("Paris is the capital of France", ["Paris"])
    assert label == 0 and strategy == "semantic"


def test_semantic_label_hallucinated() -> None:
    pytest.importorskip("sentence_transformers")
    label, _ = semantic_label("The moon is made of green cheese", ["Paris"])
    assert label == 1


def test_substring_label_fallback_on_importerror(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    import veritas.data as dm
    monkeypatch.setattr(dm, "_ST_MODEL_CACHE", {})
    label, strategy = semantic_label("The answer is Paris", ["Paris"])
    assert label == 0 and strategy == "substring"
    assert "sentence-transformers" in capsys.readouterr().out.lower()


def test_append_mode_skips_existing(tmp_path: Path) -> None:
    from veritas.data import build_truthfulqa_dataset

    mock_ds = [
        {"question": "Capital of France?", "correct_answers": ["Paris"]},
        {"question": "Who wrote Hamlet?", "correct_answers": ["Shakespeare"]},
        {"question": "Boiling point of water?", "correct_answers": ["100 degrees"]},
        {"question": "Speed of light units?", "correct_answers": ["metres per second"]},
        {"question": "Chemical symbol of gold?", "correct_answers": ["Au"]},
    ]

    out = tmp_path / "dataset.jsonl"
    with open(out, "w") as f:
        for i in range(3):
            f.write(json.dumps({
                "prompt": f"Q: {mock_ds[i]['question']}\nA:",
                "answer": mock_ds[i]["correct_answers"][0],
                "label": 0,
            }) + "\n")

    mock_result = MagicMock()
    mock_result.generated_strs = ["some answer"]

    with (
        patch("datasets.load_dataset", return_value=mock_ds),
        patch("veritas.data.generate", return_value=mock_result),
    ):
        items = build_truthfulqa_dataset(MagicMock(), n_items=5, output_path=out, seed=42)

    assert len(items) == 5
    assert len([ln for ln in out.read_text().strip().split("\n") if ln]) == 5
