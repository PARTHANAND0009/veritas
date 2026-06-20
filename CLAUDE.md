# Veritas — CLAUDE.md

## Quick start

```bash
# Install (editable)
python3 -m pip install -e .

# Fast tests (no model download)
python3 -m pytest tests/ -m "not slow" -v

# Slow tests (requires pythia-1.4b download, ~2.8 GB)
python3 -m pytest tests/ -m slow -v

# All tests
python3 -m pytest tests/ -v

# Lint
python3 -m ruff check veritas/ tests/
python3 -m ruff check --fix veritas/ tests/
```

## CLI

```bash
# Audit a prompt (greedy, default model)
veritas audit --prompt "The capital of Australia is"

# Audit with JSON output
veritas audit --prompt "Marie Curie won the Nobel Prize in" --json --output-dir out/

# Audit LLaMA-2-7b (requires HF access token)
veritas audit --model meta-llama/Llama-2-7b-hf --prompt "..."

# Build eval dataset from TruthfulQA then evaluate
veritas eval --build-dataset --n-items 50 --output-dir out/

# Evaluate on a pre-built JSONL
veritas eval --dataset eval/dataset.jsonl

# Evaluate with calibration (80/20 split)
veritas eval --dataset eval/dataset.jsonl --calibrate
```

## Model flags

| Flag | Model | Notes |
|------|-------|-------|
| *(default)* | `EleutherAI/pythia-1.4b` | Primary target. MPS/CPU. ~2.8 GB download. |
| `--model meta-llama/Llama-2-7b-hf` | LLaMA-2-7B | Requires HF gated access. ~13 GB. |

## Hardware

- Developed on Apple Silicon. No flash-attn.
- **dtype**: float16 on MPS/CUDA (halves RAM: ~2.8 GB for pythia-1.4b), float32 on CPU.
  Set `VERITAS_FLOAT32=1` to force float32 everywhere.
- **Device**: defaults to CPU. TransformerLens warns MPS may produce incorrect results
  on PyTorch ≥ 2.x (TL issue #1178). Set `VERITAS_ALLOW_MPS=1` to opt-in to MPS.
- On 8 GB unified memory machines, use MPS + float16 for viable eval:
  `VERITAS_ALLOW_MPS=1 TRANSFORMERLENS_ALLOW_MPS=1 veritas eval ...`
- pythia-1.4b: ~2.8 GB (float16) / ~5.6 GB (float32). LLaMA-2-7b: ~13 GB (float16).

## Repository layout

```
veritas/
  model.py      # load_model, generate → GenerationResult (residuals per layer)
  signals.py    # extract_features → list[TokenFeatures] (3 signals)
  score.py      # score_tokens, aggregate_to_spans, calibrate
  viz.py        # heatmap, trajectory, layer_agreement, roc_pr plots
  cli.py        # `veritas audit` and `veritas eval`
  data.py       # load_dataset, build_truthfulqa_dataset
eval/
  dataset.jsonl # 50-item labeled set (built by veritas eval --build-dataset)
tests/
  test_model.py   # GenerationResult types, shapes, determinism
  test_signals.py # signal ranges, logit-lens shape, feature counts
  test_score.py   # risk range, span pooling, calibration shape
```

## Key design notes

- Generation: manual autoregressive loop (not `model.generate`). Each step calls
  `run_with_cache` so `resid_post` at every layer is available for every *generated* token.
- Logit lens: `model.ln_final(residual) @ model.W_U + model.b_U`.
- Signal-1 baseline for AUROC comparison uses weights `[-1.5, 0.8, -1.0, -0.8, 0, 0, 0, 0]`
  (only first 4 features active, signals 2+3 zeroed out).
- Unsupported models raise `NotImplementedError` — no silent HF hook fallback.
- Seeds: `torch.manual_seed(seed)` at generation start. Default seed=42.
