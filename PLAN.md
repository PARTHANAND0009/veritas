# Veritas — Implementation Plan

## Environment assumptions
- Hardware: MacBook with Apple Silicon (M-series). Device = `mps` when available, else `cpu`.
- dtype = `torch.float32` throughout. MPS has incomplete float16 support with TransformerLens.
- No flash-attn. No CUDA-specific code in the 1.4B path.
- Python 3.11+, TransformerLens ≥ 2.x, HuggingFace Transformers, datasets, scikit-learn, matplotlib, rich, click.
- pythia-1.4b is confirmed supported by TransformerLens (`HookedTransformer.from_pretrained`).
- LLaMA-2-7b-hf is also supported by TransformerLens but gated behind `--model meta-llama/Llama-2-7b-hf`.

---

## Feature dataclass schema

```python
@dataclass
class TokenFeatures:
    token_id:             int          # vocab index of the chosen token
    token_str:            str          # decoded string
    position:             int          # index in the generated sequence (0-based)

    # Signal 1 — final-layer confidence
    max_prob:             float        # max softmax prob
    entropy:              float        # Shannon entropy of final distribution (nats)
    margin:               float        # top1_prob - top2_prob
    log_prob_chosen:      float        # log P(chosen token)

    # Signal 2 — layer-wise agreement
    layer_agreement:      float        # fraction of last K layers whose top-1 == chosen

    # Signal 3 — trajectory stability
    crystallization_depth: float       # first-layer-as-top1 / n_layers, in [0, 1]
    volatility_count:     int          # number of top-1 flips across all layers
    prob_variance:        float        # variance of chosen-token prob over last K layers

    # Raw arrays preserved for plotting and future analysis
    per_layer_top1:       list[int]    # top-1 token at each layer (len = n_layers)
    per_layer_chosen_prob: list[float] # P(chosen token) at each layer (len = n_layers)
```

**Feature vector** (8 scalars fed to the scorer, in this fixed order):
```
[max_prob, entropy, margin, log_prob_chosen,
 layer_agreement, crystallization_depth, volatility_count, prob_variance]
```

Unsupervised weight signs:
`[-1, +1, -1, -1, -1, +1, +1, +1]`
(high prob / high agreement / early crystallization / low volatility → low risk)

---

## Eval schema

`eval/dataset.jsonl` — one JSON object per line:
```json
{"prompt": "...", "answer": "...", "label": 1}
```
- `label` = 1 if the generation is hallucinated, 0 if correct.
- Built automatically from TruthfulQA via `veritas eval --build-dataset`.
- Loader in `data.py` reads any JSONL matching this schema.

---

## File-by-file plan

### `pyproject.toml`
- `[project]` with `name = "veritas"`, `requires-python = ">=3.11"`.
- Dependencies: `transformer_lens`, `transformers`, `torch`, `datasets`, `scikit-learn`,
  `matplotlib`, `click`, `rich`, `numpy`.
- `[project.scripts]` entry point: `veritas = "veritas.cli:cli"`.
- `[tool.ruff]` with `select = ["E", "F", "I"]`, `line-length = 100`.

---

### `veritas/model.py`

**Responsibility**: load a HookedTransformer and run autoregressive generation while capturing
per-layer residual streams.

**Public API**:
```python
@dataclass
class GenerationResult:
    prompt_tokens:   list[int]
    generated_tokens: list[int]           # token ids, length = max_new_tokens (or until EOS)
    generated_strs:  list[str]            # decoded per-token strings
    # residuals[t] is a tensor of shape [n_layers, d_model] — the resid_post at each layer
    # for the generation step that produced generated_tokens[t]
    residuals:       list[torch.Tensor]
    # final_logits[t] is the final-layer logit vector [vocab_size] before the token was chosen
    final_logits:    list[torch.Tensor]

def load_model(model_name: str, device: str | None = None) -> HookedTransformer: ...
def generate(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    seed: int = 42,
) -> GenerationResult: ...
```

**Implementation notes**:
- `load_model`: call `HookedTransformer.from_pretrained(model_name, dtype=torch.float32)`.
  Then `.to(device)`. Auto-detect device: `"mps"` if `torch.backends.mps.is_available()`, else `"cpu"`.
- `generate`: manual autoregressive loop (not `model.generate`). At each step call
  `model.run_with_cache(current_tokens)`, extract `cache["resid_post", i]` for all layers i,
  take the last-position slice `[:, -1, :]`, stack into `[n_layers, d_model]`. Take `logits[:, -1, :]`
  as `final_logits[t]`. Apply temperature then argmax (greedy if temp=0) or sample.
- Stop at EOS token or `max_new_tokens`.
- Fix seed with `torch.manual_seed(seed)`.
- LLaMA-2 guard: if `model_name` contains `"llama"` but `HookedTransformer.from_pretrained`
  raises, re-raise with a clear message pointing to the `--model` flag.
- **Not** implemented: raw HF hook fallback. Unsupported models raise `NotImplementedError`
  with a message listing the supported model list.

---

### `veritas/signals.py`

**Responsibility**: given a `GenerationResult` and a loaded model, compute `TokenFeatures`
for every generated token.

**Public API**:
```python
def extract_features(
    model: HookedTransformer,
    result: GenerationResult,
    K: int = 8,
) -> list[TokenFeatures]: ...

def apply_logit_lens(
    model: HookedTransformer,
    residual: torch.Tensor,   # [n_layers, d_model]
) -> torch.Tensor:            # [n_layers, vocab_size]  (logits, not probs)
    ...
```

**Signal 1** (per generated token t):
- Input: `result.final_logits[t]` — shape `[vocab_size]`.
- `probs = softmax(logits)`.
- `max_prob = probs.max().item()`.
- `entropy = -(probs * probs.log().clamp(min=-1e9)).sum().item()`.
- `sorted_p = probs.sort(descending=True).values; margin = (sorted_p[0] - sorted_p[1]).item()`.
- `log_prob_chosen = log_softmax(logits)[chosen_id].item()`.

**Signal 2** (per generated token t):
- Apply `apply_logit_lens` to `result.residuals[t]` → `[n_layers, vocab_size]`.
- `per_layer_top1[i] = logit_lens_logits[i].argmax().item()`.
- `layer_agreement = mean(per_layer_top1[-K:] == chosen_id)`.

**Signal 3** (per generated token t):
- `crystallization_depth`: find smallest layer index `i` where `per_layer_top1[i] == chosen_id`.
  Normalize by `n_layers`. If never top-1, set to 1.0 (worst case).
- `volatility_count`: count transitions `per_layer_top1[i] != per_layer_top1[i-1]` for i in 1..n_layers-1.
- `per_layer_chosen_prob[i] = softmax(logit_lens_logits[i])[chosen_id].item()`.
- `prob_variance = variance of per_layer_chosen_prob[-K:]`.

**`apply_logit_lens`**:
```python
# residual: [n_layers, d_model]
normed = model.ln_final(residual)       # [n_layers, d_model]
logits = normed @ model.W_U + model.b_U # [n_layers, vocab_size]
```
(TransformerLens exposes `model.W_U: [d_model, vocab]` and `model.b_U: [vocab]`.)

---

### `veritas/score.py`

**Responsibility**: convert feature lists to per-token risk scores and span-level aggregations.

**Public API**:
```python
FEATURE_NAMES: tuple[str, ...] = (
    "max_prob", "entropy", "margin", "log_prob_chosen",
    "layer_agreement", "crystallization_depth", "volatility_count", "prob_variance",
)

DEFAULT_WEIGHTS = np.array([-1.5, 0.8, -1.0, -0.8, -1.2, 0.6, 0.4, 0.5])

@dataclass
class TokenRisk:
    token_str: str
    position:  int
    risk:      float           # sigmoid output, ∈ (0, 1)
    features:  TokenFeatures

@dataclass
class SpanRisk:
    span_text: str
    token_risks: list[TokenRisk]
    risk_max:   float
    risk_mean:  float

def features_to_vector(f: TokenFeatures) -> np.ndarray: ...   # returns length-8 array

def score_tokens(
    features: list[TokenFeatures],
    weights: np.ndarray | None = None,   # None → DEFAULT_WEIGHTS
) -> list[TokenRisk]: ...

def aggregate_to_spans(token_risks: list[TokenRisk]) -> list[SpanRisk]: ...
# Groups by whitespace words: accumulate tokens until a token_str ends with a whitespace-
# prefixed boundary (SentencePiece "▁" prefix) or starts a new word.

def calibrate(
    feature_vectors: np.ndarray,   # [N, 8]
    labels: np.ndarray,            # [N] binary
) -> np.ndarray:                   # weight vector length 8
    # Fit sklearn LogisticRegression, return coef_.flatten()
    ...
```

**Scoring**:
- Normalize each feature by a fixed scale factor (stored as `FEATURE_SCALES`) so the
  dot product is well-conditioned regardless of raw units.
- `risk = sigmoid(weights @ normalize(feature_vector))`.
- Log the weight vector used at `DEBUG` level.

---

### `veritas/viz.py`

**Responsibility**: produce the three diagnostic plots plus the ROC/PR eval plot.

**Public API**:
```python
def plot_token_heatmap(
    token_risks: list[TokenRisk],
    save_path: Path,
    title: str = "Token Risk Heatmap",
) -> None: ...

def plot_trajectory(
    features: list[TokenFeatures],
    token_idx: int,
    n_layers: int,
    save_path: Path,
) -> None: ...

def plot_layer_agreement(
    features: list[TokenFeatures],
    save_path: Path,
) -> None: ...

def plot_roc_pr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Path,
) -> dict[str, float]:   # returns {"auroc": ..., "ap": ...}
    ...
```

- Heatmap: horizontal bar of colored cells, one per token, labeled with the token string,
  color-mapped from green (low risk) to red (high risk).
- Trajectory: line plot of `per_layer_chosen_prob` across layers for one chosen token,
  with a vertical dashed line at `crystallization_depth * n_layers`.
- Agreement bar chart: one bar per generated position, height = `layer_agreement`.
- ROC/PR: two-panel figure, prints AUROC and AP to stdout.

---

### `veritas/data.py`

**Responsibility**: build and load the eval dataset.

**Public API**:
```python
@dataclass
class EvalItem:
    prompt:  str
    answer:  str
    label:   int    # 1 = hallucinated, 0 = correct

def load_dataset(path: Path) -> list[EvalItem]: ...

def build_truthfulqa_dataset(
    model: HookedTransformer,
    n_items: int = 50,
    output_path: Path = Path("eval/dataset.jsonl"),
    seed: int = 42,
) -> list[EvalItem]: ...
```

- `build_truthfulqa_dataset`: load `"truthful_qa"` split `"generation"` from HF datasets.
  Sample `n_items` questions. For each: run `generate()` with the model, check if any
  correct answer string appears in the generated text (case-insensitive substring). If yes,
  `label = 0`; else `label = 1`. Write JSONL. Return items.
- `load_dataset`: stream JSONL, parse each line, return `list[EvalItem]`.

---

### `veritas/cli.py`

**Responsibility**: CLI entry points.

```
veritas audit
  --model          model name or HF path      [default: EleutherAI/pythia-1.4b]
  --prompt         quoted string              [required]
  --max-tokens     int                        [default: 200]
  --temperature    float                      [default: 0.0]
  --k              int                        [default: 8]
  --output-dir     directory for PNGs         [default: ./veritas_output]
  --json           also write JSON of features+scores
  --weights        path to .npy weight file   [optional, for calibrated mode]

veritas eval
  --dataset        path to JSONL              [default: eval/dataset.jsonl]
  --model          model name                 [default: EleutherAI/pythia-1.4b]
  --build-dataset  flag: generate dataset first, then eval
  --calibrate      flag: fit weights on first 80% of dataset, eval on last 20%
  --output-dir     directory for plots
```

Output of `veritas audit`: rich table with span text and risk score, then prints path to saved plots.

Output of `veritas eval`:
```
AUROC (full Veritas):     0.XXX
AUROC (Signal-1 only):    0.XXX
Delta:                   +0.XXX
AP (full):                0.XXX
```

---

### `veritas/__init__.py`

Re-export: `load_model`, `generate`, `extract_features`, `score_tokens`, `aggregate_to_spans`.

---

## Test list

### `tests/test_model.py`
1. `test_load_model_pythia` — `load_model("EleutherAI/pythia-1.4b")` returns a `HookedTransformer` without error. (Marks: `slow`)
2. `test_generate_returns_correct_types` — `GenerationResult` fields have expected Python types.
3. `test_residuals_shape` — `result.residuals[t]` has shape `[n_layers, d_model]` for all t.
4. `test_generate_greedy_deterministic` — two calls with same seed return identical token sequences.
5. `test_unsupported_model_raises` — `load_model("gpt2-xl")` (if not in TL's list) raises `NotImplementedError`.

### `tests/test_signals.py`
1. `test_apply_logit_lens_shape` — output shape is `[n_layers, vocab_size]`.
2. `test_signal1_ranges` — `max_prob ∈ (0,1)`, `entropy ≥ 0`, `margin ∈ (0,1)`, `log_prob_chosen ≤ 0`.
3. `test_signal2_agreement_range` — `layer_agreement ∈ [0, 1]`.
4. `test_signal3_crystallization_range` — `crystallization_depth ∈ [0, 1]`.
5. `test_signal3_volatility_nonneg` — `volatility_count ≥ 0`.
6. `test_features_length_matches_tokens` — `len(features) == len(result.generated_tokens)`.
7. `test_per_layer_arrays_length` — `len(f.per_layer_top1) == n_layers` for every feature.

### `tests/test_score.py`
1. `test_risk_score_range` — `TokenRisk.risk ∈ (0, 1)` for all outputs.
2. `test_span_max_ge_mean` — `span.risk_max >= span.risk_mean` always.
3. `test_span_count_matches_words` — number of spans equals number of whitespace-delimited words.
4. `test_default_weights_length` — `DEFAULT_WEIGHTS` has length 8 matching `FEATURE_NAMES`.
5. `test_calibrate_returns_correct_shape` — `calibrate(X, y)` returns array of length 8.
6. `test_features_to_vector_length` — `len(features_to_vector(f)) == 8`.

---

## Guesses / open questions

1. **MPS + TransformerLens**: TL's `run_with_cache` may not be fully tested on MPS. Fallback:
   if MPS fails at runtime, retry on CPU with a warning. Will discover in `test_model.py`.

2. **Logit lens exact API**: assuming `model.ln_final` is a callable `nn.Module` and
   `model.W_U`, `model.b_U` are directly accessible parameter attributes. This is true in TL ≥ 1.x
   but I will verify in the first test and adjust if the attribute names differ.

3. **TruthfulQA label quality**: substring-match labeling will have false negatives (paraphrased
   correct answers). Acceptable for a ~50-item bootstrap; document in README.

4. **Pythia-1.4b EOS token**: pythia uses `<|endoftext|>` (token 0 in GPT-2 tokenizer).
   I will stop generation there. Confirm in `test_generate_returns_correct_types`.

5. **DEFAULT_WEIGHTS**: the values I chose are reasonable but untested. Calibration mode
   (`--calibrate`) is the path to better weights. The unsupervised defaults are just for
   the zero-label case.

---

## Build order

```
Step 1  pyproject.toml + package skeleton (empty modules + __init__)
Step 2  tests/test_model.py  →  veritas/model.py  (mark slow tests)
Step 3  tests/test_signals.py  →  veritas/signals.py
Step 4  tests/test_score.py  →  veritas/score.py
Step 5  veritas/viz.py  (no tests needed; visual output)
Step 6  veritas/data.py  (TruthfulQA builder)
Step 7  veritas/cli.py  (ties everything together)
Step 8  CLAUDE.md + README.md
Step 9  veritas eval  →  print AUROC delta  →  commit
```

Commit at each green milestone.
