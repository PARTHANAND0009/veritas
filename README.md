# Veritas

basically i got tired of not knowing when an AI is making stuff up so i built a tool that looks *inside* the model while it's generating and figures out which tokens it's uncertain about

not vibes-based. actual math. reads the residual stream at every layer.

---

## what it does

you give it a prompt, it runs the model, and instead of just showing you the output it shows you a heatmap of which words the model was confident about vs which ones it was basically guessing

like if you ask "the capital of Australia is" and the model says "a city of contrasts" — veritas will flag "city" at 0.85 risk before you even have to google it

it also plots how each token's probability evolves across the 24 layers of the network (the "logit lens trajectory") which is actually really cool to look at

---

## the signals

three things get measured per token:

**signal 1** — how confident was the final layer? (entropy, max prob, margin between top-2)

**signal 2** — how many of the last 8 layers agreed on this token? if the layers are arguing with each other that's a bad sign

**signal 3** — at which layer did the model first commit to this token? late crystallization = usually hallucinating. also tracks how many times the top prediction flipped across layers

these get combined into a risk score per token, then grouped into words/spans

---

## results

ran it on 50 TruthfulQA questions with pythia-1.4b:

```
AUROC (full):       0.796
AUROC (confidence): 0.755
delta:             +0.041
```

signals 2 and 3 (the internal trajectory stuff) add real lift over just checking confidence. which is the whole point

---

## install

```bash
pip install -e .
pip install veritas-audit==0.1.0
```

needs python 3.11+. will download pythia-1.4b (~2.8gb) on first run.

on mac with apple silicon set these for speed:
```bash
export VERITAS_ALLOW_MPS=1
export TRANSFORMERLENS_ALLOW_MPS=1
```

---

## usage

**audit a prompt:**
```bash
veritas audit --prompt "Marie Curie won the Nobel Prize in" --max-tokens 20
```

saves a token heatmap, trajectory plot, and layer agreement chart to `veritas_output/`

**add `--json` for machine-readable output:**
```bash
veritas audit --prompt "..." --json
```

**build a labeled eval set and run evaluation:**
```bash
veritas eval --build-dataset --n-items 50
veritas eval --dataset eval/dataset.jsonl --calibrate
```

**compare two audit runs:**
```bash
veritas compare run1.json run2.json
```

**launch the gradio demo:**
```bash
veritas demo
```

**FCL connection** (this is for the research side — connects to frequency-depth scaling):
```bash
veritas audit --prompt "..." --fcl
```

---

## how it works (slightly more detail)

the core idea is the "logit lens" — at each layer of the transformer you can project the residual stream through the unembedding matrix to get a pseudo-probability distribution over vocab. this lets you watch how the model's "opinion" on the next token changes as it processes through layers.

hallucinated tokens tend to:
- crystallize late (model doesn't commit until the very last layers)
- flip a lot between candidates across layers
- have high variance in probability across the last K layers

this is the empirical version of the FCL frequency-depth scaling relation from my AISB paper

---

## models

| model | size | notes |
|-------|------|-------|
| `EleutherAI/pythia-1.4b` (default) | ~2.8gb float16 | works on 8gb ram |
| `EleutherAI/pythia-2.8b` | ~5.6gb float16 | needs more memory, same code path |

---

## project structure

```
veritas/
  model.py      # load model, generate with per-layer cache
  signals.py    # the three signals → TokenFeatures
  score.py      # logistic scorer, span aggregation, calibration
  fcl.py        # FCL formula + comparison to observed depth
  viz.py        # heatmap, trajectory, agreement, FCL scatter
  cli.py        # all the CLI commands
  data.py       # TruthfulQA dataset builder (substring + semantic labeling)
  demo.py       # gradio app
  schema.py     # pydantic output schema
```

---

## running the tests

```bash
# fast (no model needed, runs in seconds)
pytest tests/ -m "not slow"

# full suite (downloads pythia-1.4b, takes ~30 min on cpu)
pytest tests/ -m slow
```

52 tests total, all green
