"""HuggingFace Spaces entry point."""
import os

os.environ["VERITAS_FLOAT32"] = "1"  # CPU-only on Spaces, float32 is fine

from veritas.demo import build_app, load_demo_model

print("Loading model... (takes a minute on first run)")
load_demo_model("EleutherAI/pythia-1.4b")

demo = build_app()
demo.launch()
