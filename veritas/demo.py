from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr
import pandas as pd

from veritas.model import generate, load_model
from veritas.score import score_tokens
from veritas.signals import extract_features
from veritas.viz import plot_layer_agreement, plot_token_heatmap, plot_trajectory

_model = None
_model_name_loaded = ""


def load_demo_model(model_name: str) -> None:
    global _model, _model_name_loaded
    if _model is None or _model_name_loaded != model_name:
        _model = load_model(model_name)
        _model_name_loaded = model_name


def _run_callback(
    model_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> tuple[str, pd.DataFrame, str, str, str]:
    if _model is None:
        return "Model not loaded.", pd.DataFrame(), "", "", ""

    result = generate(_model, prompt, max_new_tokens=max_tokens, temperature=temperature, seed=42)
    gen_text = "".join(result.generated_strs)

    features = extract_features(_model, result, K=8)
    token_risks = score_tokens(features)

    df = pd.DataFrame([
        {
            "token": tr.token_str,
            "risk": round(tr.risk, 3),
            "max_prob": round(tr.features.max_prob, 3),
            "layer_agreement": round(tr.features.layer_agreement, 3),
            "crystallization_depth": round(tr.features.crystallization_depth, 3),
        }
        for tr in token_risks
    ])

    n_layers = _model.cfg.n_layers
    riskiest = max(range(len(token_risks)), key=lambda i: token_risks[i].risk)

    def _tmpfile() -> str:
        f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        f.close()
        return f.name

    hm, traj, ag = _tmpfile(), _tmpfile(), _tmpfile()
    plot_token_heatmap(token_risks, Path(hm))
    plot_trajectory(features, riskiest, n_layers, Path(traj))
    plot_layer_agreement(features, Path(ag))

    return gen_text, df, hm, traj, ag


def build_app(default_model: str = "EleutherAI/pythia-1.4b") -> gr.Blocks:
    with gr.Blocks(title="Veritas") as demo:
        gr.Markdown("## Veritas — hallucination auditor")

        with gr.Row():
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=["EleutherAI/pythia-1.4b", "EleutherAI/pythia-2.8b"],
                    value=default_model,
                    label="Model",
                )
                prompt_box = gr.Textbox(label="Prompt", placeholder="The capital of Australia is", lines=3)
                max_tokens_slider = gr.Slider(5, 80, value=20, step=1, label="Max tokens")
                temp_slider = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature")
                run_btn = gr.Button("Run", variant="primary")

            with gr.Column(scale=2):
                gen_output = gr.Textbox(label="Generation", interactive=False)
                risk_table = gr.Dataframe(
                    headers=["token", "risk", "max_prob", "layer_agreement", "crystallization_depth"],
                    label="Token risks",
                )
                heatmap_img = gr.Image(label="Heatmap", type="filepath")
                traj_img = gr.Image(label="Trajectory (riskiest token)", type="filepath")
                agree_img = gr.Image(label="Layer agreement", type="filepath")

        run_btn.click(
            fn=_run_callback,
            inputs=[model_dd, prompt_box, max_tokens_slider, temp_slider],
            outputs=[gen_output, risk_table, heatmap_img, traj_img, agree_img],
        )

    return demo


def launch(default_model: str = "EleutherAI/pythia-1.4b") -> None:
    load_demo_model(default_model)
    build_app(default_model).launch(inbrowser=True)
