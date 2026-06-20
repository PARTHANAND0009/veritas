"""CLI entry points: `veritas audit` and `veritas eval`."""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

DEFAULT_MODEL = "EleutherAI/pythia-1.4b"


def _check_memory_gb(required_gb: float = 6.0) -> tuple[bool, float]:
    """Return (ok, available_gb). ok=False means insufficient memory to load safely."""
    import psutil
    available_bytes = psutil.virtual_memory().available
    available_gb = available_bytes / (1024 ** 3)
    return available_gb >= required_gb, available_gb


@click.group()
def cli() -> None:
    """Veritas — white-box hallucination auditor for open-weight transformers."""


@cli.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="HuggingFace model name.")
@click.option("--prompt", required=True, help="Prompt string to audit.")
@click.option("--max-tokens", default=200, show_default=True, help="Max new tokens to generate.")
@click.option("--temperature", default=0.0, show_default=True, help="Sampling temperature.")
@click.option("--k", default=8, show_default=True, help="Last K layers for agreement/variance.")
@click.option(
    "--output-dir", default="veritas_output", show_default=True, help="Directory for saved plots."
)
@click.option(
    "--json", "emit_json", is_flag=True, default=False, help="Write JSON of features+scores."
)
@click.option(
    "--weights", default=None, type=click.Path(exists=True), help=".npy calibrated weights file."
)
@click.option("--seed", default=42, show_default=True, help="RNG seed for reproducibility.")
@click.option("--fcl", "show_fcl", is_flag=True, default=False, help="Compare to FCL prediction.")
def audit(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    k: int,
    output_dir: str,
    emit_json: bool,
    weights: str | None,
    seed: int,
    show_fcl: bool,
) -> None:
    """Generate from MODEL given PROMPT and score hallucination risk per token and span."""
    from veritas.model import generate, load_model
    from veritas.score import aggregate_to_spans, score_tokens
    from veritas.signals import extract_features
    from veritas.viz import plot_layer_agreement, plot_token_heatmap, plot_trajectory

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Loading model:[/bold] {model}")
    m = load_model(model)

    console.print(f"[bold]Generating:[/bold] up to {max_tokens} tokens …")
    result = generate(m, prompt, max_new_tokens=max_tokens, temperature=temperature, seed=seed)
    generation_text = "".join(result.generated_strs)
    console.print(f"\n[bold]Generation:[/bold] {generation_text}\n")

    console.print("[bold]Extracting features …[/bold]")
    features = extract_features(m, result, K=k)

    w_arr: np.ndarray | None = None
    if weights:
        w_arr = np.load(weights)
        console.print(f"[green]Using calibrated weights from {weights}[/green]")

    token_risks = score_tokens(features, weights=w_arr)
    spans = aggregate_to_spans(token_risks)

    table = Table(title="Span Risk Scores", show_lines=True)
    table.add_column("Span", style="cyan", no_wrap=False)
    table.add_column("Risk (max)", justify="right")
    table.add_column("Risk (mean)", justify="right")
    for span in spans:
        color = "red" if span.risk_max > 0.6 else ("yellow" if span.risk_max > 0.4 else "green")
        table.add_row(
            span.span_text,
            f"[{color}]{span.risk_max:.3f}[/{color}]",
            f"{span.risk_mean:.3f}",
        )
    console.print(table)

    n_layers = m.cfg.n_layers
    heatmap_path = out / "token_heatmap.png"
    traj_path = out / "trajectory.png"
    agreement_path = out / "layer_agreement.png"

    plot_token_heatmap(token_risks, heatmap_path)
    if features:
        riskiest_idx = max(range(len(token_risks)), key=lambda i: token_risks[i].risk)
        plot_trajectory(features, riskiest_idx, n_layers, traj_path)
    plot_layer_agreement(features, agreement_path)

    console.print(f"\n[green]Plots saved to {out}/[/green]")
    console.print(f"  {heatmap_path}")
    console.print(f"  {traj_path}")
    console.print(f"  {agreement_path}")

    if emit_json:
        from datetime import datetime, timezone

        from veritas.schema import (
            AuditResult,
            SpanRiskRecord,
            TokenFeatureRecord,
            TokenRiskRecord,
        )

        def _to_feature_record(f) -> TokenFeatureRecord:
            return TokenFeatureRecord(
                token_id=f.token_id,
                token_str=f.token_str,
                position=f.position,
                max_prob=f.max_prob,
                entropy=f.entropy,
                margin=f.margin,
                log_prob_chosen=f.log_prob_chosen,
                layer_agreement=f.layer_agreement,
                crystallization_depth=f.crystallization_depth,
                volatility_count=f.volatility_count,
                prob_variance=f.prob_variance,
                per_layer_top1=f.per_layer_top1,
                per_layer_chosen_prob=f.per_layer_chosen_prob,
            )

        token_risk_records = [
            TokenRiskRecord(
                token_str=tr.token_str,
                position=tr.position,
                risk=tr.risk,
                features=_to_feature_record(tr.features),
            )
            for tr in token_risks
        ]
        span_risk_records = [
            SpanRiskRecord(
                span_text=sp.span_text,
                risk_max=sp.risk_max,
                risk_mean=sp.risk_mean,
                token_risks=[
                    TokenRiskRecord(
                        token_str=str_tr.token_str,
                        position=str_tr.position,
                        risk=str_tr.risk,
                        features=_to_feature_record(str_tr.features),
                    )
                    for str_tr in sp.token_risks
                ],
            )
            for sp in spans
        ]
        audit_result = AuditResult(
            model_name=model,
            prompt=prompt,
            generation=generation_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
            seed=seed,
            weights_used=(w_arr.tolist() if w_arr is not None else list(
                __import__("veritas.score", fromlist=["DEFAULT_WEIGHTS"]).DEFAULT_WEIGHTS
            )),
            token_risks=token_risk_records,
            span_risks=span_risk_records,
        )
        json_path = out / "audit_result.json"
        json_path.write_text(audit_result.model_dump_json(indent=2))
        console.print(f"  {json_path}")

    if show_fcl:
        from veritas.fcl import compare_to_fcl
        from veritas.viz import plot_fcl_residuals

        # Proxy for concept frequency: normalize token_id rank by vocab size.
        vocab_size = m.cfg.d_vocab
        freq_map = {
            tr.token_str: 1.0 - tr.features.token_id / vocab_size for tr in token_risks
        }
        fcl_df = compare_to_fcl(features, freq_map, n_layers)
        console.print("\n[bold]FCL comparison:[/bold]")
        console.print(fcl_df.to_string(index=False))

        fcl_csv = out / "fcl_comparison.csv"
        fcl_df.to_csv(fcl_csv, index=False)
        fcl_plot = out / "fcl_residuals.png"
        plot_fcl_residuals(fcl_df, fcl_plot)
        console.print(f"\n[green]FCL output saved to {fcl_csv} and {fcl_plot}[/green]")


@cli.command()
@click.option(
    "--dataset",
    default="eval/dataset.jsonl",
    show_default=True,
    help="Path to JSONL eval set.",
)
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="HuggingFace model name.")
@click.option(
    "--models",
    multiple=True,
    default=(),
    help="Eval multiple models sequentially (repeatable flag). Overrides --model when provided.",
)
@click.option(
    "--build-dataset",
    "build",
    is_flag=True,
    default=False,
    help="Build dataset from TruthfulQA first.",
)
@click.option(
    "--n-items", default=50, show_default=True, help="Items to sample when building dataset."
)
@click.option(
    "--calibrate",
    "do_calibrate",
    is_flag=True,
    default=False,
    help="Fit weights on 80/20 split.",
)
@click.option("--k", default=8, show_default=True, help="Last K layers for agreement/variance.")
@click.option(
    "--output-dir", default="veritas_output", show_default=True, help="Directory for plots."
)
@click.option("--seed", default=42, show_default=True, help="RNG seed.")
@click.option("--max-tokens", default=80, show_default=True, help="Max tokens per eval generation.")
@click.option(
    "--label-strategy",
    "label_strategy",
    default="substring",
    show_default=True,
    type=click.Choice(["substring", "semantic"]),
    help="Labeling strategy when building dataset. 'semantic' requires sentence-transformers.",
)
def eval(
    dataset: str,
    model: str,
    models: tuple[str, ...],
    build: bool,
    n_items: int,
    do_calibrate: bool,
    k: int,
    output_dir: str,
    seed: int,
    max_tokens: int,
    label_strategy: str,
) -> None:
    """Evaluate Veritas on a labeled JSONL dataset and report AUROC."""
    import gc

    import numpy as np
    from sklearn.metrics import roc_auc_score

    from veritas.data import build_truthfulqa_dataset, load_dataset
    from veritas.model import generate, load_model
    from veritas.score import calibrate, features_to_vector, score_tokens
    from veritas.signals import extract_features
    from veritas.viz import plot_roc_pr

    # --models overrides --model when provided.
    model_list = list(models) if models else [model]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(dataset)
    items = None  # loaded once; reused across models

    # Multi-model comparison table rows
    comparison_rows: list[dict] = []

    for model_name in model_list:
        # Memory guard: require ≥6 GB free before loading each model.
        ok, avail_gb = _check_memory_gb(required_gb=6.0)
        if not ok:
            console.print(
                f"[yellow]Skipping {model_name}: only {avail_gb:.1f} GB available "
                f"(need 6 GB). Free memory and retry.[/yellow]"
            )
            continue

        console.print(f"\n[bold]Loading model:[/bold] {model_name}")
        m = load_model(model_name)

        if items is None:
            if build:
                console.print("[bold]Building TruthfulQA dataset …[/bold]")
                items = build_truthfulqa_dataset(
                    m, n_items=n_items, output_path=dataset_path, seed=seed,
                    max_new_tokens=max_tokens, label_strategy=label_strategy,
                )
            else:
                items = load_dataset(dataset_path)

        console.print(f"[bold]Evaluating on {len(items)} items …[/bold]")

        # Pass 1 — collect features and labels for every item.
        all_feature_vecs: list[np.ndarray] = []
        all_features_list: list = []
        all_labels: list[int] = []

        s1_weights = np.array([-1.5, 0.8, -1.0, -0.8, 0.0, 0.0, 0.0, 0.0])

        for i, item in enumerate(items):
            console.print(f"  [{i + 1}/{len(items)}] {item.prompt[:60]!r}")
            result = generate(
                m, item.prompt, max_new_tokens=max_tokens, temperature=0.0, seed=seed
            )
            features = extract_features(m, result, K=k)
            if not features:
                continue
            vec = np.mean([features_to_vector(f) for f in features], axis=0)
            all_feature_vecs.append(vec)
            all_features_list.append(features)
            all_labels.append(item.label)

        y_true = np.array(all_labels)
        X_all = np.stack(all_feature_vecs)

        def _max_risk(feats, weights=None):
            return max(tr.risk for tr in score_tokens(feats, weights=weights))

        y_full = np.array([_max_risk(f) for f in all_features_list])
        y_s1 = np.array([_max_risk(f, weights=s1_weights) for f in all_features_list])

        auroc_full = roc_auc_score(y_true, y_full)
        auroc_s1 = roc_auc_score(y_true, y_s1)

        auroc_cal: float | None = None
        ap_cal: float | None = None
        n_train = 0
        y_cal_test = y_full
        y_true_test = y_true
        if do_calibrate and len(all_feature_vecs) >= 5:
            split = int(0.8 * len(all_feature_vecs))
            n_train = split
            cal_weights = calibrate(X_all[:split], y_true[:split])
            y_cal_test = np.array([
                _max_risk(f, weights=cal_weights) for f in all_features_list[split:]
            ])
            y_true_test = y_true[split:]
            from sklearn.metrics import average_precision_score
            auroc_cal = roc_auc_score(y_true_test, y_cal_test)
            ap_cal = average_precision_score(y_true_test, y_cal_test)
            weights_path = out / "calibrated_weights.npy"
            np.save(weights_path, cal_weights)
            console.print(f"[green]Calibrated weights saved to {weights_path}[/green]")

        console.print(f"\n[bold]AUROC (full, default weights):    {auroc_full:.3f}[/bold]")
        if auroc_cal is not None:
            console.print(
                f"[bold]AUROC (full, calibrated weights): {auroc_cal:.3f}"
                f"  [fitted on {n_train} train items][/bold]"
            )
        console.print(f"[bold]AUROC (Signal-1 only):            {auroc_s1:.3f}[/bold]")
        baseline = auroc_cal if auroc_cal is not None else auroc_full
        delta = baseline - auroc_s1
        delta_color = "green" if delta >= 0 else "red"
        console.print(
            f"[bold][{delta_color}]Delta (calibrated vs S1 only):   "
            f"{delta:+.3f}[/{delta_color}][/bold]"
        )
        roc_path = out / f"roc_pr_{model_name.replace('/', '_')}.png"
        plot_scores = y_cal_test if auroc_cal is not None else y_full
        plot_true = y_true_test if auroc_cal is not None else y_true
        metrics = plot_roc_pr(plot_true, plot_scores, roc_path)
        ap_to_show = ap_cal if ap_cal is not None else metrics["ap"]
        console.print(f"AP   (calibrated):                {ap_to_show:.3f}")
        console.print(f"\n[green]ROC/PR plot saved to {roc_path}[/green]")

        comparison_rows.append({
            "Model": model_name,
            "AUROC (full)": f"{auroc_full:.3f}",
            "AUROC (S1)": f"{auroc_s1:.3f}",
            "Delta": f"{delta:+.3f}",
        })

        # Unload model to free memory before loading the next one.
        del m
        gc.collect()
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    if len(comparison_rows) > 1:
        console.print("\n[bold]Multi-model comparison:[/bold]")
        header = f"{'Model':<35} {'AUROC (full)':>12} {'AUROC (S1)':>10} {'Delta':>8}"
        console.print(header)
        console.print("-" * len(header))
        for row in comparison_rows:
            console.print(
                f"{row['Model']:<35} {row['AUROC (full)']:>12} "
                f"{row['AUROC (S1)']:>10} {row['Delta']:>8}"
            )


@cli.command()
@click.option("--model", default=DEFAULT_MODEL, show_default=True, help="Model to load.")
def demo(model: str) -> None:
    """Launch the Gradio demo: enter a prompt, see heatmap + trajectory inline."""
    from veritas.demo import launch

    console.print(f"[bold]Loading model for demo:[/bold] {model}")
    launch(default_model=model)


@cli.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
def compare(file1: str, file2: str) -> None:
    """Compare two AuditResult JSON files side-by-side by span risk scores."""
    from veritas.schema import AuditResult

    r1 = AuditResult.model_validate_json(Path(file1).read_text())
    r2 = AuditResult.model_validate_json(Path(file2).read_text())

    console.print(f"[bold]File 1:[/bold] {file1}  model={r1.model_name}")
    console.print(f"[bold]File 2:[/bold] {file2}  model={r2.model_name}")
    console.print(f"[bold]Prompt:[/bold] {r1.prompt!r}\n")

    spans1 = {sp.span_text: sp for sp in r1.span_risks}
    spans2 = {sp.span_text: sp for sp in r2.span_risks}
    all_spans = sorted(set(spans1) | set(spans2))

    table = Table(title="Span Risk Comparison", show_lines=True)
    table.add_column("Span", style="cyan")
    table.add_column("File1 max", justify="right")
    table.add_column("File2 max", justify="right")
    table.add_column("Δ", justify="right")

    for span_text in all_spans:
        r1_risk = spans1[span_text].risk_max if span_text in spans1 else float("nan")
        r2_risk = spans2[span_text].risk_max if span_text in spans2 else float("nan")
        both_valid = (r1_risk == r1_risk) and (r2_risk == r2_risk)
        delta = (r2_risk - r1_risk) if both_valid else float("nan")
        delta_str = f"{delta:+.3f}" if delta == delta else "—"
        color = "red" if delta > 0.1 else ("green" if delta < -0.1 else "white")
        table.add_row(
            span_text,
            f"{r1_risk:.3f}" if r1_risk == r1_risk else "—",
            f"{r2_risk:.3f}" if r2_risk == r2_risk else "—",
            f"[{color}]{delta_str}[/{color}]",
        )
    console.print(table)
