"""Diagnostic plots for Veritas audit results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from veritas.score import TokenRisk
from veritas.signals import TokenFeatures


def plot_token_heatmap(
    token_risks: list[TokenRisk],
    save_path: Path,
    title: str = "Token Risk Heatmap",
) -> None:
    risks = np.array([tr.risk for tr in token_risks])
    labels = [tr.token_str.replace("\n", "↵") for tr in token_risks]

    n = len(risks)
    fig_w = max(10, n * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, 1.8))

    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.Normalize(vmin=0, vmax=1)

    for i, (risk, label) in enumerate(zip(risks, labels)):
        color = cmap(norm(risk))
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.text(
            i + 0.5, 0.5, label,
            ha="center", va="center",
            fontsize=7, color="black" if risk < 0.65 else "white",
            clip_on=True,
        )

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.15, label="Risk")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectory(
    features: list[TokenFeatures],
    token_idx: int,
    n_layers: int,
    save_path: Path,
) -> None:
    f = features[token_idx]
    probs = f.per_layer_chosen_prob
    layers = list(range(n_layers))
    cryst_layer = f.crystallization_depth * n_layers

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, probs, marker="o", markersize=3, linewidth=1.5, color="#2563eb")
    ax.axvline(cryst_layer, color="#dc2626", linestyle="--", linewidth=1.2,
               label=f"crystallization @ layer {cryst_layer:.1f}")
    ax.set_xlabel("Layer")
    ax.set_ylabel(f'P("{f.token_str.strip()}")')
    ax.set_title(f'Logit-lens trajectory for token {token_idx}: "{f.token_str.strip()}"')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_agreement(
    features: list[TokenFeatures],
    save_path: Path,
) -> None:
    agreements = [f.layer_agreement for f in features]
    labels = [f.token_str.replace("\n", "↵") for f in features]
    x = list(range(len(agreements)))

    fig_w = max(8, len(x) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    colors = ["#16a34a" if a >= 0.5 else "#dc2626" for a in agreements]
    ax.bar(x, agreements, color=colors, width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Layer agreement (last K layers)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Layer-wise Agreement per Generated Token")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Path,
) -> dict[str, float]:
    auroc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(fpr, tpr, color="#2563eb", linewidth=2, label=f"AUROC = {auroc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.set_title("ROC Curve")
    ax1.legend()

    ax2.plot(recall, precision, color="#16a34a", linewidth=2, label=f"AP = {ap:.3f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"auroc": auroc, "ap": ap}


def plot_fcl_residuals(df: "pd.DataFrame", save_path: Path) -> None:  # noqa: F821
    """Scatter: x=predicted_depth, y=observed_depth, color=risk proxy, diagonal y=x."""

    risks = df["is_hallucinated_proxy"].astype(float).values
    colors = ["#dc2626" if r > 0.5 else "#16a34a" for r in risks]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        df["predicted_depth"],
        df["observed_depth"],
        c=colors,
        alpha=0.7,
        edgecolors="none",
        s=60,
    )
    lims = [0.0, 1.0]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="y = x (perfect FCL match)")
    ax.set_xlabel("FCL-predicted crystallization depth")
    ax.set_ylabel("Observed crystallization depth")
    ax.set_title("Observed vs FCL-predicted crystallization depth")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#dc2626", label="hallucinated (proxy)"),
        Patch(facecolor="#16a34a", label="correct (proxy)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
