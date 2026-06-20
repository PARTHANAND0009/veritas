from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformer_lens import HookedTransformer

from veritas.model import GenerationResult


@dataclass
class TokenFeatures:
    token_id: int
    token_str: str
    position: int

    max_prob: float
    entropy: float
    margin: float
    log_prob_chosen: float

    layer_agreement: float
    crystallization_depth: float
    volatility_count: int
    prob_variance: float
    per_layer_top1: list[int] = field(default_factory=list)
    per_layer_chosen_prob: list[float] = field(default_factory=list)


def apply_logit_lens(model: HookedTransformer, residual: torch.Tensor) -> torch.Tensor:
    """residual: [n_layers, d_model] → logits: [n_layers, vocab]"""
    with torch.no_grad():
        residual = residual.to(next(model.parameters()).device)
        normed = model.ln_final(residual.unsqueeze(0)).squeeze(0)
        return (normed @ model.W_U + model.b_U).float().cpu()


def _signal1(logits: torch.Tensor, chosen_id: int) -> tuple[float, float, float, float]:
    logits = logits.float()
    probs = torch.softmax(logits, dim=-1)
    sorted_p = probs.sort(descending=True).values
    return (
        float(probs.max().item()),
        float(-(probs * probs.log().clamp(min=-1e9)).sum().item()),
        float((sorted_p[0] - sorted_p[1]).item()),
        float(torch.log_softmax(logits, dim=-1)[chosen_id].item()),
    )


def _signal2_and_3(
    per_layer_logits: torch.Tensor,
    chosen_id: int,
    K: int,
) -> tuple[list[int], list[float], float, float, int, float]:
    n = per_layer_logits.shape[0]
    top1 = per_layer_logits.argmax(dim=-1).tolist()
    chosen_probs = torch.softmax(per_layer_logits, dim=-1)[:, chosen_id].tolist()

    agreement = sum(1 for t in top1[-K:] if t == chosen_id) / K

    cryst_idx = next((i for i, t in enumerate(top1) if t == chosen_id), None)
    cryst_depth = (cryst_idx + 1) / n if cryst_idx is not None else 1.0

    volatility = sum(1 for i in range(1, n) if top1[i] != top1[i - 1])

    last_k = chosen_probs[-K:]
    mean_p = sum(last_k) / K
    variance = sum((p - mean_p) ** 2 for p in last_k) / K

    return top1, chosen_probs, agreement, cryst_depth, volatility, variance


def extract_features(
    model: HookedTransformer,
    result: GenerationResult,
    K: int = 8,
) -> list[TokenFeatures]:
    features = []
    for t, (token_id, token_str, step_logits, step_residuals) in enumerate(
        zip(result.generated_tokens, result.generated_strs, result.final_logits, result.residuals)
    ):
        max_prob, entropy, margin, lp = _signal1(step_logits, token_id)
        layer_logits = apply_logit_lens(model, step_residuals)
        top1, chosen_probs, agreement, cryst, vol, var = _signal2_and_3(layer_logits, token_id, K)

        features.append(TokenFeatures(
            token_id=token_id,
            token_str=token_str,
            position=t,
            max_prob=max_prob,
            entropy=entropy,
            margin=margin,
            log_prob_chosen=lp,
            layer_agreement=agreement,
            crystallization_depth=cryst,
            volatility_count=vol,
            prob_variance=var,
            per_layer_top1=top1,
            per_layer_chosen_prob=chosen_probs,
        ))

    return features
