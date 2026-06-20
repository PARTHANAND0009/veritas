from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from veritas.signals import TokenFeatures

logger = logging.getLogger(__name__)

FEATURE_NAMES: tuple[str, ...] = (
    "max_prob", "entropy", "margin", "log_prob_chosen",
    "layer_agreement", "crystallization_depth", "volatility_count", "prob_variance",
)

DEFAULT_WEIGHTS = np.array([-1.5, 0.8, -1.0, -0.8, -1.2, 0.6, 0.4, 0.5], dtype=np.float64)




# (entropy can be ~10 nats, volatility up to n_layers, variance is tiny)
FEATURE_SCALES = np.array(
    [1.0, 4.0, 1.0, 10.0, 1.0, 1.0, 24.0, 0.05],
    dtype=np.float64,
)


@dataclass
class TokenRisk:
    token_str: str
    position: int
    risk: float
    features: TokenFeatures


@dataclass
class SpanRisk:
    span_text: str
    token_risks: list[TokenRisk] = field(default_factory=list)
    risk_max: float = 0.0
    risk_mean: float = 0.0


def features_to_vector(f: TokenFeatures) -> np.ndarray:
    return np.array(
        [f.max_prob, f.entropy, f.margin, f.log_prob_chosen,
         f.layer_agreement, f.crystallization_depth, float(f.volatility_count), f.prob_variance],
        dtype=np.float64,
    )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def score_tokens(
    features: list[TokenFeatures],
    weights: np.ndarray | None = None,
) -> list[TokenRisk]:
    w = DEFAULT_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float64)
    logger.debug("weights: %s", dict(zip(FEATURE_NAMES, w)))
    return [
        TokenRisk(
            token_str=f.token_str,
            position=f.position,
            risk=_sigmoid(float(w @ (features_to_vector(f) / FEATURE_SCALES))),
            features=f,
        )
        for f in features
    ]


def aggregate_to_spans(token_risks: list[TokenRisk]) -> list[SpanRisk]:
    # GPT-2 tokenizer prefixes word-starting tokens with a space; ▁ for SentencePiece
    spans: list[SpanRisk] = []
    current: list[TokenRisk] = []

    for tr in token_risks:
        starts_word = tr.token_str.startswith((" ", "▁")) or not current
        if starts_word and current:
            spans.append(_finalize_span(current))
            current = []
        current.append(tr)

    if current:
        spans.append(_finalize_span(current))
    return spans


def _finalize_span(trs: list[TokenRisk]) -> SpanRisk:
    risks = [tr.risk for tr in trs]
    return SpanRisk(
        span_text="".join(tr.token_str for tr in trs).strip(),
        token_risks=list(trs),
        risk_max=max(risks),
        risk_mean=sum(risks) / len(risks),
    )


def calibrate(feature_vectors: np.ndarray, labels: np.ndarray) -> np.ndarray:
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf.fit(feature_vectors / FEATURE_SCALES, labels)
    return clf.coef_.flatten().astype(np.float64)
