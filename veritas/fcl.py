"""FCL(f, L) = (0.45 + 0.42 * exp(-1.80 * f)) * L — predicted crystallization depth.

See: Frequency-Depth Scaling Relation from the AISB paper written by me ofc
"""
from __future__ import annotations

import math

import pandas as pd

from veritas.score import DEFAULT_WEIGHTS, FEATURE_SCALES, features_to_vector
from veritas.signals import TokenFeatures


def fcl_predicted_depth(concept_frequency: float, n_layers: int) -> float:
    """Predicted normalized crystallization depth ∈ (0, 1]."""
    return 0.45 + 0.42 * math.exp(-1.80 * concept_frequency)


def compare_to_fcl(
    features: list[TokenFeatures],
    concept_frequencies: dict[str, float],
    n_layers: int,
) -> pd.DataFrame:
    rows = []
    for f in features:
        freq = concept_frequencies.get(f.token_str, 0.5)
        predicted = fcl_predicted_depth(freq, n_layers)
        vec = features_to_vector(f) / FEATURE_SCALES
        risk = 1.0 / (1.0 + math.exp(-float(DEFAULT_WEIGHTS @ vec)))
        rows.append({
            "token_str": f.token_str,
            "observed_depth": f.crystallization_depth,
            "predicted_depth": predicted,
            "residual": f.crystallization_depth - predicted,
            "is_hallucinated_proxy": risk > 0.5,
        })
    return pd.DataFrame(rows)
