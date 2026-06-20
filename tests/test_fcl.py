from __future__ import annotations

import math

from veritas.fcl import compare_to_fcl, fcl_predicted_depth
from veritas.signals import TokenFeatures


def _feat(pos: int = 0, cryst: float = 0.5) -> TokenFeatures:
    return TokenFeatures(
        token_id=pos, token_str=f" tok{pos}", position=pos,
        max_prob=0.8, entropy=1.0, margin=0.3, log_prob_chosen=-0.3,
        layer_agreement=0.75, crystallization_depth=cryst,
        volatility_count=2, prob_variance=0.01,
        per_layer_top1=[pos] * 24, per_layer_chosen_prob=[0.8] * 24,
    )


def test_fcl_predicted_depth_range() -> None:
    for freq in [0.0, 0.1, 0.5, 0.9, 1.0]:
        d = fcl_predicted_depth(freq, 24)
        assert 0.0 < d <= 1.0


def test_fcl_predicted_depth_monotonic() -> None:
    depths = [fcl_predicted_depth(f, 24) for f in [0.0, 0.25, 0.5, 0.75, 1.0]]
    assert all(depths[i] >= depths[i + 1] for i in range(len(depths) - 1))


def test_fcl_spot_check_low_freq() -> None:
    # at f=0: 0.45 + 0.42*exp(0) = 0.87
    assert abs(fcl_predicted_depth(0.0, 24) - 0.87) < 0.01


def test_fcl_spot_check_high_freq() -> None:
    expected = 0.45 + 0.42 * math.exp(-1.80)
    assert abs(fcl_predicted_depth(1.0, 24) - expected) < 0.01


def test_compare_to_fcl_dataframe() -> None:
    import pandas as pd
    features = [_feat(i, 0.4 + i * 0.1) for i in range(4)]
    df = compare_to_fcl(features, {f" tok{i}": 0.1 * i for i in range(4)}, 24)
    assert isinstance(df, pd.DataFrame)
    for col in ["token_str", "observed_depth", "predicted_depth", "residual", "is_hallucinated_proxy"]:
        assert col in df.columns


def test_compare_to_fcl_row_count() -> None:
    features = [_feat(i) for i in range(6)]
    df = compare_to_fcl(features, {f" tok{i}": 0.5 for i in range(6)}, 24)
    assert len(df) == 6


def test_compare_residual_identity() -> None:
    df = compare_to_fcl([_feat(0, cryst=0.6)], {" tok0": 0.0}, 24)
    row = df.iloc[0]
    assert abs(row["residual"] - (row["observed_depth"] - row["predicted_depth"])) < 1e-9
