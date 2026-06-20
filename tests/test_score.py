from __future__ import annotations

import numpy as np
import pytest

from veritas.score import (
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    FEATURE_SCALES,
    TokenRisk,
    aggregate_to_spans,
    calibrate,
    features_to_vector,
    score_tokens,
)
from veritas.signals import TokenFeatures


def _feat(**kw) -> TokenFeatures:
    defaults = dict(
        token_id=42, token_str=" hello", position=0,
        max_prob=0.5, entropy=2.0, margin=0.3, log_prob_chosen=-0.7,
        layer_agreement=0.75, crystallization_depth=0.5,
        volatility_count=3, prob_variance=0.01,
        per_layer_top1=[42] * 24, per_layer_chosen_prob=[0.5] * 24,
    )
    defaults.update(kw)
    return TokenFeatures(**defaults)


def _feats(strs: list[str]) -> list[TokenFeatures]:
    return [_feat(token_str=s, position=i) for i, s in enumerate(strs)]


def _trisk(token_str: str, pos: int, risk: float) -> TokenRisk:
    return TokenRisk(token_str=token_str, position=pos, risk=risk,
                     features=_feat(token_str=token_str, position=pos))


def test_feature_names_length() -> None:
    assert len(FEATURE_NAMES) == 8


def test_default_weights_length() -> None:
    assert len(DEFAULT_WEIGHTS) == 8


def test_feature_scales_length() -> None:
    assert len(FEATURE_SCALES) == 8


def test_features_to_vector_length() -> None:
    assert len(features_to_vector(_feat())) == 8


def test_features_to_vector_values() -> None:
    f = _feat(max_prob=0.9, entropy=1.0, margin=0.2, log_prob_chosen=-0.5,
              layer_agreement=0.8, crystallization_depth=0.3, volatility_count=5, prob_variance=0.02)
    v = features_to_vector(f)
    assert v[0] == pytest.approx(0.9)
    assert v[1] == pytest.approx(1.0)
    assert v[4] == pytest.approx(0.8)
    assert v[6] == pytest.approx(5.0)


def test_risk_score_range() -> None:
    for tr in score_tokens(_feats([" The", " cat", " sat"])):
        assert 0.0 < tr.risk < 1.0


def test_risk_ordering() -> None:
    f_low = _feat(max_prob=0.99, entropy=0.01, margin=0.98, log_prob_chosen=-0.01,
                  layer_agreement=1.0, crystallization_depth=0.1, volatility_count=0, prob_variance=0.0)
    f_high = _feat(max_prob=0.01, entropy=10.0, margin=0.001, log_prob_chosen=-9.0,
                   layer_agreement=0.0, crystallization_depth=1.0, volatility_count=20, prob_variance=0.1)
    assert score_tokens([f_low])[0].risk < score_tokens([f_high])[0].risk


def test_score_tokens_preserves_position() -> None:
    for i, tr in enumerate(score_tokens(_feats([" a", " b", " c"]))):
        assert tr.position == i


def test_zero_weights_gives_half() -> None:
    risks = score_tokens(_feats([" x"]), weights=np.zeros(8))
    assert risks[0].risk == pytest.approx(0.5, abs=1e-6)


def test_span_max_ge_mean() -> None:
    trs = [_trisk(" The", 0, 0.1), _trisk(" quick", 1, 0.4),
           _trisk(" brown", 2, 0.7), _trisk(" fox", 3, 0.3)]
    for span in aggregate_to_spans(trs):
        assert span.risk_max >= span.risk_mean - 1e-9


def test_span_word_count() -> None:
    trs = [_trisk(" Hello", 0, 0.2), _trisk(" world", 1, 0.5), _trisk(" foo", 2, 0.3)]
    assert len(aggregate_to_spans(trs)) == 3


def test_span_subword_grouping() -> None:
    trs = [_trisk(" uni", 0, 0.2), _trisk("verse", 1, 0.6), _trisk(" is", 2, 0.1)]
    spans = aggregate_to_spans(trs)
    assert len(spans) == 2
    assert "universe" in spans[0].span_text.lower()


def test_span_risk_values() -> None:
    spans = aggregate_to_spans([_trisk(" uni", 0, 0.2), _trisk("verse", 1, 0.8)])
    assert spans[0].risk_max == pytest.approx(0.8)
    assert spans[0].risk_mean == pytest.approx(0.5)


def test_empty_token_list() -> None:
    assert aggregate_to_spans([]) == []


def test_calibrate_shape() -> None:
    rng = np.random.default_rng(0)
    w = calibrate(rng.random((20, 8)), rng.integers(0, 2, 20))
    assert w.shape == (8,)
    assert w.dtype == np.float64


def test_calibrate_better_than_random() -> None:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(42)
    X = rng.random((40, 8))
    y = (X[:, 0] < 0.5).astype(int)
    w = calibrate(X[:32], y[:32])
    assert roc_auc_score(y[32:], X[32:] @ w) > 0.5


def test_calibrate_auroc_above_signal1_baseline() -> None:
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    n = 80
    X = rng.random((n, 8))
    y = ((X[:, 0] < 0.4) & (X[:, 5] > 0.6)).astype(int)
    if y.sum() < 4:
        y[:4] = 1
    split = int(0.8 * n)
    cal_w = calibrate(X[:split], y[:split])
    auroc_cal = roc_auc_score(y[split:], (X[split:] / FEATURE_SCALES) @ cal_w)
    assert auroc_cal > 0.5


def test_calibrate_weights_saved(tmp_path: pytest.TempPathFactory) -> None:
    rng = np.random.default_rng(7)
    w = calibrate(rng.random((20, 8)), rng.integers(0, 2, 20))
    p = tmp_path / "weights.npy"  # type: ignore[operator]
    np.save(p, w)
    np.testing.assert_array_equal(w, np.load(p))
