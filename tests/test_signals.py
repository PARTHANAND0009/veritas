from __future__ import annotations

import math

import pytest
import torch
from transformer_lens import HookedTransformer

from veritas.signals import _signal1, _signal2_and_3, apply_logit_lens, extract_features

N_LAYERS = 24
VOCAB = 50304
K = 8


def _peaked(idx: int, val: float = 10.0) -> torch.Tensor:
    t = torch.zeros(VOCAB)
    t[idx] = val
    return t


def _layer_logits(chosen: int, agree_last: int, n: int = N_LAYERS) -> torch.Tensor:
    ll = torch.zeros(n, VOCAB)
    other = 0 if chosen != 0 else 1
    for i in range(n):
        ll[i, chosen if i >= n - agree_last else other] = 10.0
    return ll


def test_signal1_ranges() -> None:
    max_p, ent, margin, lp = _signal1(_peaked(7), 7)
    assert 0.0 < max_p <= 1.0
    assert ent >= 0.0
    assert 0.0 <= margin <= 1.0
    assert lp <= 0.0


def test_signal1_uniform_high_entropy() -> None:
    _, ent, margin, _ = _signal1(torch.zeros(VOCAB), 0)
    assert ent > math.log(VOCAB) * 0.99
    assert margin < 1e-3


def test_signal1_peaked_confident() -> None:
    max_p, ent, _, _ = _signal1(_peaked(42, val=20.0), 42)
    assert max_p > 0.99
    assert ent < 0.1


def test_signal1_log_prob_nonpositive() -> None:
    for val in [1.0, 5.0, 20.0]:
        _, _, _, lp = _signal1(_peaked(0, val), 0)
        assert lp <= 0.0


def test_agreement_range() -> None:
    for agree in [0, 4, 8]:
        assert 0.0 <= _signal2_and_3(_layer_logits(5, agree), 5, K)[2] <= 1.0


def test_agreement_full() -> None:
    assert _signal2_and_3(_layer_logits(5, N_LAYERS), 5, K)[2] == 1.0


def test_agreement_zero() -> None:
    assert _signal2_and_3(_layer_logits(5, 0), 5, K)[2] == 0.0


def test_crystallization_range() -> None:
    cryst = _signal2_and_3(_layer_logits(3, 10), 3, K)[3]
    assert 0.0 < cryst <= 1.0


def test_crystallization_never_top1() -> None:
    ll = torch.zeros(N_LAYERS, VOCAB)
    ll[:, 0] = 10.0
    assert _signal2_and_3(ll, 999, K)[3] == 1.0


def test_volatility_nonneg() -> None:
    assert _signal2_and_3(_layer_logits(5, 12), 5, K)[4] >= 0


def test_variance_nonneg() -> None:
    assert _signal2_and_3(_layer_logits(5, 8), 5, K)[5] >= 0.0


def test_per_layer_arrays_length() -> None:
    top1, probs, *_ = _signal2_and_3(_layer_logits(5, 8), 5, K)
    assert len(top1) == N_LAYERS
    assert len(probs) == N_LAYERS


@pytest.mark.slow
def test_apply_logit_lens_shape(pythia: HookedTransformer) -> None:
    residual = torch.randn(pythia.cfg.n_layers, pythia.cfg.d_model)
    out = apply_logit_lens(pythia, residual)
    assert out.shape == (pythia.cfg.n_layers, pythia.cfg.d_vocab)


@pytest.mark.slow
def test_features_length_matches_tokens(pythia: HookedTransformer) -> None:
    from veritas.model import generate
    result = generate(pythia, "The speed of light is", max_new_tokens=5)
    assert len(extract_features(pythia, result)) == len(result.generated_tokens)


@pytest.mark.slow
def test_per_layer_arrays_match_n_layers(pythia: HookedTransformer) -> None:
    from veritas.model import generate
    result = generate(pythia, "Water boils at", max_new_tokens=3)
    for f in extract_features(pythia, result):
        assert len(f.per_layer_top1) == pythia.cfg.n_layers
        assert len(f.per_layer_chosen_prob) == pythia.cfg.n_layers


@pytest.mark.slow
def test_all_feature_ranges(pythia: HookedTransformer) -> None:
    from veritas.model import generate
    result = generate(pythia, "Napoleon was exiled to", max_new_tokens=5)
    for f in extract_features(pythia, result):
        assert 0.0 < f.max_prob <= 1.0
        assert f.entropy >= 0.0
        assert 0.0 <= f.margin <= 1.0
        assert f.log_prob_chosen <= 0.0
        assert 0.0 <= f.layer_agreement <= 1.0
        assert 0.0 < f.crystallization_depth <= 1.0
        assert f.volatility_count >= 0
        assert f.prob_variance >= 0.0
