"""model.py tests — slow ones need pythia-1.4b (session fixture, don't load inline)."""
from __future__ import annotations

import pytest
import torch
from transformer_lens import HookedTransformer

from veritas.model import GenerationResult, _default_device, generate, load_model


def test_default_device_returns_string() -> None:
    assert _default_device() in ("mps", "cpu", "cuda")


def test_unsupported_model_raises() -> None:
    with pytest.raises(NotImplementedError):
        load_model("facebook/opt-1.3b")


def test_generation_result_is_dataclass() -> None:
    r = GenerationResult(
        prompt_tokens=[0, 1, 2],
        generated_tokens=[3, 4],
        generated_strs=["hello", " world"],
        residuals=[torch.zeros(24, 2048), torch.zeros(24, 2048)],
        final_logits=[torch.zeros(50304), torch.zeros(50304)],
    )
    assert isinstance(r.prompt_tokens, list)
    assert isinstance(r.residuals, list)


def test_supported_prefixes_includes_pythia_2_8b() -> None:
    from veritas.model import _SUPPORTED_PREFIXES
    assert any("EleutherAI/pythia-2.8b".startswith(p) for p in _SUPPORTED_PREFIXES)


def test_memory_guard_warns_when_below_threshold() -> None:
    from unittest.mock import MagicMock, patch

    from veritas.cli import _check_memory_gb

    mock_vm = MagicMock()
    mock_vm.available = 4 * 1024 ** 3  # 4 GB
    with patch("psutil.virtual_memory", return_value=mock_vm):
        ok, avail = _check_memory_gb(required_gb=6.0)
    assert not ok
    assert avail < 6.0


@pytest.mark.slow
def test_load_model_pythia(pythia: HookedTransformer) -> None:
    assert "pythia" in pythia.cfg.model_name.lower()


@pytest.mark.slow
def test_generate_returns_correct_types(pythia: HookedTransformer) -> None:
    result = generate(pythia, "The capital of France is", max_new_tokens=5)
    assert len(result.generated_tokens) <= 5
    assert len(result.generated_strs) == len(result.generated_tokens)
    assert len(result.residuals) == len(result.generated_tokens)


@pytest.mark.slow
def test_residuals_shape(pythia: HookedTransformer) -> None:
    result = generate(pythia, "The Eiffel Tower is in", max_new_tokens=3)
    n, d = pythia.cfg.n_layers, pythia.cfg.d_model
    for t, res in enumerate(result.residuals):
        assert res.shape == (n, d), f"residuals[{t}] shape {res.shape} != ({n}, {d})"


@pytest.mark.slow
def test_generate_greedy_deterministic(pythia: HookedTransformer) -> None:
    p = "Marie Curie was born in"
    r1 = generate(pythia, p, max_new_tokens=5, seed=42)
    r2 = generate(pythia, p, max_new_tokens=5, seed=42)
    assert r1.generated_tokens == r2.generated_tokens


@pytest.mark.slow
def test_generate_stops_at_max_tokens(pythia: HookedTransformer) -> None:
    assert len(generate(pythia, "Tell me a long story:", max_new_tokens=3).generated_tokens) <= 3


@pytest.mark.slow
def test_final_logits_shape(pythia: HookedTransformer) -> None:
    result = generate(pythia, "Hello", max_new_tokens=2)
    for logits in result.final_logits:
        assert logits.shape == (pythia.cfg.d_vocab,)
