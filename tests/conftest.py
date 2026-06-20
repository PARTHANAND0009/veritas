"""Shared pytest fixtures. The model is loaded once per test session."""

from __future__ import annotations

import pytest
from transformer_lens import HookedTransformer

from veritas.model import load_model


@pytest.fixture(scope="session")
def pythia() -> HookedTransformer:
    """Load pythia-1.4b once for the whole session; reused by all slow tests."""
    return load_model("EleutherAI/pythia-1.4b")
