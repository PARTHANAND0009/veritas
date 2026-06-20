from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd


def test_demo_imports() -> None:
    from veritas.demo import build_app  # noqa: F401


def test_demo_callback_returns_correct_types() -> None:
    from veritas.demo import _run_callback
    from veritas.score import TokenRisk
    from veritas.signals import TokenFeatures

    def _f(pos: int) -> TokenFeatures:
        return TokenFeatures(
            token_id=pos, token_str=f" tok{pos}", position=pos,
            max_prob=0.8, entropy=1.0, margin=0.3, log_prob_chosen=-0.3,
            layer_agreement=0.75, crystallization_depth=0.4,
            volatility_count=2, prob_variance=0.01,
            per_layer_top1=[pos] * 24, per_layer_chosen_prob=[0.8] * 24,
        )

    feats = [_f(i) for i in range(3)]
    trisks = [TokenRisk(token_str=f.token_str, position=f.position, risk=0.5, features=f) for f in feats]

    mock_result = MagicMock()
    mock_result.generated_strs = [" hello", " world", "!"]
    mock_result.generated_tokens = [1, 2, 3]
    mock_result.residuals = [MagicMock() for _ in range(3)]
    mock_result.final_logits = [MagicMock() for _ in range(3)]

    mock_model = MagicMock()
    mock_model.cfg.n_layers = 24

    with (
        patch("veritas.demo._model", mock_model),
        patch("veritas.demo.generate", return_value=mock_result),
        patch("veritas.demo.extract_features", return_value=feats),
        patch("veritas.demo.score_tokens", return_value=trisks),
        patch("veritas.demo.plot_token_heatmap"),
        patch("veritas.demo.plot_trajectory"),
        patch("veritas.demo.plot_layer_agreement"),
    ):
        gen, df, hm, traj, ag = _run_callback("EleutherAI/pythia-1.4b", "France is", 5, 0.0)

    assert isinstance(gen, str)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(hm, str) and isinstance(traj, str) and isinstance(ag, str)
