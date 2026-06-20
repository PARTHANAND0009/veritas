from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from veritas.schema import AuditResult, SpanRiskRecord, TokenFeatureRecord, TokenRiskRecord


def _feat(**kw) -> TokenFeatureRecord:
    d = dict(
        token_id=1, token_str=" Paris", position=0,
        max_prob=0.9, entropy=0.5, margin=0.8, log_prob_chosen=-0.1,
        layer_agreement=1.0, crystallization_depth=0.3, volatility_count=1,
        prob_variance=0.001, per_layer_top1=[1] * 24, per_layer_chosen_prob=[0.9] * 24,
    )
    d.update(kw)
    return TokenFeatureRecord(**d)


def _trisk(token_str: str = " Paris", risk: float = 0.2) -> TokenRiskRecord:
    return TokenRiskRecord(token_str=token_str, position=0, risk=risk, features=_feat(token_str=token_str))


def _audit(**kw) -> AuditResult:
    tr = _trisk()
    span = SpanRiskRecord(span_text="Paris", risk_max=0.2, risk_mean=0.2, token_risks=[tr])
    d = dict(
        model_name="EleutherAI/pythia-1.4b",
        prompt="The capital of France is",
        generation=" Paris",
        timestamp=datetime.now(timezone.utc).isoformat(),
        seed=42,
        weights_used=[-1.5, 0.8, -1.0, -0.8, -1.2, 0.6, 0.4, 0.5],
        token_risks=[tr],
        span_risks=[span],
    )
    d.update(kw)
    return AuditResult(**d)


def test_audit_result_roundtrip() -> None:
    orig = _audit()
    restored = AuditResult.model_validate_json(orig.model_dump_json(indent=2))
    assert restored.model_name == orig.model_name
    assert restored.seed == orig.seed
    assert restored.weights_used == orig.weights_used
    assert len(restored.span_risks) == len(orig.span_risks)


def test_audit_result_timestamp_is_iso() -> None:
    assert datetime.fromisoformat(_audit().timestamp).year >= 2024


def test_span_risks_nonempty() -> None:
    r = _audit()
    assert len(r.span_risks) >= 1 and r.span_risks[0].span_text


def test_compare_command_prints_diff(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from veritas.cli import cli

    low = _audit(span_risks=[SpanRiskRecord(
        span_text="Paris", risk_max=0.1, risk_mean=0.1, token_risks=[_trisk(" Paris", 0.1)]
    )])
    high = _audit(span_risks=[SpanRiskRecord(
        span_text="Lyon", risk_max=0.8, risk_mean=0.8, token_risks=[_trisk(" Lyon", 0.8)]
    )])

    f1, f2 = tmp_path / "low.json", tmp_path / "high.json"
    f1.write_text(low.model_dump_json(indent=2))
    f2.write_text(high.model_dump_json(indent=2))

    result = CliRunner().invoke(cli, ["compare", str(f1), str(f2)])
    assert result.exit_code == 0
    assert "0.1" in result.output and "0.8" in result.output
