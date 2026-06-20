"""Stable Pydantic v2 schema for veritas audit JSON output."""

from __future__ import annotations

from pydantic import BaseModel


class TokenFeatureRecord(BaseModel):
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
    per_layer_top1: list[int]
    per_layer_chosen_prob: list[float]


class TokenRiskRecord(BaseModel):
    token_str: str
    position: int
    risk: float
    features: TokenFeatureRecord


class SpanRiskRecord(BaseModel):
    span_text: str
    risk_max: float
    risk_mean: float
    token_risks: list[TokenRiskRecord]


class AuditResult(BaseModel):
    model_name: str
    prompt: str
    generation: str
    timestamp: str       
    seed: int
    weights_used: list[float]
    token_risks: list[TokenRiskRecord]
    span_risks: list[SpanRiskRecord]
