from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
from transformer_lens import HookedTransformer

_USE_MPS = os.environ.get("VERITAS_ALLOW_MPS") == "1"
_FORCE_F32 = os.environ.get("VERITAS_FLOAT32") == "1"

_SUPPORTED_PREFIXES = (
    "EleutherAI/pythia",
    "meta-llama/Llama-2",
    "gpt2",
    "EleutherAI/gpt-neo",
    "EleutherAI/gpt-j",
)


def _default_device() -> str:
    if _USE_MPS and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_dtype() -> torch.dtype:
    if _FORCE_F32:
        return torch.float32
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        return torch.float16
    return torch.float32


@dataclass
class GenerationResult:
    prompt_tokens: list[int]
    generated_tokens: list[int]
    generated_strs: list[str]
    residuals: list[torch.Tensor] = field(default_factory=list)
    final_logits: list[torch.Tensor] = field(default_factory=list)


def load_model(model_name: str, device: str | None = None) -> HookedTransformer:
    if not any(model_name.startswith(p) for p in _SUPPORTED_PREFIXES):
        raise NotImplementedError(
            f"'{model_name}' not supported. Add it to _SUPPORTED_PREFIXES "
            "after verifying TransformerLens compatibility."
        )
    if device is None:
        device = _default_device()
    model = HookedTransformer.from_pretrained(model_name, dtype=_default_dtype())
    model.eval()
    model.to(device)
    return model


def generate(
    model: HookedTransformer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    seed: int = 42,
    capture_residuals: bool = True,
) -> GenerationResult:
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers
    eos = model.tokenizer.eos_token_id

    prompt_tokens: list[int] = model.to_tokens(prompt, prepend_bos=True).squeeze(0).tolist()
    ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    gen_tokens: list[int] = []
    gen_strs: list[str] = []
    residuals: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if capture_residuals:
                logits, cache = model.run_with_cache(
                    ids,
                    names_filter=lambda name: name.endswith("resid_post"),
                )
                step_logits = logits[0, -1, :]
                layer_residuals = torch.stack(
                    [cache[f"blocks.{i}.hook_resid_post"][0, -1, :] for i in range(n_layers)]
                )
                residuals.append(layer_residuals.cpu())
                logits_list.append(step_logits.float().cpu())
            else:
                logits = model(ids)
                step_logits = logits[0, -1, :]

            if temperature == 0.0:
                next_id = int(step_logits.argmax().item())
            else:
                probs = torch.softmax(step_logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            gen_tokens.append(next_id)
            gen_strs.append(model.to_string([next_id]))

            if next_id == eos:
                break

            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    return GenerationResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=gen_tokens,
        generated_strs=gen_strs,
        residuals=residuals,
        final_logits=logits_list,
    )
