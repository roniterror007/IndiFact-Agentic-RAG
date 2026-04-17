from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class ComputeStats:
    latency_ms: float
    input_bytes: int
    output_bytes: int
    estimated_tokens_in: int
    estimated_tokens_out: int
    flops_per_byte: float


def estimate_tokens(text: str) -> int:
    # Rough cross-model estimate suitable for relative benchmarking.
    return max(1, len(text) // 4)


def estimate_flops_per_byte(tokens: int) -> float:
    # Approximate decoder inference FLOPs using model parameter count.
    model_params = float(os.getenv("CRITIC_MODEL_PARAMS", "8000000000"))
    estimated_flops = 2.0 * model_params * max(1, tokens)
    return estimated_flops / float(max(1, tokens * 4))


def profile_call(func, *args, **kwargs) -> tuple[Any, ComputeStats]:
    input_text = kwargs.get("prompt", "") if isinstance(kwargs, dict) else ""
    input_bytes = len(str(input_text).encode("utf-8"))
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    output_text = str(result)
    output_bytes = len(output_text.encode("utf-8"))

    tokens_in = estimate_tokens(str(input_text))
    tokens_out = estimate_tokens(output_text)
    flops_per_byte = estimate_flops_per_byte(tokens_in + tokens_out)

    return (
        result,
        ComputeStats(
            latency_ms=latency_ms,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            estimated_tokens_in=tokens_in,
            estimated_tokens_out=tokens_out,
            flops_per_byte=flops_per_byte,
        ),
    )


def profile_pipeline_call(func, input_text: str, *args, **kwargs) -> tuple[Any, ComputeStats]:
    """Profile non-prompt pipeline calls by explicitly providing input text."""
    input_bytes = len(str(input_text).encode("utf-8"))
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    output_text = str(result)
    output_bytes = len(output_text.encode("utf-8"))
    tokens_in = estimate_tokens(str(input_text))
    tokens_out = estimate_tokens(output_text)
    flops_per_byte = estimate_flops_per_byte(tokens_in + tokens_out)

    return (
        result,
        ComputeStats(
            latency_ms=latency_ms,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
            estimated_tokens_in=tokens_in,
            estimated_tokens_out=tokens_out,
            flops_per_byte=flops_per_byte,
        ),
    )


def to_dict(stats: ComputeStats) -> Dict[str, Any]:
    return {
        "latency_ms": round(stats.latency_ms, 2),
        "input_bytes": stats.input_bytes,
        "output_bytes": stats.output_bytes,
        "estimated_tokens_in": stats.estimated_tokens_in,
        "estimated_tokens_out": stats.estimated_tokens_out,
        "flops_per_byte": stats.flops_per_byte,
    }
