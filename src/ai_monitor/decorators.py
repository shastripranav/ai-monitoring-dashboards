from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm_metrics import LLMMetrics
    from .model_metrics import ModelMetrics


def track_llm_call(
    metrics: LLMMetrics,
    provider: str = "unknown",
    model: str = "unknown",
):
    """Decorator factory for wrapping functions that make LLM API calls.

    Usage:
        @track_llm_call(llm_metrics, provider="openai", model="gpt-4o")
        def ask_gpt(prompt):
            return client.chat(prompt)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                metrics.record_call(
                    provider=provider,
                    model=model,
                    latency_ms=duration * 1000,
                )
                return result
            except Exception as exc:
                duration = time.perf_counter() - start
                metrics.record_call(
                    provider=provider,
                    model=model,
                    latency_ms=duration * 1000,
                    error=True,
                    error_type=type(exc).__name__,
                )
                raise

        return wrapper

    return decorator


def track_inference(
    metrics: ModelMetrics,
    model_name: str = "unknown",
    version: str = "unknown",
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                metrics.record_prediction(
                    model_name=model_name,
                    version=version,
                    latency_seconds=time.perf_counter() - start,
                )
                return result
            except Exception as exc:
                metrics.record_prediction(
                    model_name=model_name,
                    version=version,
                    latency_seconds=time.perf_counter() - start,
                    error=True,
                    error_type=type(exc).__name__,
                )
                raise

        return wrapper

    return decorator
