from __future__ import annotations

import functools
import time
from contextlib import contextmanager

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Histogram

from .config import COST_PER_TOKEN, DEFAULT_LATENCY_BUCKETS


class _CallTracker:
    __slots__ = ("input_tokens", "output_tokens", "error", "error_type")

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.error = False
        self.error_type = ""

    def set_tokens(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class LLMMetrics:
    """Prometheus metrics collector for LLM API call monitoring.

    Supports context manager, decorator, and manual recording interfaces
    for tracking request latency, token usage, cost, and errors.
    """

    def __init__(self, prefix: str = "", registry: CollectorRegistry | None = None):
        self._registry = registry or REGISTRY
        pfx = f"{prefix}_" if prefix else ""

        self._requests = Counter(
            f"{pfx}llm_requests",
            "Total LLM API calls",
            ["provider", "model", "status"],
            registry=self._registry,
        )
        self._tokens = Counter(
            f"{pfx}llm_tokens",
            "Total tokens consumed",
            ["provider", "model", "direction"],
            registry=self._registry,
        )
        self._duration = Histogram(
            f"{pfx}llm_request_duration_seconds",
            "LLM request latency in seconds",
            ["provider", "model"],
            buckets=DEFAULT_LATENCY_BUCKETS,
            registry=self._registry,
        )
        self._cost = Counter(
            f"{pfx}llm_cost_dollars",
            "Estimated cost in USD",
            ["provider", "model"],
            registry=self._registry,
        )
        self._errors = Counter(
            f"{pfx}llm_errors",
            "Total LLM errors",
            ["provider", "model", "error_type"],
            registry=self._registry,
        )

    @contextmanager
    def track_call(self, provider: str, model: str):
        """Context manager that tracks an LLM API call.

        Yields a tracker object — call tracker.set_tokens() inside the
        block to record token usage. Duration and error status are
        captured automatically.
        """
        tracker = _CallTracker()
        start = time.perf_counter()
        try:
            yield tracker
        except Exception as exc:
            tracker.error = True
            tracker.error_type = type(exc).__name__
            raise
        finally:
            duration = time.perf_counter() - start
            # track duration even on error — helps identify slow failures
            self._record_from_tracker(provider, model, tracker, duration)

    def track(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            provider = kwargs.get("provider", "unknown")
            model = kwargs.get("model", "unknown")
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                self._requests.labels(provider=provider, model=model, status="success").inc()
                return result
            except Exception as exc:
                self._requests.labels(provider=provider, model=model, status="error").inc()
                self._errors.labels(
                    provider=provider, model=model, error_type=type(exc).__name__
                ).inc()
                raise
            finally:
                self._duration.labels(provider=provider, model=model).observe(
                    time.perf_counter() - start
                )

        return wrapper

    def record_call(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0,
        error: bool = False,
        error_type: str = "",
    ):
        """Manually record a completed LLM API call with all metrics."""
        status = "error" if error else "success"
        self._requests.labels(provider=provider, model=model, status=status).inc()

        if latency_ms > 0:
            self._duration.labels(provider=provider, model=model).observe(latency_ms / 1000)

        if input_tokens:
            self._tokens.labels(provider=provider, model=model, direction="input").inc(
                input_tokens
            )
        if output_tokens:
            self._tokens.labels(provider=provider, model=model, direction="output").inc(
                output_tokens
            )

        if input_tokens or output_tokens:
            cost = self._calculate_cost(provider, model, input_tokens, output_tokens)
            self._cost.labels(provider=provider, model=model).inc(cost)

        if error:
            etype = error_type or "unknown"
            self._errors.labels(provider=provider, model=model, error_type=etype).inc()

    def _record_from_tracker(
        self, provider: str, model: str, tracker: _CallTracker, duration: float
    ):
        status = "error" if tracker.error else "success"
        self._requests.labels(provider=provider, model=model, status=status).inc()
        self._duration.labels(provider=provider, model=model).observe(duration)

        if tracker.input_tokens:
            self._tokens.labels(provider=provider, model=model, direction="input").inc(
                tracker.input_tokens
            )
        if tracker.output_tokens:
            self._tokens.labels(provider=provider, model=model, direction="output").inc(
                tracker.output_tokens
            )

        if tracker.input_tokens or tracker.output_tokens:
            cost = self._calculate_cost(
                provider, model, tracker.input_tokens, tracker.output_tokens
            )
            self._cost.labels(provider=provider, model=model).inc(cost)

        if tracker.error:
            self._errors.labels(
                provider=provider, model=model, error_type=tracker.error_type or "unknown"
            ).inc()

    def _calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        pricing = COST_PER_TOKEN.get(provider, {}).get(model)
        if not pricing:
            return 0.0
        return input_tokens * pricing["input"] + output_tokens * pricing["output"]
