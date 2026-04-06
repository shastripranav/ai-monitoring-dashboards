from __future__ import annotations

# pricing per token, not per 1K — avoids division in hot path
COST_PER_TOKEN: dict[str, dict[str, dict[str, float]]] = {
    "openai": {
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
    },
    "anthropic": {
        "claude-3-5-sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-3-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
    },
}

DEFAULT_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 30.0)

DEFAULT_ITERATION_BUCKETS = (1, 2, 3, 4, 5, 7, 10, 15, 20)

DEFAULT_PREFIX = ""
