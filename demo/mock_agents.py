from __future__ import annotations

import random

AGENTS = {
    "finance": {"weight": 0.40, "latency": (0.3, 1.2)},
    "operations": {"weight": 0.35, "latency": (0.2, 0.8)},
    "planning": {"weight": 0.25, "latency": (0.5, 2.0)},
}

_agent_names = list(AGENTS.keys())
_agent_weights = [AGENTS[a]["weight"] for a in _agent_names]


def simulate_routing() -> dict:
    selected = random.choices(_agent_names, weights=_agent_weights, k=1)[0]
    cfg = AGENTS[selected]

    latency = random.uniform(*cfg["latency"])
    iterations = random.choices([1, 2, 3, 4, 5], weights=[40, 30, 15, 10, 5], k=1)[0]

    is_error = random.random() < 0.03
    is_fallback = is_error and random.random() < 0.5

    return {
        "agent_name": selected,
        "latency": latency,
        "iterations": iterations,
        "error": is_error,
        "fallback": is_fallback,
        "error_type": random.choice(["timeout", "parse_error"]) if is_error else None,
    }
