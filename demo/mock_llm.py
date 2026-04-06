from __future__ import annotations

import asyncio
import random

MODELS = {
    ("openai", "gpt-4o"): {"latency": (0.5, 2.0), "tokens": (200, 800)},
    ("openai", "gpt-4o-mini"): {"latency": (0.1, 0.5), "tokens": (50, 400)},
    ("anthropic", "claude-3-5-sonnet"): {"latency": (0.4, 1.8), "tokens": (100, 600)},
}


async def simulate_llm_call(provider: str, model: str) -> dict:
    cfg = MODELS.get((provider, model), {"latency": (0.2, 1.0), "tokens": (100, 500)})

    latency = random.uniform(*cfg["latency"])
    # sleep a fraction of the simulated latency so we don't block the loop too long
    await asyncio.sleep(latency * 0.1)

    input_toks = random.randint(*cfg["tokens"])
    output_toks = random.randint(input_toks // 4, input_toks)

    is_error = random.random() < 0.02
    err_type = random.choice(["timeout", "rate_limit", "server_error"]) if is_error else None

    return {
        "latency": latency,
        "input_tokens": input_toks,
        "output_tokens": output_toks,
        "error": is_error,
        "error_type": err_type,
    }
