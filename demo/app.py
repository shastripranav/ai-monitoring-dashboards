from __future__ import annotations

import asyncio
import os
import random
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ai_monitor import AgentMetrics, LLMMetrics, ModelMetrics

from .mock_agents import simulate_routing
from .mock_llm import simulate_llm_call
from .mock_model import simulate_prediction

prefix = os.getenv("METRICS_PREFIX", "demo")
_sim_interval = float(os.getenv("SIMULATION_INTERVAL", "2.0"))

llm = LLMMetrics(prefix=prefix)
ml = ModelMetrics(prefix=prefix)
agent = AgentMetrics(prefix=prefix)

_LLM_MODELS = [
    ("openai", "gpt-4o"),
    ("openai", "gpt-4o-mini"),
    ("anthropic", "claude-3-5-sonnet"),
]
_ML_MODELS = [
    ("sentiment-v2", "2.1.0"),
    ("toxicity-clf", "1.4.2"),
]

_tasks: list[asyncio.Task] = []


async def _run_llm_simulation():
    """Generates realistic LLM API call metrics in a loop."""
    while True:
        provider, model_name = random.choice(_LLM_MODELS)
        result = await simulate_llm_call(provider, model_name)
        llm.record_call(
            provider=provider,
            model=model_name,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            latency_ms=result["latency"] * 1000,
            error=result["error"],
            error_type=result.get("error_type", ""),
        )
        await asyncio.sleep(random.uniform(_sim_interval * 0.25, _sim_interval * 1.5))


async def _run_model_simulation():
    while True:
        model_name, version = random.choice(_ML_MODELS)
        result = simulate_prediction(model_name, version)

        ml.record_prediction(
            model_name=result["model_name"],
            version=result["version"],
            latency_seconds=result["latency"],
            error=result["error"],
        )
        ml.set_accuracy(result["model_name"], result["accuracy"])

        for feat, score in result["feature_drift"].items():
            ml.set_drift(result["model_name"], feat, score)

        await asyncio.sleep(random.uniform(_sim_interval * 0.15, _sim_interval * 0.75))


async def _run_agent_simulation():
    while True:
        result = simulate_routing()
        agent.record_routing(result["agent_name"])
        agent.observe_response(result["agent_name"], result["latency"])
        agent.observe_iterations(result["iterations"])

        if result["error"]:
            agent.record_error(result["agent_name"], result["error_type"] or "unknown")
        if result["fallback"]:
            agent.record_fallback(result["agent_name"])

        await asyncio.sleep(random.uniform(_sim_interval * 0.25, _sim_interval))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # stagger tasks to avoid metric bursts on startup
    _tasks.append(asyncio.create_task(_run_llm_simulation()))
    await asyncio.sleep(0.3)
    _tasks.append(asyncio.create_task(_run_model_simulation()))
    await asyncio.sleep(0.3)
    _tasks.append(asyncio.create_task(_run_agent_simulation()))
    yield
    for t in _tasks:
        t.cancel()


app = FastAPI(title="AI Monitoring Demo", lifespan=lifespan)


@app.get("/metrics")
async def metrics_endpoint():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/simulate")
async def simulate_burst(count: int = 20):
    """Trigger a burst of simulated activity for demo purposes."""
    for _ in range(count):
        provider, model_name = random.choice(_LLM_MODELS)
        result = await simulate_llm_call(provider, model_name)
        llm.record_call(
            provider=provider,
            model=model_name,
            input_tokens=result["input_tokens"],
            output_tokens=result["output_tokens"],
            latency_ms=result["latency"] * 1000,
            error=result["error"],
            error_type=result.get("error_type", ""),
        )

        routing = simulate_routing()
        agent.record_routing(routing["agent_name"])
        agent.observe_response(routing["agent_name"], routing["latency"])

    return {"simulated": count, "types": ["llm", "agent"]}
