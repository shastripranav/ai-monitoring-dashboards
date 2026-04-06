# AI/ML Monitoring Dashboard Templates

Pre-built Grafana dashboards and a Python Prometheus metrics library for monitoring LLM APIs, ML model performance, and multi-agent orchestration.

Ships as a Docker Compose stack — `docker compose up` gives you Prometheus, Grafana, and a demo FastAPI app with three dashboards auto-provisioned and showing live data within 60 seconds.

## Quick Start

```bash
docker compose up --build
```

Then open:
- **Grafana**: [http://localhost:3000](http://localhost:3000) (no login needed, dashboards auto-loaded)
- **Prometheus**: [http://localhost:9090](http://localhost:9090)
- **Demo API**: [http://localhost:8000/health](http://localhost:8000/health)

Trigger a burst of activity: `curl -X POST http://localhost:8000/api/simulate?count=50`

## Dashboards

| Dashboard | What it monitors |
|---|---|
| **LLM API Monitor** | Token usage, cost tracking, latency percentiles, error rates across LLM providers |
| **Model Performance** | Prediction accuracy, feature drift, inference latency, throughput |
| **Agent Orchestration** | Routing distribution, agent response times, supervisor iterations, fallback rates |

## Using the Library

Install the metrics library in your own project:

```bash
pip install -e .
```

### LLM Metrics

```python
from ai_monitor import LLMMetrics

metrics = LLMMetrics(prefix="myapp")

# Context manager — tracks duration automatically
with metrics.track_call(provider="openai", model="gpt-4o") as tracker:
    response = openai.chat.completions.create(...)
    tracker.set_tokens(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )

# Manual recording
metrics.record_call(
    provider="openai",
    model="gpt-4o-mini",
    input_tokens=150,
    output_tokens=340,
    latency_ms=820,
)
```

### Model Metrics

```python
from ai_monitor import ModelMetrics

ml = ModelMetrics(prefix="myapp")

ml.record_prediction(model_name="sentiment-v2", version="2.1.0", latency_seconds=0.045)
ml.set_accuracy("sentiment-v2", 0.94)
ml.set_drift("sentiment-v2", "text_length", 0.08)
```

### Agent Metrics

```python
from ai_monitor import AgentMetrics

agent = AgentMetrics(prefix="myapp")

agent.record_routing("finance")
agent.observe_response("finance", duration_seconds=0.85)
agent.observe_iterations(3)
```

### Decorators

```python
from ai_monitor import LLMMetrics, track_llm_call

metrics = LLMMetrics(prefix="myapp")

@track_llm_call(metrics, provider="openai", model="gpt-4o")
def ask_gpt(prompt: str):
    return client.chat.completions.create(model="gpt-4o", messages=[...])
```

## Architecture

```
┌──────────────┐     scrape /metrics     ┌────────────┐     query      ┌─────────┐
│  Demo App    │ ◄───────────────────── │ Prometheus │ ◄──────────── │ Grafana │
│  (FastAPI)   │                         │            │               │         │
│  port 8000   │                         │  port 9090 │               │ port 3k │
└──────────────┘                         └────────────┘               └─────────┘
       │
       ├── ai_monitor.LLMMetrics      → llm_requests_total, llm_tokens_total, ...
       ├── ai_monitor.ModelMetrics    → model_predictions_total, model_accuracy_score, ...
       └── ai_monitor.AgentMetrics   → agent_routing_total, agent_response_duration, ...
```

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `METRICS_PREFIX` | `demo` | Prefix prepended to all Prometheus metric names |
| `SIMULATION_INTERVAL` | `2.0` | Base interval between simulated events (seconds) |

## Development

```bash
pip install -e ".[dev]"
pytest -v
ruff check src/ tests/
```

## License

MIT
