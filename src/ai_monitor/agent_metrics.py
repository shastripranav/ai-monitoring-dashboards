from __future__ import annotations

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Histogram

from .config import DEFAULT_ITERATION_BUCKETS, DEFAULT_LATENCY_BUCKETS


class AgentMetrics:
    """Prometheus instrumentation for multi-agent orchestration systems."""

    def __init__(self, prefix: str = "", registry: CollectorRegistry | None = None):
        self._registry = registry or REGISTRY
        pfx = f"{prefix}_" if prefix else ""

        self._routing = Counter(
            f"{pfx}agent_routing",
            "Routing decisions to agents",
            ["agent_name", "from_supervisor"],
            registry=self._registry,
        )
        self._response_duration = Histogram(
            f"{pfx}agent_response_duration",
            "Agent response time in seconds",
            ["agent_name"],
            buckets=DEFAULT_LATENCY_BUCKETS,
            registry=self._registry,
        )
        self._iterations = Histogram(
            f"{pfx}agent_iterations_total",
            "Supervisor routing hops per query",
            [],
            buckets=DEFAULT_ITERATION_BUCKETS,
            registry=self._registry,
        )
        self._errors = Counter(
            f"{pfx}agent_errors",
            "Agent processing errors",
            ["agent_name", "error_type"],
            registry=self._registry,
        )
        self._fallbacks = Counter(
            f"{pfx}agent_fallback",
            "Fallback trigger count",
            ["agent_name"],
            registry=self._registry,
        )

        # TODO: add support for custom supervisor names

    def record_routing(self, agent_name: str, supervisor: str = "default"):
        """Record a routing decision to a specific agent."""
        self._routing.labels(agent_name=agent_name, from_supervisor=supervisor).inc()

    def observe_response(self, agent_name: str, duration_seconds: float):
        self._response_duration.labels(agent_name=agent_name).observe(duration_seconds)

    def observe_iterations(self, count: int):
        """Record the number of supervisor hops for a single query resolution."""
        self._iterations.observe(count)

    def record_error(self, agent_name: str, error_type: str = "unknown"):
        self._errors.labels(agent_name=agent_name, error_type=error_type).inc()

    def record_fallback(self, agent_name: str):
        self._fallbacks.labels(agent_name=agent_name).inc()
