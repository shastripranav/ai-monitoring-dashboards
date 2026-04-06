from __future__ import annotations

from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

from .config import DEFAULT_LATENCY_BUCKETS


class ModelMetrics:
    """Tracks ML model prediction performance, accuracy, and feature drift."""

    def __init__(self, prefix: str = "", registry: CollectorRegistry | None = None):
        self._registry = registry or REGISTRY
        pfx = f"{prefix}_" if prefix else ""

        self._predictions = Counter(
            f"{pfx}model_predictions",
            "Total model predictions",
            ["model_name", "model_version"],
            registry=self._registry,
        )
        self._latency = Histogram(
            f"{pfx}model_prediction_latency",
            "Inference latency in seconds",
            ["model_name"],
            buckets=DEFAULT_LATENCY_BUCKETS,
            registry=self._registry,
        )
        self._accuracy = Gauge(
            f"{pfx}model_accuracy_score",
            "Current accuracy/F1 score",
            ["model_name"],
            registry=self._registry,
        )
        # FIXME: drift calculation assumes normal distribution
        self._drift = Gauge(
            f"{pfx}model_feature_drift",
            "Feature drift score from training distribution",
            ["model_name", "feature_name"],
            registry=self._registry,
        )
        self._errors = Counter(
            f"{pfx}model_prediction_errors",
            "Total prediction errors",
            ["model_name", "error_type"],
            registry=self._registry,
        )
        self._info = Gauge(
            f"{pfx}model_info",
            "Model deployment info",
            ["model_name", "model_version"],
            registry=self._registry,
        )

    def record_prediction(
        self,
        model_name: str,
        version: str = "unknown",
        latency_seconds: float = 0,
        error: bool = False,
        error_type: str = "",
    ):
        """Record a single model prediction with timing and error info."""
        self._predictions.labels(model_name=model_name, model_version=version).inc()
        self._info.labels(model_name=model_name, model_version=version).set(1)

        if latency_seconds > 0:
            self._latency.labels(model_name=model_name).observe(latency_seconds)

        if error:
            self._errors.labels(
                model_name=model_name, error_type=error_type or "prediction_error"
            ).inc()

    def set_accuracy(self, model_name: str, score: float):
        self._accuracy.labels(model_name=model_name).set(score)

    def set_drift(self, model_name: str, feature_name: str, score: float):
        self._drift.labels(model_name=model_name, feature_name=feature_name).set(score)
