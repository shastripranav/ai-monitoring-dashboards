from prometheus_client import CollectorRegistry

from ai_monitor import ModelMetrics


class TestModelMetrics:
    def setup_method(self):
        self.registry = CollectorRegistry()
        self.metrics = ModelMetrics(prefix="test", registry=self.registry)

    def test_record_prediction(self):
        self.metrics.record_prediction(
            model_name="sentiment-v2", version="2.1.0", latency_seconds=0.05,
        )
        val = self.registry.get_sample_value(
            "test_model_predictions_total",
            {"model_name": "sentiment-v2", "model_version": "2.1.0"},
        )
        assert val == 1.0

    def test_set_accuracy(self):
        self.metrics.set_accuracy("sentiment-v2", 0.94)
        val = self.registry.get_sample_value(
            "test_model_accuracy_score", {"model_name": "sentiment-v2"},
        )
        assert val == 0.94

    def test_set_drift(self):
        self.metrics.set_drift("sentiment-v2", "text_length", 0.12)
        val = self.registry.get_sample_value(
            "test_model_feature_drift",
            {"model_name": "sentiment-v2", "feature_name": "text_length"},
        )
        assert val == 0.12

    def test_record_prediction_with_error(self):
        self.metrics.record_prediction(
            model_name="toxicity-clf", version="1.0", latency_seconds=0.1,
            error=True, error_type="shape_mismatch",
        )
        err = self.registry.get_sample_value(
            "test_model_prediction_errors_total",
            {"model_name": "toxicity-clf", "error_type": "shape_mismatch"},
        )
        assert err == 1.0

    # TODO: add edge case tests for zero-latency predictions
