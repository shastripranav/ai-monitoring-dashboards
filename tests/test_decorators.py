from prometheus_client import CollectorRegistry

from ai_monitor import LLMMetrics, ModelMetrics
from ai_monitor.decorators import track_inference, track_llm_call


class TestDecorators:
    def test_track_llm_call_records_success(self):
        registry = CollectorRegistry()
        metrics = LLMMetrics(prefix="dtest", registry=registry)

        @track_llm_call(metrics, provider="openai", model="gpt-4o")
        def fake_llm():
            return "response"

        result = fake_llm()
        assert result == "response"

        val = registry.get_sample_value(
            "dtest_llm_requests_total",
            {"provider": "openai", "model": "gpt-4o", "status": "success"},
        )
        assert val == 1.0

    def test_track_llm_call_records_errors(self):
        registry = CollectorRegistry()
        metrics = LLMMetrics(prefix="derr", registry=registry)

        @track_llm_call(metrics, provider="openai", model="gpt-4o")
        def failing_call():
            raise RuntimeError("API down")

        try:
            failing_call()
        except RuntimeError:
            pass

        err = registry.get_sample_value(
            "derr_llm_errors_total",
            {"provider": "openai", "model": "gpt-4o", "error_type": "RuntimeError"},
        )
        assert err == 1.0

    def test_track_inference_records_prediction(self):
        registry = CollectorRegistry()
        metrics = ModelMetrics(prefix="dinf", registry=registry)

        @track_inference(metrics, model_name="sentiment", version="1.0")
        def predict(text):
            return {"label": "positive"}

        result = predict("hello world")
        assert result["label"] == "positive"

        val = registry.get_sample_value(
            "dinf_model_predictions_total",
            {"model_name": "sentiment", "model_version": "1.0"},
        )
        assert val == 1.0

    def test_track_inference_records_latency(self):
        registry = CollectorRegistry()
        metrics = ModelMetrics(prefix="dlat", registry=registry)

        @track_inference(metrics, model_name="toxicity", version="2.0")
        def slow_predict():
            import time
            time.sleep(0.02)
            return 0.85

        slow_predict()
        count = registry.get_sample_value(
            "dlat_model_prediction_latency_count", {"model_name": "toxicity"},
        )
        assert count == 1.0
