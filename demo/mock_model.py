from __future__ import annotations

import random
import time

_boot_time = time.monotonic()


def simulate_prediction(
    model_name: str = "sentiment-v2", version: str = "2.1.0"
) -> dict:
    latency = max(0.005, random.gauss(0.045, 0.015))

    # drift increases linearly with time to simulate real-world degradation
    elapsed_hours = (time.monotonic() - _boot_time) / 3600
    drift_factor = min(0.15, elapsed_hours * 0.02)

    accuracy = max(0.0, 0.94 - drift_factor + random.gauss(0, 0.008))

    feature_drift = {
        "text_length": abs(drift_factor + random.gauss(0, 0.01)),
        "sentiment_score": abs(drift_factor * 0.8 + random.gauss(0, 0.005)),
    }

    return {
        "model_name": model_name,
        "version": version,
        "latency": latency,
        "accuracy": accuracy,
        "feature_drift": feature_drift,
        "error": random.random() < 0.01,
    }
