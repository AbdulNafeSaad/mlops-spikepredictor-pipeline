import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

TRACKING_URI = "http://localhost:5000"
MODEL_URI = "models:/SpikePredictor_V1@champion"
SPIKE_THRESHOLD = 150.0  # µg/m³ alert threshold


def build_incoming_batch() -> pd.DataFrame:
    """Simulates 5 incoming hourly sensor packets awaiting next-hour inference."""
    timestamps = pd.to_datetime([
        "2026-05-01 07:00:00",
        "2026-05-01 08:00:00",
        "2026-05-01 14:00:00",
        "2026-05-01 20:00:00",
        "2026-05-01 21:00:00",
    ])
    hours = timestamps.hour

    return pd.DataFrame({
        "temp": [23.5, 25.0, 33.2, 27.1, 25.8],
        "humidity": [72.0, 68.0, 44.0, 66.0, 70.0],
        "pressure": [1013.2, 1013.5, 1010.8, 1012.4, 1012.9],
        "wind_speed": [4.2, 3.8, 11.5, 4.0, 3.1],
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
        "pm25_lag_1": [118.0, 142.0, 74.0, 148.0, 176.0],
        "pm25_lag_3": [95.0, 108.0, 82.0, 119.0, 138.0],
        "pm25_rolling_mean_6h": [102.5, 115.0, 79.4, 124.2, 145.6],
    })


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"Loading champion model from {MODEL_URI}...")

    model = mlflow.sklearn.load_model(MODEL_URI)
    batch = build_incoming_batch()
    predictions = model.predict(batch)

    results = batch[["temp", "humidity", "wind_speed", "pm25_lag_1"]].copy()
    results["predicted_pm25"] = np.round(predictions, 2)
    results["spike_alert"] = np.where(results["predicted_pm25"] >= SPIKE_THRESHOLD, "HIGH_SPIKE", "NORMAL")

    print("\n--- Registry Inference Output ---")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()