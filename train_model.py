import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Configure MLflow Tracking Server
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Predictive_Forecasting_Engine"
MODEL_NAME = "SpikePredictor_V1"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


# 2. Load or Generate Spatiotemporal Environmental Dataset
def load_spatiotemporal_data(csv_path: str = "historical_data.csv") -> pd.DataFrame:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        return df.sort_values("timestamp").reset_index(drop=True)

    rng = np.random.default_rng(42)
    n_hours = 24 * 120  # 120 days of hourly sensor + weather telemetry
    timestamps = pd.date_range("2026-01-01", periods=n_hours, freq="h")

    hour = timestamps.hour.to_numpy()
    temp = 26 + 7 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 1.5, n_hours)
    humidity = 62 - 14 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 4.0, n_hours)
    pressure = 1012 + 4 * np.cos(2 * np.pi * np.arange(n_hours) / (24 * 7)) + rng.normal(0, 1.2, n_hours)
    wind_speed = np.clip(8 + 3 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 2.0, n_hours), 0.5, None)

    # Simulate realistic PM2.5 dynamics with diurnal traffic/inversion spikes
    pm25 = np.zeros(n_hours)
    pm25[0] = 65.0
    for t in range(1, n_hours):
        rush_hour_boost = 18.0 if hour[t] in (8, 9, 20, 21, 22) else 0.0
        inversion_effect = 0.65 * humidity[t] - 1.8 * wind_speed[t] - 0.5 * temp[t]
        noise = rng.normal(0, 6.5)
        pm25[t] = np.clip(0.72 * pm25[t - 1] + 0.35 * inversion_effect + rush_hour_boost + noise + 22, 10, 450)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "temp": temp,
        "humidity": humidity,
        "pressure": pressure,
        "wind_speed": wind_speed,
        "pm25": pm25,
    })

    # Feature Engineering: cyclical time + autoregressive lags + rolling window
    df["hour_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["pm25_lag_1"] = df["pm25"].shift(1)
    df["pm25_lag_3"] = df["pm25"].shift(3)
    df["pm25_rolling_mean_6h"] = df["pm25"].shift(1).rolling(window=6).mean()

    return df.dropna().reset_index(drop=True)


def save_diagnostic_plot(y_true, y_pred, run_name: str) -> str:
    os.makedirs("artifacts", exist_ok=True)
    plot_path = os.path.join("artifacts", f"{run_name}_forecast.png")

    plt.figure(figsize=(10, 4))
    plt.plot(y_true[:150], label="Actual PM2.5", linewidth=1.6)
    plt.plot(y_pred[:150], label="Predicted PM2.5", linestyle="--", linewidth=1.6)
    plt.title(f"{run_name} — Chronological Test Window (First 150 Hours)")
    plt.xlabel("Test Hour Index")
    plt.ylabel("Concentration (µg/m³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def main():
    df = load_spatiotemporal_data()

    feature_cols = [
        "temp", "humidity", "pressure", "wind_speed",
        "hour_sin", "hour_cos",
        "pm25_lag_1", "pm25_lag_3", "pm25_rolling_mean_6h",
    ]
    X = df[feature_cols]
    y = df["pm25"]

    # Chronological 80/20 split (no random shuffling on time-series data)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    candidates = {
        "Ridge_Baseline": (
            Ridge(alpha=1.0),
            {"model_type": "Ridge", "alpha": 1.0},
        ),
        "RandomForest_Tuned": (
            RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=3, random_state=42),
            {"model_type": "RandomForestRegressor", "n_estimators": 150, "max_depth": 12, "min_samples_leaf": 3},
        ),
        "GradientBoosting_Tuned": (
            GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
            {"model_type": "GradientBoostingRegressor", "n_estimators": 200, "learning_rate": 0.05, "max_depth": 4},
        ),
    }

    best_rmse = float("inf")
    best_run_id = None
    best_run_name = None

    for run_name, (model, params) in candidates.items():
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(params)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            mae = float(mean_absolute_error(y_test, preds))
            r2 = float(r2_score(y_test, preds))

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            plot_path = save_diagnostic_plot(y_test.to_numpy(), preds, run_name)
            mlflow.log_artifact(plot_path, artifact_path="diagnostics")

            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(f"[{run_name}] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_run_id = run.info.run_id
                best_run_name = run_name

    # Register the winning model and tag it with the @champion alias
    model_uri = f"runs:/{best_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=mv.version,
    )
    print(f"\nPromoted '{best_run_name}' (Run {best_run_id}) -> {MODEL_NAME} v{mv.version} (@champion)")


if __name__ == "__main__":
    main()