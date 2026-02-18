"""
Data Preprocessing Module
=========================
Normalization, cleaning, feature engineering, and train/val/test splitting
for the digital twin output data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_raw_data(data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load the combined CSV from the digital twin."""
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    return pd.read_csv(Path(data_dir) / "all_homes.csv", parse_dates=["timestamp"])


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing values in numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].interpolate(method="linear").bfill().ffill()
    return df


def detect_outliers(df: pd.DataFrame, columns: list, iqr_factor: float = 1.5) -> pd.DataFrame:
    """Clip outliers using IQR method."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_factor * IQR
        upper = Q3 + iqr_factor * IQR
        df[col] = df[col].clip(lower, upper)
    return df


def add_features(df: pd.DataFrame, rolling_window: int = 4) -> pd.DataFrame:
    """Add time-based and rolling features."""
    # Time encodings (cyclical)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["slot_sin"] = np.sin(2 * np.pi * df["slot_in_day"] / 96)
    df["slot_cos"] = np.cos(2 * np.pi * df["slot_in_day"] / 96)

    # Rolling averages per home
    df["rolling_avg_power"] = (
        df.groupby("home_id")["total_power_kw"]
        .transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
    )
    df["rolling_std_power"] = (
        df.groupby("home_id")["total_power_kw"]
        .transform(lambda x: x.rolling(rolling_window, min_periods=1).std().fillna(0))
    )

    # Lag features
    df["power_lag_1"] = df.groupby("home_id")["total_power_kw"].shift(1).fillna(0)
    df["power_lag_4"] = df.groupby("home_id")["total_power_kw"].shift(4).fillna(0)

    return df


def normalize(df: pd.DataFrame, columns: list,
              method: str = "minmax") -> Tuple[pd.DataFrame, object]:
    """Normalize specified columns. Returns (df, scaler)."""
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.7,
                   val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data temporally (preserving time order) per home."""
    train_dfs, val_dfs, test_dfs = [], [], []

    for home_id in df["home_id"].unique():
        home_df = df[df["home_id"] == home_id].sort_values("timestamp")
        n = len(home_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_dfs.append(home_df.iloc[:train_end])
        val_dfs.append(home_df.iloc[train_end:val_end])
        test_dfs.append(home_df.iloc[val_end:])

    return (pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs))


def get_feature_columns() -> list:
    """Return the list of feature columns used for ML models."""
    appliance_cols = [
        "power_hvac_kw", "power_washer_kw", "power_dryer_kw",
        "power_dishwasher_kw", "power_ev_charger_kw",
        "power_lighting_kw", "power_water_heater_kw"
    ]
    env_cols = ["outdoor_temp_c", "indoor_temp_c", "humidity_pct"]
    time_cols = ["hour_sin", "hour_cos"]
    context_cols = ["price_inr_kwh", "occupancy", "is_weekend",
                    "rolling_avg_power", "rolling_std_power",
                    "power_lag_1", "power_lag_4"]
    return appliance_cols + env_cols + time_cols + context_cols


def run_preprocessing(config: dict, data_dir: Optional[str] = None) -> dict:
    """
    Full preprocessing pipeline.
    Returns dict with train/val/test DataFrames and scaler.
    """
    cfg = config["preprocessing"]

    print("Loading raw data...")
    df = load_raw_data(data_dir)
    print(f"  Loaded {len(df):,} records")

    print("Handling missing values...")
    df = handle_missing_values(df)

    # Outlier detection on power columns
    power_cols = [c for c in df.columns if c.startswith("power_") or c == "total_power_kw"]
    print(f"Detecting outliers in {len(power_cols)} columns...")
    df = detect_outliers(df, power_cols, cfg["outlier_iqr_factor"])

    print("Engineering features...")
    df = add_features(df, cfg["rolling_window_slots"])

    # Normalize
    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]
    print(f"Normalizing {len(available_cols)} features ({cfg['normalization']})...")
    df, scaler = normalize(df, available_cols, cfg["normalization"])

    # Split
    print("Splitting data (temporal)...")
    train_df, val_df, test_df = temporal_split(df, cfg["train_ratio"], cfg["val_ratio"])
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Save processed data
    out_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    print(f"Saved processed data to {out_dir}")

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "scaler": scaler,
        "feature_columns": available_cols,
        "full_df": df,
    }
