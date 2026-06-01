"""Feature engineering (notebook STEP 5).

Adds three derived features:
- stress_score:   traffic x weather severity (delivery difficulty)
- is_peak_hour:   1 if order placed during lunch (11-14) or dinner (18-22)
- delivery_speed: distance / (time + 1), km per minute
"""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with engineered features added."""
    df = df.copy()

    # High traffic + bad weather = more stress on delivery.
    df["stress_score"] = df["traffic_level_score"] * df["weather_severity_score"]

    # Peak meal windows.
    df["is_peak_hour"] = (
        df["order_hour"].between(11, 14) | df["order_hour"].between(18, 22)
    ).astype(int)

    # Partner speed in km per minute (+1 avoids divide-by-zero).
    df["delivery_speed"] = df["delivery_distance_km"] / (df["delivery_time_minutes"] + 1)

    return df
