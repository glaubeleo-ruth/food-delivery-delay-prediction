"""Central configuration: paths, target column, and split settings."""
from pathlib import Path

# Project root (one level above src/)
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"

# Place the Kaggle CSV here (see README for the download link).
DATA_PATH = DATA_DIR / "food_delivery_analytics_cleaned.csv"

TARGET = "delayed_delivery_flag"
# Columns dropped from features to avoid target leakage.
LEAKAGE_COLS = ["cancellation_flag", "refund_flag"]
ID_COL = "order_id"

TEST_SIZE = 0.20
RANDOM_STATE = 42

# Ensure output dirs exist when the pipeline runs.
for _d in (MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
