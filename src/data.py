"""Data loading and preprocessing.

Mirrors notebook STEP 2-3: load CSV, drop ID column, impute missing
values (median for numeric, mode for boolean), and cast booleans to int.
"""
import pandas as pd

from . import config


def load_raw(path=config.DATA_PATH) -> pd.DataFrame:
    """Load the raw delivery analytics CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from Kaggle (see README) and place it in the data/ folder."
        )
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe.

    - Drop the order ID column.
    - Fill numeric NaNs with the column median.
    - Fill boolean NaNs with the column mode.
    - Cast boolean columns to int.
    """
    df = df.copy()

    if config.ID_COL in df.columns:
        df = df.drop(columns=[config.ID_COL])

    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    bool_cols = df.select_dtypes(include="bool").columns
    for col in bool_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Boolean -> int for model compatibility.
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)

    return df
