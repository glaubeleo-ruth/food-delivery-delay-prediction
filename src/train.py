"""Model training (notebook STEP 6).

Builds the feature matrix, splits train/test, scales features for KNN,
trains KNN and a class-imbalance-aware XGBoost, and saves all artifacts
to the models/ directory.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from . import config
from .data import load_raw, preprocess
from .features import add_features


def build_xy(df: pd.DataFrame):
    """Split a processed dataframe into features X and target y."""
    drop_cols = [config.TARGET, *config.LEAKAGE_COLS]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df[config.TARGET]
    return X, y


def get_data():
    """Full data pipeline: load -> preprocess -> features -> X/y."""
    df = add_features(preprocess(load_raw()))
    return build_xy(df)


def split(X, y):
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def train_knn(X_train_scaled, y_train, n_neighbors: int = 5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train_scaled, y_train)
    return knn


def train_xgboost(X_train, y_train):
    # Weight the minority (delayed) class to counter imbalance.
    counts = y_train.value_counts()
    scale_pos_weight = counts[0] / counts[1]

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=config.RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    return xgb


def main():
    X, y = get_data()
    print(f"Features (X): {X.shape[1]} columns")
    print(f"Target   (y): {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = split(X, y)
    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # Scaling is required for the distance-based KNN.
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("\n[Training KNN...]")
    knn = train_knn(X_train_sc, y_train)

    print("[Training XGBoost...]")
    xgb = train_xgboost(X_train, y_train)

    # Persist artifacts for evaluation / inference.
    joblib.dump(knn, config.MODELS_DIR / "knn.joblib")
    joblib.dump(xgb, config.MODELS_DIR / "xgboost.joblib")
    joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")
    joblib.dump(list(X.columns), config.MODELS_DIR / "feature_names.joblib")
    print(f"\nSaved models to {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
