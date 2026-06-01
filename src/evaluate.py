"""Model evaluation (notebook STEP 7).

Reproduces the metrics report, ROC comparison, confusion matrix, and
feature-importance plot. Figures are saved to reports/figures/.
"""
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from . import config
from .train import get_data, split

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 100


def evaluate_model(name, model, X_test_data, y_test_data):
    """Print accuracy / AUC / classification report; return (auc, probs)."""
    y_pred = model.predict(X_test_data)
    y_pred_prob = model.predict_proba(X_test_data)[:, 1]

    acc = accuracy_score(y_test_data, y_pred)
    auc = roc_auc_score(y_test_data, y_pred_prob)

    print("=" * 50)
    print(name)
    print("=" * 50)
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  AUC-ROC  : {auc:.4f}\n")
    print(classification_report(y_test_data, y_pred, target_names=["On Time", "Delayed"]))
    return auc, y_pred_prob


def plot_roc(y_test, results, out=config.FIGURES_DIR / "roc_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, prob, auc in results:
        fpr, tpr, _ = roc_curve(y_test, prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", label="Random Guess")
    axes[0].set(title="ROC Curve — All Models",
                xlabel="False Positive Rate", ylabel="True Positive Rate")
    axes[0].legend()

    names = [r[0] for r in results]
    scores = [r[2] for r in results]
    bars = axes[1].bar(names, scores, color=["#3498db", "#2ecc71"], width=0.5)
    axes[1].set(ylim=(0, 1.0), title="AUC-ROC Score Comparison", ylabel="AUC-ROC Score")
    for bar, score in zip(bars, scores):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{score:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


def plot_confusion(y_test, y_pred, out=config.FIGURES_DIR / "confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["On Time", "Delayed"],
                yticklabels=["On Time", "Delayed"], ax=ax)
    ax.set(title="Confusion Matrix — XGBoost (Best Model)",
           xlabel="Predicted Label", ylabel="Actual Label")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


def plot_importance(model, feature_names, out=config.FIGURES_DIR / "feature_importance.png"):
    imp = (pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
           .sort_values("Importance", ascending=False).head(15))
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=imp, x="Importance", y="Feature", palette="viridis", ax=ax)
    ax.set(title="Top 15 Features That Predict Delivery Delays", xlabel="Importance Score")
    plt.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


def main():
    X, y = get_data()
    X_train, X_test, y_train, y_test = split(X, y)

    knn = joblib.load(config.MODELS_DIR / "knn.joblib")
    xgb = joblib.load(config.MODELS_DIR / "xgboost.joblib")
    scaler = joblib.load(config.MODELS_DIR / "scaler.joblib")
    X_test_sc = scaler.transform(X_test)

    auc_knn, prob_knn = evaluate_model("KNN", knn, X_test_sc, y_test)
    auc_xgb, prob_xgb = evaluate_model("XGBoost", xgb, X_test, y_test)

    plot_roc(y_test, [("KNN", prob_knn, auc_knn), ("XGBoost", prob_xgb, auc_xgb)])
    plot_confusion(y_test, xgb.predict(X_test))
    plot_importance(xgb, X.columns)


if __name__ == "__main__":
    main()
