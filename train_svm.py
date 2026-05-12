"""
train_svm.py — ToneFit ML Project
====================================
Traditional CV Baseline: SVM (RBF kernel) on handcrafted color features.

Features used (from preprocess.py → features.csv):
    L_mean  — CIELab Lightness  (light vs dark seasons)
    a_mean  — CIELab a* axis    (directly encodes warm vs cool)
    b_mean  — CIELab b* axis    (yellow-blue)
    ITA     — Individual Typology Angle (skin undertone proxy)
    H_mean  — HSV Hue
    S_mean  — HSV Saturation    (muted vs clear)
    V_mean  — HSV Value         (brightness)

Motivation: Stacchio et al. (2024) identified warm/cool confusion as the
primary failure mode. CIELab a* directly encodes this axis. SVM with RBF
kernel is the established CV baseline for face color classification tasks.

Requires:
    Run preprocess.py first → features.csv + models/scaler.pkl

Usage:
    python train_svm.py

Outputs:
    models/svm_model.pkl           — trained SVM (joblib)
    results/svm_test_report.txt    — classification report on test set
    results/confusion_svm.png      — confusion matrix heatmap
    results/svm_results.json       — metrics for comparison table
"""

import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

FEATURES_CSV  = "features.csv"
SCALER_PATH   = "models/scaler.pkl"
MODEL_OUT     = "models/svm_model.pkl"
REPORT_OUT    = "results/svm_test_report.txt"
CONFUSION_OUT = "results/confusion_svm.png"
RESULTS_OUT   = "results/svm_results.json"

SEASONS   = ["autumn", "spring", "summer", "winter"]
LABEL_MAP = {"autumn": 0, "spring": 1, "summer": 2, "winter": 3}

# 7 color features — same selection used in color fusion scripts
COLOR_COLS     = ["L_mean", "a_mean", "b_mean", "ITA", "H_mean", "S_mean", "V_mean"]
_ALL_FEAT_COLS = [
    "L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std",
    "ITA", "H_mean", "S_mean", "V_mean", "skin_ratio", "blur_score",
]

# SVM hyperparameter search space
PARAM_GRID = {
    "C":     [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1],
}
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_features():
    """
    Load features.csv, apply the preprocess.py scaler, and return
    train/test arrays for the 7 selected color features.
    """
    if not os.path.exists(FEATURES_CSV):
        raise FileNotFoundError(
            f"{FEATURES_CSV} not found. Run preprocess.py first."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"{SCALER_PATH} not found. Run preprocess.py first."
        )

    df = pd.read_csv(FEATURES_CSV)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    col_indices = [_ALL_FEAT_COLS.index(c) for c in COLOR_COLS]
    scaled = scaler.transform(df[_ALL_FEAT_COLS].values.astype(np.float32))

    train_mask = df["split"] == "train"
    test_mask  = df["split"] == "test"

    X_train = scaled[train_mask][:, col_indices]
    y_train = df.loc[train_mask, "label"].values
    X_test  = scaled[test_mask][:, col_indices]
    y_test  = df.loc[test_mask, "label"].values

    return X_train, y_train, X_test, y_test, len(df)


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[s.capitalize() for s in SEASONS],
        yticklabels=[s.capitalize() for s in SEASONS],
        ax=ax,
    )
    ax.set_title("SVM (RBF) — Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  ToneFit — SVM Color Feature Classifier (CV Baseline)")
    print("=" * 60)

    os.makedirs("models",  exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --- Load features -------------------------------------------------------
    X_train, y_train, X_test, y_test, total = load_features()
    print(f"[INFO] Total samples in features.csv : {total}")
    print(f"[INFO] Train : {len(X_train)} | Test : {len(X_test)}")
    print(f"[INFO] Features ({len(COLOR_COLS)}) : {COLOR_COLS}")

    print("\n[INFO] Train class distribution:")
    for i, s in enumerate(SEASONS):
        print(f"  {s:<8}: {(y_train == i).sum()}")

    # --- GridSearchCV --------------------------------------------------------
    print(f"\n[INFO] Running {CV_FOLDS}-fold GridSearchCV ...")
    print(f"[INFO] Search space : {PARAM_GRID}")

    t_start = time.time()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    grid = GridSearchCV(
        SVC(kernel="rbf", class_weight="balanced",
            probability=True, random_state=42),
        PARAM_GRID,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)
    train_time = time.time() - t_start

    print(f"\n[INFO] Grid search complete in {train_time:.1f}s")
    print(f"[INFO] Best params  : {grid.best_params_}")
    print(f"[INFO] Best CV F1   : {grid.best_score_:.4f}")

    best_svm = grid.best_estimator_
    joblib.dump(best_svm, MODEL_OUT)
    print(f"[INFO] Model saved  → {MODEL_OUT}")

    # --- Test evaluation -----------------------------------------------------
    y_pred  = best_svm.predict(X_test)
    y_proba = best_svm.predict_proba(X_test)

    accuracy  = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall    = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1        = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    top2      = float(np.mean([
        y_test[i] in np.argsort(y_proba[i])[-2:]
        for i in range(len(y_test))
    ]))

    cm = confusion_matrix(y_test, y_pred)
    autumn_recall = float(cm[0, 0] / cm[0].sum()) if cm[0].sum() > 0 else 0.0

    print("\n" + "=" * 60)
    print("  Test Set Results")
    print("=" * 60)
    print(f"  Accuracy      : {accuracy:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  F1 Score      : {f1:.4f}")
    print(f"  Top-2 Accuracy: {top2:.4f}")
    print(f"  Autumn Recall : {autumn_recall:.4f}")

    report_str = classification_report(y_test, y_pred, target_names=SEASONS, digits=4)
    print("\n" + report_str)

    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(f"{'':12s} " + "  ".join(f"{s:>8s}" for s in SEASONS))
    for i, row in enumerate(cm):
        print(f"{SEASONS[i]:12s} " + "  ".join(f"{v:>8d}" for v in row))

    print(f"\n[NOTE] Autumn recall: {autumn_recall:.4f}")
    print("       Autumn is historically the hardest class — track separately.")

    # --- Save results JSON (same schema as evaluate.py comparison table) -----
    results = {
        "model":          "SVM (RBF)",
        "best_params":    grid.best_params_,
        "best_cv_f1":     round(grid.best_score_, 6),
        "accuracy":       round(accuracy, 6),
        "precision":      round(precision, 6),
        "recall":         round(recall, 6),
        "f1":             round(f1, 6),
        "top2_accuracy":  round(top2, 6),
        "autumn_recall":  round(autumn_recall, 6),
        "train_time_sec": round(train_time, 2),
        "features":       COLOR_COLS,
        "n_train":        int(len(X_train)),
        "n_test":         int(len(X_test)),
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results JSON saved → {RESULTS_OUT}")

    # --- Text report ---------------------------------------------------------
    with open(REPORT_OUT, "w") as f:
        f.write("ToneFit — SVM Color Feature Classifier — Test Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Kernel      : RBF\n")
        f.write(f"Best params : {grid.best_params_}\n")
        f.write(f"Features    : {COLOR_COLS}\n")
        f.write(f"Train time  : {train_time:.1f}s\n")
        f.write("=" * 60 + "\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
        f.write("\nConfusion Matrix (rows=actual, cols=predicted):\n")
        f.write(f"{'':12s} " + "  ".join(f"{s:>8s}" for s in SEASONS) + "\n")
        for i, row in enumerate(cm):
            f.write(f"{SEASONS[i]:12s} " + "  ".join(f"{v:>8d}" for v in row) + "\n")
        f.write(f"\nAutumn recall  : {autumn_recall:.4f}\n")
        f.write(f"Top-2 accuracy : {top2:.4f}\n")
        f.write(f"Training time  : {train_time:.1f}s\n")
    print(f"[INFO] Report saved → {REPORT_OUT}")

    # --- Confusion matrix plot -----------------------------------------------
    plot_confusion_matrix(cm, CONFUSION_OUT)

    print("\n[DONE] SVM training complete.")
    print(f"       Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Top-2: {top2:.4f}")


if __name__ == "__main__":
    main()
