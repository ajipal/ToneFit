"""
ToneFit ML — Traditional ML Training (Step 4)
===============================================
Trains two classical machine learning classifiers on the extracted color features.

Models trained:
  A1. SVM  — Support Vector Machine (RBF kernel, tuned with GridSearchCV)
  A2. RF   — Random Forest (100 estimators, shows feature importance)

What this script does:
  1. Loads X_train.npy and y_train.npy (from preprocess.py)
  2. Runs 5-fold stratified cross-validation on both models
  3. Tunes SVM hyperparameters (C, gamma) via GridSearchCV
  4. Trains final SVM and Random Forest on the full training set
  5. Plots and saves Random Forest feature importances
  6. Saves: models/svm_model.pkl, models/rf_model.pkl

Requirements:
  pip install scikit-learn numpy matplotlib seaborn

Usage:
  python train_traditional.py

Run preprocess.py first to generate X_train.npy and y_train.npy.
"""

import os
import pickle
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import classification_report, confusion_matrix

# ─── CONFIG ───────────────────────────────────────────────────────────────────

MODELS_DIR  = "models"
RESULTS_DIR = "results"
SEASONS     = ["spring", "summer", "autumn", "winter"]
LABEL_NAMES = SEASONS   # index 0=spring, 1=summer, 2=autumn, 3=winter

FEATURE_COLS = ["L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std",
                "ITA", "H_mean", "S_mean", "V_mean"]

RANDOM_STATE = 42
CV_FOLDS     = 5       # stratified k-fold cross-validation folds

# SVM grid search space
SVM_PARAM_GRID = {
    "C":     [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.001],
}

# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── SETUP ────────────────────────────────────────────────────────────────────

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_data():
    """Load training arrays from preprocess.py output."""
    required = ["X_train.npy", "y_train.npy"]
    for f in required:
        if not os.path.exists(f):
            raise FileNotFoundError(
                f"'{f}' not found. Run preprocess.py first."
            )
    X_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    log.info(f"  X_train: {X_train.shape}  —  y_train: {y_train.shape}")
    return X_train, y_train


def print_cv_results(name, cv_results):
    """Print a clean cross-validation summary table."""
    acc_scores = cv_results["test_accuracy"]
    f1_scores  = cv_results["test_f1_macro"]

    log.info(f"\n  {name} — {CV_FOLDS}-Fold Stratified Cross-Validation:")
    log.info(f"  {'Fold':<8} {'Accuracy':>10} {'F1 (macro)':>12}")
    log.info(f"  {'─'*32}")
    for i, (acc, f1) in enumerate(zip(acc_scores, f1_scores), 1):
        log.info(f"  Fold {i:<4} {acc:>10.4f} {f1:>12.4f}")
    log.info(f"  {'─'*32}")
    log.info(f"  {'Mean':<8} {acc_scores.mean():>10.4f} {f1_scores.mean():>12.4f}")
    log.info(f"  {'Std':<8} {acc_scores.std():>10.4f} {f1_scores.std():>12.4f}")


def save_confusion_matrix(y_true, y_pred, model_name):
    """Save a labeled confusion matrix heatmap to results/."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[s.capitalize() for s in SEASONS],
        yticklabels=[s.capitalize() for s in SEASONS],
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name} (Training CV)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}_train.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


def save_feature_importance(importances, model_name):
    """Bar chart of Random Forest feature importances."""
    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLS[i] for i in sorted_idx]
    sorted_values   = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0" if v >= sorted_values[2] else "#A8C4E0" for v in sorted_values]
    bars = ax.bar(sorted_features, sorted_values, color=colors, edgecolor="black", linewidth=0.6)

    for bar, val in zip(bars, sorted_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=8.5
        )

    ax.set_title(f"Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature", fontsize=11)
    ax.set_ylabel("Importance", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_importance_rf.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")


# ─── MODEL A1: SVM ─────────────────────────────────────────────────────────────

def train_svm(X_train, y_train):
    """
    Train an SVM with RBF kernel.

    Steps:
      1. GridSearchCV to find best C and gamma
         (5-fold CV inside the search — so no data leakage)
      2. Cross-validate the best model to report honest CV scores
      3. Refit on full training set with best hyperparameters
      4. Save to models/svm_model.pkl
    """
    log.info("\n" + "─" * 60)
    log.info("  MODEL A1: Support Vector Machine (RBF Kernel)")
    log.info("─" * 60)

    # ── Hyperparameter tuning ──────────────────────────────────────────────
    log.info(f"  Running GridSearchCV over {SVM_PARAM_GRID} ...")
    log.info("  (This may take a minute — searching 16 combinations × 5 folds)")

    svm_base = SVC(
        kernel="rbf",
        class_weight="balanced",   # handles class imbalance
        probability=True,          # needed for confidence scores in predict.py
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        svm_base,
        SVM_PARAM_GRID,
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_cv_acc = grid_search.best_score_
    log.info(f"\n  Best hyperparameters : C={best_params['C']}, gamma={best_params['gamma']}")
    log.info(f"  Best CV accuracy     : {best_cv_acc:.4f}")

    # ── Cross-validation with best params ─────────────────────────────────
    best_svm = SVC(
        kernel="rbf",
        C=best_params["C"],
        gamma=best_params["gamma"],
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        best_svm, X_train, y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        return_train_score=False,
    )
    print_cv_results("SVM", cv_results)

    # ── Final training on full training set ───────────────────────────────
    log.info("\n  Training final SVM on full training set...")
    best_svm.fit(X_train, y_train)

    # Training set performance (optimistic — just for reference)
    y_pred_train = best_svm.predict(X_train)
    log.info("\n  Training Set Classification Report:")
    report = classification_report(y_train, y_pred_train, target_names=LABEL_NAMES)
    for line in report.split("\n"):
        log.info("    " + line)

    save_confusion_matrix(y_train, y_pred_train, "SVM")

    # ── Save model ─────────────────────────────────────────────────────────
    svm_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    with open(svm_path, "wb") as f:
        pickle.dump(best_svm, f)
    log.info(f"\n  Model saved: {svm_path}")

    return best_svm, cv_results


# ─── MODEL A2: RANDOM FOREST ──────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    """
    Train a Random Forest classifier.

    Steps:
      1. Cross-validate with 100 estimators
      2. Train final model on full training set
      3. Plot and save feature importances
      4. Save to models/rf_model.pkl
    """
    log.info("\n" + "─" * 60)
    log.info("  MODEL A2: Random Forest (100 Estimators)")
    log.info("─" * 60)

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",   # handles class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # ── Cross-validation ───────────────────────────────────────────────────
    log.info(f"  Running {CV_FOLDS}-fold stratified cross-validation...")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        rf, X_train, y_train,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        return_train_score=False,
    )
    print_cv_results("Random Forest", cv_results)

    # ── Final training on full training set ───────────────────────────────
    log.info("\n  Training final Random Forest on full training set...")
    rf.fit(X_train, y_train)

    # Training set performance (optimistic — just for reference)
    y_pred_train = rf.predict(X_train)
    log.info("\n  Training Set Classification Report:")
    report = classification_report(y_train, y_pred_train, target_names=LABEL_NAMES)
    for line in report.split("\n"):
        log.info("    " + line)

    save_confusion_matrix(y_train, y_pred_train, "Random Forest")

    # ── Feature importances ────────────────────────────────────────────────
    importances = rf.feature_importances_
    log.info("\n  Feature Importances (higher = more useful for classification):")
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        log.info(f"    {rank}. {FEATURE_COLS[idx]:<10} : {importances[idx]:.4f}")

    save_feature_importance(importances, "Random Forest")

    # ── Save model ─────────────────────────────────────────────────────────
    rf_path = os.path.join(MODELS_DIR, "rf_model.pkl")
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    log.info(f"\n  Model saved: {rf_path}")

    return rf, cv_results


# ─── COMPARISON SUMMARY ───────────────────────────────────────────────────────

def print_comparison(svm_cv, rf_cv):
    """Side-by-side CV comparison of both models."""
    log.info("\n" + "=" * 60)
    log.info("  TRAINING SUMMARY — Cross-Validation Comparison")
    log.info("=" * 60)
    log.info(f"  {'Model':<20} {'CV Accuracy':>12} {'CV F1 (macro)':>14}")
    log.info(f"  {'─' * 50}")

    for name, cv in [("SVM (RBF)", svm_cv), ("Random Forest", rf_cv)]:
        acc = cv["test_accuracy"]
        f1  = cv["test_f1_macro"]
        log.info(
            f"  {name:<20} {acc.mean():.4f} ± {acc.std():.4f}"
            f"  {f1.mean():.4f} ± {f1.std():.4f}"
        )

    log.info(f"\n  Note: These are TRAINING cross-validation scores.")
    log.info(f"  Final evaluation on the held-out test set → run evaluate.py")
    log.info("=" * 60)
    log.info("\n  Saved files:")
    log.info(f"    {MODELS_DIR}/svm_model.pkl")
    log.info(f"    {MODELS_DIR}/rf_model.pkl")
    log.info(f"    {RESULTS_DIR}/confusion_matrix_svm_train.png")
    log.info(f"    {RESULTS_DIR}/confusion_matrix_random_forest_train.png")
    log.info(f"    {RESULTS_DIR}/feature_importance_rf.png")
    log.info(f"\n  Next step → python train_deeplearning.py")
    log.info("=" * 60)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run():
    log.info("=" * 60)
    log.info("  ToneFit ML — Traditional ML Training")
    log.info("=" * 60)

    # Load data
    log.info("\n[Step 1] Loading training data...")
    X_train, y_train = load_data()

    log.info("\n  Class distribution in training set:")
    for i, season in enumerate(SEASONS):
        count = int((y_train == i).sum())
        log.info(f"    {season:<8}: {count} samples")

    # Train models
    svm_model, svm_cv = train_svm(X_train, y_train)
    rf_model,  rf_cv  = train_random_forest(X_train, y_train)

    # Compare
    print_comparison(svm_cv, rf_cv)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
