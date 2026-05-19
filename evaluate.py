"""
<<<<<<< HEAD
evaluate.py — ToneFit ML Project
==================================
Compares all trained models side-by-side and produces a summary table.

Models compared:
    Model A — FaRL-64 Baseline      (results/farl_history.json + farl_test_report.txt)
    Model B — FaRL-64 12-class      (results/farl_12class_history.json)
    Model C — FaRL-64 Improved      (results/farl_improved_history.json)   [optional]

Paper baselines (Stacchio et al. 2024) are included for reference.
=======
ToneFit ML — Model Evaluation (Step 6)
========================================
Evaluates FaRL, DINOv2, and MCF on the held-out test set and compares them
against each other and against the Deep Armocromia paper baselines.
>>>>>>> 1a31814da67fa0778a8ad4f409ba39357db76389

Usage:
    python evaluate.py

<<<<<<< HEAD
Outputs:
    results/evaluation_summary.csv   — machine-readable comparison table
    results/evaluation_summary.txt   — human-readable comparison table
=======
Requires:
    - models/farl_model.pth    (from train_farl.py)
    - models/dinov2_model.pth  (from train_dinov2.py)
    - models/mcf_model.pth     (from train_mcf.py)
    - RGB-M/test/              (READ ONLY — loaded via ImageFolder)
>>>>>>> 1a31814da67fa0778a8ad4f409ba39357db76389
"""

import json
import os
import re

import numpy as np

# ---------------------------------------------------------------------------
# PAPER BASELINES (Stacchio et al., ECCV 2024)
# ---------------------------------------------------------------------------

PAPER_BASELINES = [
    {
        "Model":       "FaRL-16 (paper)",
        "Source":      "Stacchio et al. 2024",
        "Season Acc":  0.525,
        "F1 (macro)":  0.516,
        "SubType Acc": 0.318,
        "Top-3 Acc":   0.663,
    },
    {
        "Model":       "FaRL-64 (paper)",
        "Source":      "Stacchio et al. 2024",
        "Season Acc":  0.554,
        "F1 (macro)":  0.548,
        "SubType Acc": 0.313,
        "Top-3 Acc":   0.651,
    },
    {
        "Model":       "ResNeXt50 (paper)",
        "Source":      "Stacchio et al. 2024",
        "Season Acc":  0.513,
        "F1 (macro)":  0.502,
        "SubType Acc": 0.281,
        "Top-3 Acc":   0.614,
    },
]

<<<<<<< HEAD
# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
=======
FARL_PATH     = os.path.join(MODELS_DIR, "farl_model.pth")
DINOV2_PATH   = os.path.join(MODELS_DIR, "dinov2_model.pth")
MCF_PATH      = os.path.join(MODELS_DIR, "mcf_model.pth")
>>>>>>> 1a31814da67fa0778a8ad4f409ba39357db76389

NA = "N/A"


def _fmt(val, decimals=4):
    """Format a numeric value or return 'N/A'."""
    if val is None or val == NA:
        return NA
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return NA


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _extract_macro_f1_from_report(report_path):
    """Parse macro avg F1 from a sklearn classification_report text file."""
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        text = f.read()
    m = re.search(r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", text)
    return float(m.group(1)) if m else None


<<<<<<< HEAD
def _extract_season_f1_from_12class_report(report_path):
    """Parse macro F1 from the derived season section of the 12-class report."""
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        text = f.read()
    # Look for macro avg in the derived season section (appears after "Derived 4-Season")
    parts = text.split("Derived 4-Season")
    if len(parts) < 2:
        return None
    m = re.search(r"macro avg\s+[\d.]+\s+[\d.]+\s+([\d.]+)", parts[1])
    return float(m.group(1)) if m else None
=======

def load_mcf(path):
    """Load MCF (Mask Contrastive Face) ViT-B/16 model."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for MCF. Install with: pip install timm>=0.9.0")
    backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    feature_dim = 768
    head = build_classifier_head(feature_dim)

    checkpoint = torch.load(path, map_location=DEVICE)
    if "backbone" in checkpoint and "head" in checkpoint:
        backbone.load_state_dict(checkpoint["backbone"], strict=False)
        head.load_state_dict(checkpoint["head"])
    else:
        try:
            backbone.load_state_dict(checkpoint, strict=False)
        except Exception:
            pass

    class _Model(nn.Module):
        def __init__(self, bb, h):
            super().__init__()
            self.backbone = bb
            self.head = h
        def forward(self, x):
            with torch.no_grad():
                feats = self.backbone(x)
            return self.head(feats)

    model = _Model(backbone, head).to(DEVICE)
    model.eval()
    return model

# ── INFERENCE ──────────────────────────────────────────────────────────────────
>>>>>>> 1a31814da67fa0778a8ad4f409ba39357db76389


# ---------------------------------------------------------------------------
# MODEL RESULT LOADERS
# ---------------------------------------------------------------------------

def load_model_a():
    """FaRL-64 Baseline — reads farl_history.json + farl_test_report.txt."""
    h = _load_json("results/farl_history.json")
    if h is None:
        print("[WARN] results/farl_history.json not found — Model A skipped.")
        return None

    macro_f1     = _extract_macro_f1_from_report("results/farl_test_report.txt")
    best_val_acc = h.get("best_val_acc")

    # Try to also read sub-type acc if the new flat-baseline format saved it
    subtype_acc = h.get("best_val_subtype_acc", NA)

    return {
        "Model":       "Model A — FaRL-64 Baseline (ours)",
        "Source":      "This study",
        "Season Acc":  best_val_acc,
        "F1 (macro)":  macro_f1,
        "SubType Acc": subtype_acc,
        "Top-3 Acc":   NA,
    }


def load_model_b():
    """FaRL-64 12-class — reads farl_12class_history.json."""
    h = _load_json("results/farl_12class_history.json")
    if h is None:
        print("[WARN] results/farl_12class_history.json not found — Model B skipped.")
        return None

    # Season F1 from the derived season section of the report
    season_f1 = _extract_season_f1_from_12class_report("results/farl_12class_report.txt")

    return {
        "Model":       "Model B — FaRL-64 12-class (ours)",
        "Source":      "This study",
        "Season Acc":  h.get("best_val_season_acc"),
        "F1 (macro)":  season_f1,
        "SubType Acc": h.get("best_val_acc"),        # 12-class acc IS the subtype acc
        "Top-3 Acc":   h.get("best_val_top3"),
    }


<<<<<<< HEAD
def load_model_c():
    """FaRL-64 Improved (LLRD + unfreeze) — reads farl_improved_history.json."""
    h = _load_json("results/farl_improved_history.json")
    if h is None:
        return None   # Not trained yet — silently skip

    macro_f1    = _extract_macro_f1_from_report("results/farl_improved_report.txt")
    subtype_acc = h.get("best_val_subtype_acc", NA)

    return {
        "Model":       "Model C — FaRL-64 Improved (ours)",
        "Source":      "This study",
        "Season Acc":  h.get("best_val_acc"),
        "F1 (macro)":  macro_f1,
        "SubType Acc": subtype_acc,
        "Top-3 Acc":   NA,
    }
=======
def save_combined_confusion_matrices(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[s.capitalize() for s in SEASONS],
            yticklabels=[s.capitalize() for s in SEASONS],
            linewidths=0.5, ax=ax,
        )
        ax.set_title(f"{r['name']}\nAcc: {r['metrics']['Accuracy']:.4f}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
    fig.suptitle("Confusion Matrices — FaRL vs DINOv2 vs MCF (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {path}")
>>>>>>> 1a31814da67fa0778a8ad4f409ba39357db76389


def load_hierarchical():
    """Hierarchical FaRL-64 — reads hierarchical_history.json + hierarchical_test_report.txt."""
    h = _load_json("results/hierarchical_history.json")
    if h is None:
        return None

    best = max(h, key=lambda r: r.get("val_season_acc", 0))
    macro_f1 = _extract_macro_f1_from_report("results/hierarchical_test_report.txt")

<<<<<<< HEAD
    return {
        "Model":       "Hierarchical FaRL-64 (ours)",
        "Source":      "This study",
        "Season Acc":  best.get("val_season_acc"),
        "F1 (macro)":  macro_f1,
        "SubType Acc": best.get("val_subtype_acc"),
        "Top-3 Acc":   NA,
    }
=======
    # Color: gray for paper baselines, colored for ours
    n_base  = len(baselines)
    colors  = ["#AAAAAA"] * n_base + ["#4C72B0", "#DD8452", "#55A868"][:len(df_our)]
>>>>>>> 1a31814da67fa0778a8ad4f409ba39357db76389


def load_dinov2():
    """DINOv2 fallback model — reads dinov2_history.json if present."""
    h = _load_json("results/dinov2_history.json")
    if h is None:
        return None

    macro_f1 = _extract_macro_f1_from_report("results/dinov2_report.txt")

<<<<<<< HEAD
    return {
        "Model":       "DINOv2 (ours)",
        "Source":      "This study",
        "Season Acc":  h.get("best_val_acc"),
        "F1 (macro)":  macro_f1,
        "SubType Acc": NA,
        "Top-3 Acc":   NA,
    }
=======
def run():
    log.info("=" * 65)
    log.info("  ToneFit ML — Model Evaluation")
    log.info("  FaRL vs DINOv2 vs MCF | Deep Armocromia dataset")
    log.info("=" * 65)
    log.info(f"  Device: {DEVICE}")
>>>>>>> 1a31814da67fa0778a8ad4f409ba39357db76389


# ---------------------------------------------------------------------------
# TABLE FORMATTING
# ---------------------------------------------------------------------------

COLUMNS = ["Model", "Source", "Season Acc", "F1 (macro)", "SubType Acc", "Top-3 Acc"]
COL_WIDTHS = [38, 22, 11, 11, 12, 10]


def _row_str(row: dict) -> str:
    cells = [
        str(row.get("Model",       NA)),
        str(row.get("Source",      NA)),
        _fmt(row.get("Season Acc")),
        _fmt(row.get("F1 (macro)")),
        _fmt(row.get("SubType Acc")),
        _fmt(row.get("Top-3 Acc")),
    ]
    return " | ".join(c.ljust(w) for c, w in zip(cells, COL_WIDTHS))


def _header_str() -> str:
    return " | ".join(c.ljust(w) for c, w in zip(COLUMNS, COL_WIDTHS))


def _separator() -> str:
    return "-+-".join("-" * w for w in COL_WIDTHS)


def print_table(rows: list[dict]):
    print("\n" + _header_str())
    print(_separator())
    for i, row in enumerate(rows):
        # Blank line to separate paper baselines from ours
        if i == len(PAPER_BASELINES):
            print(_separator())
        print(_row_str(row))
    print()


def save_csv(rows: list[dict], path: str):
    with open(path, "w") as f:
        f.write(",".join(COLUMNS) + "\n")
        for row in rows:
            cells = [
                row.get("Model",       NA),
                row.get("Source",      NA),
                _fmt(row.get("Season Acc")),
                _fmt(row.get("F1 (macro)")),
                _fmt(row.get("SubType Acc")),
                _fmt(row.get("Top-3 Acc")),
            ]
            f.write(",".join(str(c) for c in cells) + "\n")
    print(f"[INFO] CSV saved to: {path}")


def save_txt(rows: list[dict], path: str):
    lines = [
        "ToneFit ML — Model Comparison",
        "=" * (sum(COL_WIDTHS) + 3 * (len(COL_WIDTHS) - 1)),
        "Paper baselines: Stacchio et al., ECCV 2024",
        "",
        _header_str(),
        _separator(),
    ]
    for i, row in enumerate(rows):
        if i == len(PAPER_BASELINES):
            lines.append(_separator())
        lines.append(_row_str(row))

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[INFO] TXT saved to: {path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  ToneFit — Model Evaluation & Comparison")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)

    rows = list(PAPER_BASELINES)  # start with paper baselines

    # Load our trained models (skip gracefully if not yet trained)
    for loader in [load_model_a, load_model_b, load_model_c, load_hierarchical, load_dinov2]:
        result = loader()
        if result is not None:
            rows.append(result)

    if len(rows) == len(PAPER_BASELINES):
        print("\n[WARN] No trained model results found.")
        print("       Run training scripts first, then re-run evaluate.py.")
        return

    # Print table
    print("\n" + "=" * 60)
    print("  Comparison Table")
    print("=" * 60)
    print_table(rows)

    # Highlight best ours vs paper
    our_rows = [r for r in rows if r.get("Source") == "This study"]
    if our_rows:
        print("--- Highlights ---")
        paper_szn  = max(r["Season Acc"]  for r in PAPER_BASELINES)
        paper_sub  = max(r["SubType Acc"] for r in PAPER_BASELINES)
        paper_top3 = max(r["Top-3 Acc"]   for r in PAPER_BASELINES)

        for r in our_rows:
            szn = r.get("Season Acc")
            sub = r.get("SubType Acc")
            t3  = r.get("Top-3 Acc")

            if szn not in (None, NA):
                diff = float(szn) - paper_szn
                sign = "+" if diff >= 0 else ""
                print(f"  {r['Model']}: Season Acc {_fmt(szn)} ({sign}{diff:.4f} vs paper best {paper_szn:.3f})")

<<<<<<< HEAD
            if sub not in (None, NA):
                diff = float(sub) - paper_sub
                sign = "+" if diff >= 0 else ""
                print(f"  {r['Model']}: SubType Acc {_fmt(sub)} ({sign}{diff:.4f} vs paper best {paper_sub:.3f})")
=======
    # ── MCF ───────────────────────────────────────────────────────────────────
    log.info("\n" + "─" * 65)
    log.info("  MODEL C: MCF (Mask Contrastive Face, ViT-B/16)")
    log.info("─" * 65)
    if not os.path.exists(MCF_PATH):
        log.warning(f"  {MCF_PATH} not found — skipping MCF.")
    else:
        log.info(f"  Loading from {MCF_PATH} ...")
        mcf = load_mcf(MCF_PATH)
        y_true_m, y_pred_m, y_prob_m = run_inference(mcf, test_loader)
        metrics_m = compute_metrics("MCF (ours)", y_true_m, y_pred_m, y_prob_m)
        save_confusion_matrix(y_true_m, y_pred_m, "MCF (ours)", "confusion_mcf.png")
        model_results.append({
            "name": "MCF (ours)", "y_true": y_true_m,
            "y_pred": y_pred_m, "metrics": metrics_m,
        })

    if not model_results:
        log.error("  No models evaluated. Train both models first.")
        return
>>>>>>> 1a31814da67fa0778a8ad4f409ba39357db76389

            if t3 not in (None, NA):
                diff = float(t3) - paper_top3
                sign = "+" if diff >= 0 else ""
                print(f"  {r['Model']}: Top-3 Acc {_fmt(t3)} ({sign}{diff:.4f} vs paper best {paper_top3:.3f})")
        print()

    # Save outputs
    save_csv(rows, "results/evaluation_summary.csv")
    save_txt(rows, "results/evaluation_summary.txt")

<<<<<<< HEAD
    print("[DONE] Evaluation complete.")
=======
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    df_full.to_csv(csv_path, index=False)
    log.info(f"  Saved: {csv_path}")

    # ── Combined confusion matrices ───────────────────────────────────────────
    if len(model_results) >= 1:
        log.info("\n[Step 3] Saving confusion matrices...")
        save_combined_confusion_matrices(model_results)

    # ── Comparison chart vs paper ─────────────────────────────────────────────
    log.info("\n[Step 4] Saving comparison chart...")
    save_comparison_chart(df_our, PAPER_BASELINES)

    # ── Final printed table ───────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  FINAL RESULTS — Test Set")
    log.info("=" * 65)

    display_cols = ["Model", "Accuracy", "F1 (weighted)", "Top-2 Accuracy",
                    "Recall (Autumn)"]
    available = [c for c in display_cols if c in df_our.columns]
    log.info("\n" + df_our[available].to_string(index=False))

    log.info("\n  Paper baselines (Stacchio et al., 2024):")
    for name, vals in PAPER_BASELINES.items():
        log.info(f"    {name:<22}: acc={vals['accuracy']:.3f}  f1={vals['f1']:.3f}")

    log.info("\n  Autumn recall (expected hardest class):")
    for r in model_results:
        log.info(f"    {r['name']:<22}: {r['metrics']['Recall (Autumn)']:.4f}")

    best = max(model_results, key=lambda r: r["metrics"]["Accuracy"])
    log.info(f"\n  Best model: {best['name']}  "
             f"(acc={best['metrics']['Accuracy']:.4f})")

    log.info("\n" + "=" * 65)
    log.info("  Saved files:")
    log.info(f"    {RESULTS_DIR}/comparison_table.csv")
    log.info(f"    {RESULTS_DIR}/confusion_farl.png")
    log.info(f"    {RESULTS_DIR}/confusion_dinov2.png")
    log.info(f"    {RESULTS_DIR}/confusion_mcf.png")
    log.info(f"    {RESULTS_DIR}/confusion_matrices.png")
    log.info(f"    {RESULTS_DIR}/model_comparison.png")
    log.info("=" * 65)
>>>>>>> 1a31814da67fa0778a8ad4f409ba39357db76389


if __name__ == "__main__":
    main()
