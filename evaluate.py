"""
evaluate.py — ToneFit ML Project
==================================
Compares all trained models side-by-side and produces a summary table.

Models compared:
    Model A — FaRL-64 Baseline      (results/farl_history.json + farl_test_report.txt)
    Model B — FaRL-64 12-class      (results/farl_12class_history.json)
    Model C — FaRL-64 Improved      (results/farl_improved_history.json)   [optional]

Paper baselines (Stacchio et al. 2024) are included for reference.

Usage:
    python evaluate.py

Outputs:
    results/evaluation_summary.csv   — machine-readable comparison table
    results/evaluation_summary.txt   — human-readable comparison table
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

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

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


def load_dinov2():
    """DINOv2 fallback model — reads dinov2_history.json if present."""
    h = _load_json("results/dinov2_history.json")
    if h is None:
        return None

    macro_f1 = _extract_macro_f1_from_report("results/dinov2_report.txt")

    return {
        "Model":       "DINOv2 (ours)",
        "Source":      "This study",
        "Season Acc":  h.get("best_val_acc"),
        "F1 (macro)":  macro_f1,
        "SubType Acc": NA,
        "Top-3 Acc":   NA,
    }


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
    for loader in [load_model_a, load_model_b, load_model_c, load_dinov2]:
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

            if sub not in (None, NA):
                diff = float(sub) - paper_sub
                sign = "+" if diff >= 0 else ""
                print(f"  {r['Model']}: SubType Acc {_fmt(sub)} ({sign}{diff:.4f} vs paper best {paper_sub:.3f})")

            if t3 not in (None, NA):
                diff = float(t3) - paper_top3
                sign = "+" if diff >= 0 else ""
                print(f"  {r['Model']}: Top-3 Acc {_fmt(t3)} ({sign}{diff:.4f} vs paper best {paper_top3:.3f})")
        print()

    # Save outputs
    save_csv(rows, "results/evaluation_summary.csv")
    save_txt(rows, "results/evaluation_summary.txt")

    print("[DONE] Evaluation complete.")


if __name__ == "__main__":
    main()
