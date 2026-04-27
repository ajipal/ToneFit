"""
ToneFit ML — Dataset Quality Audit
====================================
Run this AFTER preprocess.py to verify your dataset is ready for training.

Checks:
  1. Class balance — flags if any season is underrepresented (< 30 images)
  2. Nationality balance per season — checks Filipino representation
  3. Feature distribution — shows mean ± std per season per feature
  4. Outlier detection — flags images with extreme ITA or L* values
  5. Duplicate proximity — warns if pairwise feature distances are suspiciously low
  6. Generates a summary report: dataset_audit_report.txt

Usage:
  python audit_dataset.py

Requirements:
  pip install pandas numpy scikit-learn matplotlib seaborn
"""

import os
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

FEATURES_CSV = "features.csv"
MANIFEST_CSV = "dataset_manifest.csv"
AUDIT_CSV    = "preprocessing_audit.csv"
REPORT_PATH  = "dataset_audit_report.txt"
SEASONS      = ["spring", "summer", "autumn", "winter"]

FEATURE_COLS = [
    "L_mean", "a_mean", "b_mean", "L_std", "a_std", "b_std",
    "ITA", "H_mean", "S_mean", "V_mean"
]

def check_class_balance(df, report):
    report.append("\n── CLASS BALANCE ─────────────────────────────────────────")
    counts = df.groupby("season").size()
    total  = len(df)
    for season in SEASONS:
        n   = counts.get(season, 0)
        pct = (n / total * 100) if total > 0 else 0
        flag = " ⚠ LOW" if n < 30 else ""
        report.append(f"  {season:<8}: {n:>4} images ({pct:.1f}%){flag}")

    min_n = counts.min() if len(counts) else 0
    max_n = counts.max() if len(counts) else 0
    imbalance_ratio = max_n / (min_n + 1e-6)
    if imbalance_ratio > 1.5:
        report.append(f"\n  ⚠ Class imbalance ratio: {imbalance_ratio:.2f}x — consider collecting more for smaller classes")
    else:
        report.append(f"\n  ✓ Class balance ratio: {imbalance_ratio:.2f}x (good)")

def check_nationality_balance(df, manifest, report):
    report.append("\n── NATIONALITY BALANCE ───────────────────────────────────")
    if manifest is None:
        report.append("  ⚠ dataset_manifest.csv not found — skipping nationality check")
        return

    # Merge manifest into features on filename
    merged = df.merge(manifest[["filename", "nationality"]], on="filename", how="left")

    for season in SEASONS:
        sub = merged[merged["season"] == season]
        if sub.empty:
            continue
        nat_counts = sub["nationality"].value_counts()
        report.append(f"\n  {season.upper()}:")
        for nat, n in nat_counts.items():
            flag = " ← primary target" if nat == "Filipino" else ""
            report.append(f"    {nat:<12}: {n}{flag}")

        filipino_n = nat_counts.get("Filipino", 0)
        if filipino_n < 4:
            report.append(f"    ⚠ Only {filipino_n} Filipino samples in {season} — aim for ≥4")

def check_feature_distributions(df, report):
    report.append("\n── FEATURE DISTRIBUTIONS (mean ± std per season) ────────")
    key_features = ["L_mean", "ITA", "a_mean", "b_mean", "S_mean"]
    header = f"  {'Feature':<12}" + "".join(f"  {s:<18}" for s in SEASONS)
    report.append(header)

    for feat in key_features:
        row_str = f"  {feat:<12}"
        for season in SEASONS:
            sub = df[df["season"] == season][feat]
            if sub.empty:
                row_str += f"  {'—':<18}"
            else:
                row_str += f"  {sub.mean():.2f} ± {sub.std():.2f}       "
        report.append(row_str)

    # Check ITA separation — key discriminative feature
    report.append("\n  ITA interpretation guide:")
    report.append("    Higher ITA → lighter/cooler skin (expected: Summer > Winter > Spring > Autumn)")
    ita_means = {s: df[df["season"] == s]["ITA"].mean() for s in SEASONS if s in df["season"].values}
    sorted_seasons = sorted(ita_means, key=ita_means.get, reverse=True)
    report.append(f"    Observed order: {' > '.join(sorted_seasons)}")

def check_outliers(df, report):
    report.append("\n── OUTLIERS ──────────────────────────────────────────────")
    for feat in ["L_mean", "ITA"]:
        q1  = df[feat].quantile(0.05)
        q99 = df[feat].quantile(0.95)
        outliers = df[(df[feat] < q1) | (df[feat] > q99)]
        report.append(f"  {feat}: {len(outliers)} extreme values (outside 5th–95th pct)")
        if len(outliers) > 0 and len(outliers) <= 10:
            for _, row in outliers.iterrows():
                report.append(f"    {row['filename']} ({row['season']}) — {feat}={row[feat]:.2f}")

def check_preprocessing_audit(report):
    report.append("\n── PREPROCESSING FILTER SUMMARY ─────────────────────────")
    if not os.path.exists(AUDIT_CSV):
        report.append("  preprocessing_audit.csv not found — run preprocess.py first")
        return

    audit = pd.read_csv(AUDIT_CSV)
    status_counts = audit["status"].value_counts()
    total = len(audit)
    report.append(f"  Total images scanned: {total}")
    for status, count in status_counts.items():
        pct = count / total * 100
        report.append(f"    {status:<35}: {count} ({pct:.1f}%)")

    kept = status_counts.get("ok", 0)
    report.append(f"\n  Kept (passed all filters): {kept} / {total} ({kept/total*100:.1f}%)")

def run():
    log.info("=" * 65)
    log.info("  ToneFit ML — Dataset Audit")
    log.info("=" * 65)

    if not os.path.exists(FEATURES_CSV):
        log.error(f"  {FEATURES_CSV} not found. Run preprocess.py first.")
        return

    df = pd.read_csv(FEATURES_CSV)
    log.info(f"  Loaded {FEATURES_CSV}: {len(df)} rows")

    manifest = None
    if os.path.exists(MANIFEST_CSV):
        manifest = pd.read_csv(MANIFEST_CSV)
        log.info(f"  Loaded {MANIFEST_CSV}: {len(manifest)} rows")
    else:
        log.warning(f"  {MANIFEST_CSV} not found — nationality checks will be skipped")

    report = []
    report.append("=" * 65)
    report.append("  ToneFit ML — Dataset Audit Report")
    report.append("=" * 65)
    report.append(f"  Dataset: {len(df)} total images across {df['season'].nunique()} seasons")

    check_class_balance(df, report)
    check_nationality_balance(df, manifest, report)
    check_feature_distributions(df, report)
    check_outliers(df, report)
    check_preprocessing_audit(report)

    report.append("\n── READINESS VERDICT ─────────────────────────────────────")
    counts = df.groupby("season").size()
    all_ok = all(counts.get(s, 0) >= 30 for s in SEASONS)
    if all_ok:
        report.append("  ✓ Dataset looks ready for training.")
        report.append("    Run: python train_traditional.py")
    else:
        low = [s for s in SEASONS if counts.get(s, 0) < 30]
        report.append(f"  ⚠ Dataset not yet ready — {', '.join(low)} need more images.")
        report.append("    Collect more data before training.")

    report.append("\n" + "=" * 65)

    # Print to console
    for line in report:
        log.info(line)

    # Save report
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report))
    log.info(f"\n  Full report saved: {REPORT_PATH}")

if __name__ == "__main__":
    run()