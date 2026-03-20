#!/usr/bin/env python3
"""
wesad_f1_benchmark.py — Compare F1: all windows vs S2S certified only

Loads wesad_certified/*.json, trains a stress classifier,
compares F1 on certified-only vs all windows.

Usage:
    python3.9 wesad_f1_benchmark.py
    python3.9 wesad_f1_benchmark.py --cert-dir wesad_certified/
"""
import os, sys, json, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/S2S"))

try:
    import numpy as np
except ImportError:
    print("pip3.9 install numpy"); sys.exit(1)

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    print("pip3.9 install scikit-learn"); sys.exit(1)

TIER_SCORE = {"GOLD": 4, "SILVER": 3, "BRONZE": 2, "REJECTED": 0, None: 0}
CONDITION_MAP = {"baseline": 0, "stress": 1, "amusement": 2, "meditation": 3}


def load_all_certs(cert_dir):
    """Load all subject JSON files, return list of cert dicts."""
    certs = []
    for f in sorted(Path(cert_dir).glob("S*_certified.json")):
        data = json.loads(f.read_text())
        certs.extend(data)
    return certs


def extract_features(cert):
    """
    Extract feature vector from a fusion cert dict.
    Returns 8-dim vector.
    """
    hil        = cert.get("human_in_loop_score", 0) or 0
    n_streams  = cert.get("n_valid_streams", 0) or 0
    chest_t    = TIER_SCORE.get(cert.get("chest_tier"), 0)
    wrist_t    = TIER_SCORE.get(cert.get("wrist_tier"), 0)
    ppg_t      = TIER_SCORE.get(cert.get("ppg_tier"), 0)
    hr         = cert.get("ppg_hr_bpm") or 0
    n_flags    = len(cert.get("flags", []))
    fusion_t   = TIER_SCORE.get(cert.get("tier"), 0)

    return [hil, n_streams, chest_t, wrist_t, ppg_t,
            hr, n_flags, fusion_t]


def run_cv(X, y, label=""):
    """5-fold stratified CV, returns mean F1 macro."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = LogisticRegression(max_iter=1000, random_state=42,
                                 class_weight="balanced")
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        f1s.append(f1_score(y_te, pred, average="macro", zero_division=0))

    mean_f1 = float(np.mean(f1s))
    std_f1  = float(np.std(f1s))
    print(f"  {label:<30} F1={mean_f1:.3f} ±{std_f1:.3f}  (n={len(X)})")
    return mean_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert-dir", default="wesad_certified")
    args = parser.parse_args()

    print("\nWESAD F1 Benchmark — Certified vs All Windows")
    print("=" * 55)

    certs = load_all_certs(args.cert_dir)
    print(f"Loaded {len(certs)} total windows")

    # Filter to windows with valid condition labels
    valid = [c for c in certs
             if c.get("condition") in CONDITION_MAP]
    print(f"Valid labeled windows: {len(valid)}")

    if len(valid) < 50:
        print("Not enough labeled windows. Run wesad_adapter.py first.")
        sys.exit(1)

    # Build features + labels for ALL windows
    X_all = np.array([extract_features(c) for c in valid], dtype=np.float32)
    y_all = np.array([CONDITION_MAP[c["condition"]] for c in valid])

    # Build features + labels for CERTIFIED windows only
    certified = [c for c in valid if c.get("tier") != "REJECTED"]
    print(f"Certified windows:     {len(certified)} "
          f"({100*len(certified)//len(valid)}%)")

    # Condition breakdown
    print("\nCondition distribution:")
    for cond, idx in sorted(CONDITION_MAP.items()):
        n_all  = sum(1 for c in valid if c["condition"] == cond)
        n_cert = sum(1 for c in certified if c["condition"] == cond)
        print(f"  {cond:<14} all={n_all:>5}  certified={n_cert:>5}")

    if len(certified) < 20:
        print("\nToo few certified windows for F1 comparison.")
        sys.exit(1)

    X_cert = np.array([extract_features(c) for c in certified], dtype=np.float32)
    y_cert = np.array([CONDITION_MAP[c["condition"]] for c in certified])

    print("\n5-fold cross-validation results:")
    f1_all  = run_cv(X_all,  y_all,  label="All windows (uncertified)")
    f1_cert = run_cv(X_cert, y_cert, label="Certified only (S2S)")

    delta = f1_cert - f1_all
    pct   = 100 * delta / max(f1_all, 1e-8)

    print(f"\n{'='*55}")
    print(f"F1 improvement: {delta:+.3f} ({pct:+.1f}%)")
    print(f"  Baseline (all):   {f1_all:.3f}")
    print(f"  S2S certified:    {f1_cert:.3f}")
    print(f"{'='*55}")

    # Per-condition breakdown
    print("\nPer-condition F1 (certified set, 1 split):")
    from sklearn.model_selection import train_test_split
    if len(X_cert) >= 20:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_cert, y_cert, test_size=0.3, random_state=42,
            stratify=y_cert if len(np.unique(y_cert)) > 1 else None)
        scaler = StandardScaler()
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(scaler.fit_transform(X_tr), y_tr)
        pred = clf.predict(scaler.transform(X_te))
        for cond, idx in sorted(CONDITION_MAP.items()):
            mask = y_te == idx
            if mask.sum() > 0:
                f1 = f1_score(y_te[mask], pred[mask],
                              average="macro", zero_division=0,
                              labels=[idx])
                print(f"  {cond:<14} F1={f1:.3f}  (n={mask.sum()})")

    # Save results
    results = {
        "dataset": "WESAD",
        "total_windows": len(valid),
        "certified_windows": len(certified),
        "certification_rate": round(len(certified)/len(valid), 3),
        "f1_all_windows": round(f1_all, 4),
        "f1_certified_only": round(f1_cert, 4),
        "f1_delta": round(delta, 4),
        "f1_improvement_pct": round(pct, 2),
    }
    Path("experiments/results_wesad_f1.json").write_text(
        json.dumps(results, indent=2))
    print(f"\nSaved: experiments/results_wesad_f1.json")


if __name__ == "__main__":
    main()
