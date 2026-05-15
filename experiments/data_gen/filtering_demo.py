"""
filtering_demo.py — The REAL investor demo.

Proves: training on S2S-certified real data outperforms
training on unfiltered real data.

This is the already-validated claim from the benchmark.
This script makes it reproducible by anyone.

Usage: python3.9 experiments/data_gen/filtering_demo.py --pamap2 ~/S2S_data/pamap2
"""
import sys, glob, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold
except ImportError:
    print("pip install scikit-learn"); sys.exit(1)

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

engine = PhysicsEngine()

def certify_window(acc, gyro, ts):
    engine._last_terminal_state = None
    return engine.certify({"timestamps_ns": ts, "accel": acc, "gyro": gyro},
                          segment="forearm")

def raw_features(acc):
    a = np.array(acc)
    f = []
    for ax in range(3):
        col = a[:, ax]
        f += [col.mean(), col.std(), np.abs(col).max()]
    mag = np.sqrt((a**2).sum(axis=1))
    f += [mag.mean(), mag.std()]
    return np.array(f)

def load_pamap2(pamap2_dir, n_subjects=5, n_windows=40):
    files = sorted(glob.glob(str(Path(pamap2_dir) / "subject10*.dat")))[:n_subjects]
    if not files:
        print(f"No PAMAP2 files in {pamap2_dir}"); return [], [], []
    windows, labels, tiers = [], [], []
    for subj_idx, fpath in enumerate(files):
        try:
            arr = np.genfromtxt(fpath, invalid_raise=False)
            arr = arr[~np.isnan(arr).any(axis=1)]
            if arr.shape[1] < 20: continue
            # col 1=activity, col 4-6=hand acc, col 7-9=hand gyro
            act_col = arr[:, 1].astype(int)
            acc_cols = arr[:, 4:7]
            gyro_cols = arr[:, 7:10]
            # Get unique activities with enough samples
            for act in np.unique(act_col):
                idx = np.where(act_col == act)[0]
                if len(idx) < 300: continue
                for w in range(min(n_windows // len(np.unique(act_col)), 8)):
                    start = idx[w * 150]
                    if start + 256 > len(arr): continue
                    acc = acc_cols[start:start+256].tolist()
                    gyro = gyro_cols[start:start+256].tolist()
                    ts = [int(j * 1e9 / 100) for j in range(256)]
                    r = certify_window(acc, gyro, ts)
                    windows.append(raw_features(acc))
                    labels.append(int(act))
                    tiers.append(r["tier"])
        except Exception as e:
            print(f"  {Path(fpath).name}: {e}")
    return windows, labels, tiers

def run(pamap2_dir):
    print("\n" + "="*60)
    print("  S2S Filtering Demo — Real Data F1 Improvement")
    print("="*60)
    print("\nLoading PAMAP2...")
    windows, labels, tiers = load_pamap2(pamap2_dir)
    if not windows:
        return
    X = np.array(windows)
    y = np.array(labels)
    certified_mask = np.array([t in ("GOLD","SILVER") for t in tiers])
    print(f"  Total windows:     {len(windows)}")
    print(f"  Certified (≥SILVER): {certified_mask.sum()} ({certified_mask.mean()*100:.0f}%)")
    print(f"  Rejected:          {(~certified_mask).sum()}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_full, f1_cert = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_test = X[test_idx]; y_test = y[test_idx]
        # Full dataset
        scaler = StandardScaler()
        clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        X_tr_full = scaler.fit_transform(X[train_idx])
        clf.fit(X_tr_full, y[train_idx])
        f1_full.append(f1_score(y_test, clf.predict(scaler.transform(X_test)),
                                average="weighted", zero_division=0))
        # Certified only
        cert_train = train_idx[certified_mask[train_idx]]
        if len(cert_train) < 5: continue
        scaler2 = StandardScaler()
        clf2 = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        X_tr_cert = scaler2.fit_transform(X[cert_train])
        clf2.fit(X_tr_cert, y[cert_train])
        f1_cert.append(f1_score(y_test, clf2.predict(scaler2.transform(X_test)),
                                average="weighted", zero_division=0))

    f1_full_mean = np.mean(f1_full)
    f1_cert_mean = np.mean(f1_cert)
    improvement = f1_cert_mean - f1_full_mean

    print(f"\n  F1 — full unfiltered dataset:   {f1_full_mean:.3f}")
    print(f"  F1 — S2S certified only:        {f1_cert_mean:.3f}")
    print(f"  Improvement:                    {improvement:+.3f} ({improvement*100:+.1f}%)")
    print("-"*60)
    if improvement > 0:
        print("  ✓ S2S FILTERING IMPROVES F1 ON REAL DATA")
        print("  → Removing physics-invalid windows makes training data cleaner")
    else:
        print("  ✗ No improvement at this configuration")
    print("="*60)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pamap2", default="~/S2S_data/pamap2")
    args = p.parse_args()
    run(str(Path(args.pamap2).expanduser()))
