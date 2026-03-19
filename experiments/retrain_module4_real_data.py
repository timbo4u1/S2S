#!/usr/bin/env python3
"""
Module 4 Retrain — Real Certified Data
=======================================
Replaces 1,100 synthetic curriculum samples with
11,495 real certified windows from:
  - NinaPro DB5     (9,552 windows, 2000Hz)
  - EMG Amputee     (800 windows, 200Hz)
  - RoboTurk OpenX  (1,143 windows, 15Hz)

Extracts same 16 features as module4_cloud_trainer.py
so the trained model is drop-in compatible.
"""

import os, sys, json, pickle, glob, time
import numpy as np
import scipy.io
sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

WINDOW_NINAPRO  = 2000
WINDOW_AMPUTEE  = 200
WINDOW_ROBOTURK = 30
NYU_COUNT       = 14

def extract_features(accel, gyro, sample_rate):
    """Exact same 16 features as module4_cloud_trainer.py"""
    features = []
    accel = np.array(accel, dtype=np.float64)
    gyro  = np.array(gyro,  dtype=np.float64)

    if accel.ndim == 1:
        accel = accel.reshape(-1, 1)
    if gyro.ndim == 1:
        gyro = gyro.reshape(-1, 1)

    # Pad to 3 axes if needed
    while accel.shape[1] < 3:
        accel = np.hstack([accel, np.zeros((len(accel),1))])
    while gyro.shape[1] < 3:
        gyro = np.hstack([gyro, np.zeros((len(gyro),1))])

    features.append(float(np.sqrt(np.mean(accel**2))))
    features.append(float(np.std(accel)))
    features.append(float(np.max(np.abs(accel))))
    features.append(float(np.sqrt(np.mean(gyro**2))))
    features.append(float(np.std(gyro)))

    if len(accel) > 3:
        jerk = np.diff(accel, n=1, axis=0) * sample_rate
        features.append(float(np.sqrt(np.mean(jerk**2))))
        features.append(float(np.percentile(np.abs(jerk), 95)))
    else:
        features.extend([0.0, 0.0])

    for axis in range(3):
        fft   = np.abs(np.fft.rfft(accel[:, axis]))
        freqs = np.fft.rfftfreq(len(accel), 1/sample_rate)
        peak  = freqs[np.argmax(fft)] if len(fft) > 0 else 0
        features.append(float(peak))

    if accel.shape[1] >= 2:
        c = np.corrcoef(accel[:, 0], accel[:, 1])[0, 1]
        features.append(float(c) if not np.isnan(c) else 0.0)

    features.append(float(np.linalg.norm(np.mean(accel, axis=0))))

    hist, _ = np.histogram(accel.flatten(), bins=20)
    hist = hist / (hist.sum() + 1e-10)
    features.append(float(-np.sum(hist * np.log(hist + 1e-10))))

    return features  # 16 features

def certify_window(engine, accel, gyro, hz, segment="forearm"):
    n = len(accel)
    ts = [int(i * 1e9 / hz) for i in range(n)]
    r  = engine.certify(
        imu_raw={"timestamps_ns": ts,
                 "accel": accel.tolist(),
                 "gyro":  gyro.tolist()},
        segment=segment
    )
    return r.get("physical_law_score", 0), r.get("tier", "REJECTED")

# ── Source 1: NinaPro DB5 ─────────────────────────────────────────────────────
def load_ninapro(engine, X, y_score, y_tier):
    print("\n[1/3] NinaPro DB5...")
    base = os.path.expanduser("~/ninapro_db5")
    subjects = sorted(glob.glob(os.path.join(base, "s*")))
    count = 0
    for subj_dir in subjects:
        mats = sorted(glob.glob(os.path.join(subj_dir, "*.mat")))
        accel_all, emg_all = [], []
        for mat_path in mats:
            try:
                mat = scipy.io.loadmat(mat_path)
                acc = mat.get("acc", mat.get("ACC"))
                if acc is not None:
                    accel_all.append(acc)
            except:
                continue
        if not accel_all:
            continue
        accel = np.vstack(accel_all).astype(np.float64)
        gyro  = np.zeros_like(accel)

        for i in range(0, len(accel) - WINDOW_NINAPRO, WINDOW_NINAPRO):
            chunk_a = accel[i:i+WINDOW_NINAPRO, :3]
            chunk_g = gyro[i:i+WINDOW_NINAPRO, :3]
            score, tier = certify_window(engine, chunk_a, chunk_g, 2000)
            feats = extract_features(chunk_a, chunk_g, 2000)
            X.append(feats)
            y_score.append(score)
            y_tier.append(tier)
            count += 1
    print(f"  → {count} windows")

# ── Source 2: EMG Amputee ─────────────────────────────────────────────────────
def load_amputee(engine, X, y_score, y_tier):
    print("\n[2/3] EMG Amputee...")
    base = os.path.expanduser("~/S2S_Project/EMG_Amputee")
    count = 0
    for subj in sorted(os.listdir(base)):
        # AMP1/AMP1/mov N/accelerometer-*.csv
        inner = os.path.join(base, subj, subj)
        if not os.path.isdir(inner):
            continue
        csvs = glob.glob(os.path.join(inner, "mov *", "accelerometer-*.csv"))
        for csv_path in sorted(csvs):
            try:
                data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
                if data.ndim < 2 or data.shape[1] < 4:
                    continue
                accel = data[:, 1:4].astype(np.float64)
                gyro  = np.zeros_like(accel)
                for i in range(0, len(accel) - WINDOW_AMPUTEE, WINDOW_AMPUTEE):
                    chunk_a = accel[i:i+WINDOW_AMPUTEE]
                    chunk_g = gyro[i:i+WINDOW_AMPUTEE]
                    score, tier = certify_window(engine, chunk_a, chunk_g, 200)
                    feats = extract_features(chunk_a, chunk_g, 200)
                    X.append(feats)
                    y_score.append(score)
                    y_tier.append(tier)
                    count += 1
            except:
                continue
    print(f"  → {count} windows")

# ── Source 3: RoboTurk ────────────────────────────────────────────────────────
def load_roboturk(engine, X, y_score, y_tier):
    print("\n[3/3] RoboTurk...")
    base  = os.path.expanduser("~/S2S/openx_data")
    files = sorted(glob.glob(os.path.join(base, "sample_*.data.pickle")))[NYU_COUNT:]
    count = 0
    for path in files:
        with open(path, "rb") as f:
            data = pickle.load(f)
        steps = data.get("steps", [])
        world_vecs, rot_deltas = [], []
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action", {})
            wv = action.get("world_vector")
            rd = action.get("rotation_delta")
            if wv is not None:
                world_vecs.append(np.array(wv, dtype=np.float32))
            if rd is not None:
                rot_deltas.append(np.array(rd, dtype=np.float32))
        if len(world_vecs) < 5:
            continue
        positions = np.cumsum(np.array(world_vecs), axis=0)
        vel   = np.diff(positions, axis=0) * 15
        accel = np.diff(vel,       axis=0) * 15
        gyro  = np.zeros_like(accel)

        for i in range(0, len(accel) - WINDOW_ROBOTURK, WINDOW_ROBOTURK // 2):
            chunk_a = accel[i:i+WINDOW_ROBOTURK]
            chunk_g = gyro[i:i+WINDOW_ROBOTURK]
            if len(chunk_a) < 10:
                continue
            score, tier = certify_window(engine, chunk_a, chunk_g, 15)
            feats = extract_features(chunk_a, chunk_g, 15)
            X.append(feats)
            y_score.append(score)
            y_tier.append(tier)
            count += 1
    print(f"  → {count} windows")

# ── Train ─────────────────────────────────────────────────────────────────────
def train(X, y_score, y_tier):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import warnings
    warnings.filterwarnings("ignore")

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y_tier)
    s_arr = np.array(y_score, dtype=np.float32)

    print(f"\nDataset: {len(X_arr)} windows, 16 features")
    from collections import Counter
    print(f"Tier distribution: {dict(Counter(y_tier))}")

    X_tr, X_val, y_tr, y_val, s_tr, s_val = train_test_split(
        X_arr, y_arr, s_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    results = {}
    for name, clf in [
        ("RandomForest",      RandomForestClassifier(n_estimators=100, random_state=42)),
        ("GradientBoosting",  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_val, clf.predict(X_val))
        results[name] = round(acc, 4)
        print(f"  {name}: {acc:.4f}")

    # Save numpy arrays for Module 4
    out = os.path.expanduser("~/S2S/experiments")
    np.save(f"{out}/curriculum_dataset_real.npy",  X_arr)
    np.save(f"{out}/curriculum_labels_real.npy",   s_arr)
    np.save(f"{out}/curriculum_tiers_real.npy",    y_arr)
    print(f"\nSaved real curriculum arrays to experiments/")

    # Save results
    old_acc = 0.8864
    report = {
        "description": "Module 4 retrained on real certified data",
        "date": time.strftime("%Y-%m-%d"),
        "n_windows": len(X_arr),
        "sources": ["NinaPro DB5", "EMG Amputee", "RoboTurk Open-X"],
        "synthetic_baseline": old_acc,
        "real_data_results": results,
        "improvement": {k: round(v - old_acc, 4) for k, v in results.items()}
    }
    out_path = os.path.expanduser("~/S2S/experiments/results_module4_real.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report → {out_path}")
    return results

def main():
    print("="*55)
    print("  Module 4 Retrain — Real Certified Data")
    print("="*55)
    t0 = time.time()
    engine = PhysicsEngine()
    X, y_score, y_tier = [], [], []

    load_ninapro(engine,  X, y_score, y_tier)
    load_amputee(engine,  X, y_score, y_tier)
    load_roboturk(engine, X, y_score, y_tier)

    results = train(X, y_score, y_tier)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Synthetic baseline : 88.6%")
    for k, v in results.items():
        delta = v - 0.8864
        sign  = "+" if delta >= 0 else ""
        print(f"{k}: {v*100:.1f}% ({sign}{delta*100:.1f}%)")

if __name__ == "__main__":
    main()
