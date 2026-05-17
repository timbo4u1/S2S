"""
fill_validation.py — Phase 4: Three ablations.

Experiment A: real GOLD+SILVER only         → baseline F1
Experiment B: real + min-jerk fills (no OU) → augmented F1
Experiment C: placeholder for OU residual   → future

Training subjects: S1-S8
Test subjects:     S9-S10 (never seen in gap_detector or fill)

Features: 15 simple stats per window (no PyTorch needed)
Classifier: RandomForest (100 trees)
"""

import sys, json, glob, math, random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

try:
    import numpy as np
    import scipy.io as sio
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import LabelEncoder
except ImportError as e:
    print(f"ERROR: {e}\npip install scikit-learn scipy numpy"); sys.exit(1)

# ── config ────────────────────────────────────────────────────────────────────
NINAPRO_DIR  = Path.home() / "ninapro_db5"
HZ           = 200
WIN          = 256
N            = max(5, int(HZ * 0.05))
TRAIN_SUBJS  = list(range(1, 9))    # S1-S8
TEST_SUBJS   = [9, 10]              # S9-S10 — held out
AUG_RATIOS   = [0.20, 0.50, 1.00]
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── feature extraction ────────────────────────────────────────────────────────
def extract_features(window):
    """15 statistical features from raw 256×3 accelerometer window."""
    arr = np.array(window, dtype=float)   # 256 × 3
    feats = []
    for ax in range(3):
        col = arr[:, ax]
        feats += [col.mean(), col.std(), np.sqrt((col**2).mean())]  # mean, std, rms
    # cross-axis correlations
    feats += [np.corrcoef(arr[:,0], arr[:,1])[0,1],
              np.corrcoef(arr[:,1], arr[:,2])[0,1],
              np.corrcoef(arr[:,0], arr[:,2])[0,1]]
    # jerk magnitude rms
    jerk = np.diff(arr, axis=0) * HZ
    feats.append(np.sqrt((jerk**2).sum(axis=1)).mean())
    return feats   # 15 features

# ── data loader ───────────────────────────────────────────────────────────────
def load_real_windows(subj_ids, certify=True):
    """Load certified windows with gesture labels for given subjects."""
    files = sorted(glob.glob(str(NINAPRO_DIR / "s*" / "S*_E1_A1.mat")))
    subj_files = [f for f in files
                  if any(f"/s{i}/" in f.lower() or f"\\s{i}\\" in f.lower()
                         for i in subj_ids)]

    windows, labels = [], []
    pe = PhysicsEngine()

    for fpath in subj_files:
        subj = Path(fpath).stem.split("_")[0]
        try:
            data = sio.loadmat(fpath)
        except Exception as e:
            print(f"  {subj}: {e}"); continue

        acc_raw = next((data[k] for k in ["acc","accel","ACC"]
                        if k in data and hasattr(data[k],'shape')), None)
        stim_raw = next((data[k] for k in ["stimulus","restimulus"]
                         if k in data), None)
        if acc_raw is None or stim_raw is None:
            print(f"  {subj}: missing data"); continue

        a    = acc_raw[:, :3].astype(float)
        stim = stim_raw.flatten().astype(int)
        if len(stim) != len(a):
            stim = np.resize(stim, len(a))

        # window per gesture episode
        i = 0
        while i < len(a):
            label = stim[i]
            if label == 0: i += 1; continue
            j = i
            while j < len(a) and stim[j] == label: j += 1
            ep_acc = a[i:j].tolist()
            w = 0
            while w + WIN <= len(ep_acc):
                chunk = ep_acc[w:w+WIN]
                if certify:
                    ts = [int(k*1e9/HZ) for k in range(WIN)]
                    r  = pe.certify({"timestamps_ns":ts,"accel":chunk,
                                     "gyro":[[0,0,0]]*WIN})
                    if r["tier"] not in ("GOLD","SILVER"):
                        w += WIN; continue
                windows.append(extract_features(chunk))
                labels.append(int(label))
                w += WIN
            i = j

        print(f"  {subj}: {len(windows)} windows so far")

    return np.array(windows), np.array(labels)

# ── experiment ────────────────────────────────────────────────────────────────
def run_experiment(X_train, y_train, X_test, y_test, name):
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  {name}: F1={f1:.4f}  (n_train={len(y_train)}, n_test={len(y_test)})")
    return f1

def run():
    print("Loading training data (S1-S8)...")
    X_real, y_real = load_real_windows(TRAIN_SUBJS)
    print(f"  Real training windows: {len(X_real)}")

    print("\nLoading test data (S9-S10, held out)...")
    X_test, y_test = load_real_windows(TEST_SUBJS)
    print(f"  Test windows: {len(X_test)}")

    if len(X_real) == 0 or len(X_test) == 0:
        print("ERROR: insufficient data"); sys.exit(1)

    # load fills
    fills_path = Path(__file__).parent / "certified_fills.json"
    if not fills_path.exists():
        print("ERROR: certified_fills.json not found"); sys.exit(1)

    fills_data = json.load(open(fills_path))["fills"]
    X_fills = np.array([extract_features(f["full_window"]) for f in fills_data])
    y_fills = np.array([f["gesture_label"] for f in fills_data])
    print(f"  Certified fills: {len(X_fills)}")

    print("\n" + "="*52)
    print("ABLATION RESULTS")
    print("="*52)

    results = {}

    # Experiment A — real only
    results["A_real_only"] = run_experiment(
        X_real, y_real, X_test, y_test, "A — real only (baseline)")

    # Experiment B — real + fills at each ratio
    n_real = len(X_real)
    for ratio in AUG_RATIOS:
        n_add = int(n_real * ratio)
        if n_add > len(X_fills):
            n_add = len(X_fills)
        idx    = np.random.choice(len(X_fills), n_add, replace=False)
        X_aug  = np.vstack([X_real, X_fills[idx]])
        y_aug  = np.concatenate([y_real, y_fills[idx]])
        key    = f"B_fills_{int(ratio*100)}pct"
        results[key] = run_experiment(
            X_aug, y_aug, X_test, y_test,
            f"B — real + fills {int(ratio*100)}% augmentation")

    print()
    print("INTERPRETATION:")
    base = results["A_real_only"]
    any_improvement = False
    for ratio in AUG_RATIOS:
        key = f"B_fills_{int(ratio*100)}pct"
        delta = results[key] - base
        direction = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
        print(f"  {int(ratio*100)}% fills: {direction} {delta:+.4f} vs baseline")
        if delta > 0.005: any_improvement = True

    print()
    if any_improvement:
        print("  RESULT: Fills improve F1 — interpolation is useful")
        print("  NEXT: Add OU residual (Experiment C) to see if it helps further")
    else:
        print("  RESULT: Fills do not improve F1")
        print("  CONCLUSION: min-jerk interpolation in acceleration space")
        print("  does not help gesture classifier. Known limitation confirmed.")

    print("="*52)

    out = Path(__file__).parent / "ablation_results.json"
    json.dump(results, open(out,"w"), indent=2)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    run()
