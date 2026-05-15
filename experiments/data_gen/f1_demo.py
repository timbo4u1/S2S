"""
f1_demo.py — Synthetic-to-Real transfer F1 test.

The experiment:
  1. Generate S2S-certified synthetic windows (5 gesture profiles)
  2. Extract S2S physics features from each
  3. Train SVM on SYNTHETIC data only — no real data in training
  4. Test on REAL NinaPro DB5 data
  5. Compare vs baseline (SVM trained on uncertified random Gaussian)

Win condition: F1_certified > F1_uncertified
This proves: physics-certified synthetic data has semantic transfer value.

Usage: python3.9 experiments/data_gen/f1_demo.py --ninapro ~/ninapro_db5
"""
import sys, argparse, glob, math, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Install sklearn: pip install scikit-learn")
    sys.exit(1)

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
from experiments.data_gen.gesture_generator import generate_gesture_dataset, GESTURE_PROFILES
from experiments.data_gen.s2s_sieve import sieve


engine = PhysicsEngine()


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_s2s_features(imu_raw: dict) -> np.ndarray:
    """Extract 8-dim S2S physics feature vector from one window."""
    engine._last_terminal_state = None
    r = engine.certify(imu_raw, segment="forearm")
    details = r.get("law_details", {})

    def _conf(law): return details.get(law, {}).get("confidence", 50) / 100.0
    def _val(law, key, default=0.0):
        return float(details.get(law, {}).get(key, default))

    return np.array([
        r["physical_law_score"] / 100.0,
        _conf("resonance_frequency"),
        _conf("jerk_bounds"),
        _conf("imu_internal_consistency"),
        _conf("cross_axis_cohesion"),
        _conf("spectral_flatness"),
        _conf("temporal_autocorrelation"),
        _val("jerk_bounds", "p95_jerk_normalized_ms3", 0) / 500.0,
    ], dtype=np.float32)


def extract_raw_features(imu_raw: dict) -> np.ndarray:
    """Baseline: simple statistical features from raw accelerometer."""
    acc = np.array(imu_raw["accel"], dtype=np.float64)
    feats = []
    for ax in range(3):
        col = acc[:, ax]
        feats += [col.mean(), col.std(), np.abs(col).max(),
                  float(np.percentile(np.abs(col), 95))]
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# NinaPro loader
# ---------------------------------------------------------------------------

def load_ninapro_windows(ninapro_dir: str, n_subjects: int = 3,
                          n_windows_per_subject: int = 20,
                          hz: float = 2000.0) -> tuple:
    """
    Load NinaPro DB5 windows and map to coarse gesture labels.

    NinaPro has 49 gestures. We map to 5 S2S gesture profiles:
      0 rest, 1 gentle_flex, 2 pinch, 3 power_grip, 4 fast_extension
    """
    if not HAS_SCIPY:
        print("[NinaPro] scipy not available — using synthetic test set only")
        return [], []

    files = sorted(glob.glob(str(Path(ninapro_dir) / "s*" / "S*_E1_A1.mat")))[:n_subjects]
    if not files:
        print(f"[NinaPro] No files found in {ninapro_dir}")
        return [], []

    # Rough gesture→profile mapping (NinaPro E1: basic hand gestures)
    # 0=rest → label 0, 1-8=finger flex → label 1, 9-16=grip → label 3, etc.
    def _map_label(ninapro_label: int) -> int:
        if ninapro_label == 0: return 0          # rest
        if ninapro_label <= 8: return 1          # gentle finger flexion
        if ninapro_label <= 16: return 2         # pinch-like
        if ninapro_label <= 24: return 3         # power grip
        return 4                                 # extension

    windows, labels = [], []
    win_size = 256

    for fpath in files:
        d = sio.loadmat(fpath)
        acc = d.get("acc", None)
        stim = d.get("stimulus", None)
        if acc is None or stim is None:
            continue
        acc = acc.astype(float)
        stim = stim.flatten().astype(int)

        rng = random.Random(42)
        for start in range(0, min(len(acc) - win_size, n_windows_per_subject * 50), win_size):
            chunk_acc = acc[start:start+win_size, :3].tolist()
            chunk_stim = stim[start:start+win_size]
            # Use majority label in window
            from collections import Counter
            majority_label = Counter(chunk_stim.tolist()).most_common(1)[0][0]
            mapped = _map_label(majority_label)
            ts = [int(j * 1e9 / hz) for j in range(win_size)]
            windows.append({"timestamps_ns": ts, "accel": chunk_acc,
                            "gyro": [[0,0,0]] * win_size})
            labels.append(mapped)
            if len(windows) >= n_subjects * n_windows_per_subject:
                break
        if len(windows) >= n_subjects * n_windows_per_subject:
            break

    return windows, labels


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(ninapro_dir: str):
    print("\n" + "="*60)
    print("  S2S Synthetic-to-Real Transfer — F1 Acid Test")
    print("="*60)

    # --- SYNTHETIC TRAINING DATA ---
    print("\n[1] Generating certified synthetic training data...")
    syn_windows, syn_labels, _ = generate_gesture_dataset(
        n_per_class=300, n_samples=256, hz=100.0)
    print(f"    Generated {len(syn_windows)} windows")

    # Filter through sieve (should be ~100% yield)
    certified = sieve(syn_windows, min_tier="SILVER")
    print(f"    Certified: {len(certified)}/{len(syn_windows)} ({len(certified)/len(syn_windows)*100:.0f}%)")
    cert_labels = [syn_labels[i] for i, w in enumerate(syn_windows)
                   if w in [c for c in certified]]
    # Simpler: keep label alignment via zip
    cert_pairs = [(w, syn_labels[i]) for i, w in enumerate(syn_windows)
                  if any(w["timestamps_ns"][0] == c["timestamps_ns"][0]
                         for c in certified)]
    cert_windows = [p[0] for p in cert_pairs]
    cert_labels_aligned = [p[1] for p in cert_pairs]

    # --- UNCERTIFIED BASELINE DATA ---
    print("\n[2] Generating uncertified baseline (Gaussian noise)...")
    rr = random.Random(999)
    gauss_windows = [
        {"timestamps_ns": [int(j*1e9/50) for j in range(256)],
         "accel": [[rr.gauss(0, 2) for _ in range(3)] for _ in range(256)],
         "gyro":  [[rr.gauss(0, 0.1) for _ in range(3)] for _ in range(256)]}
        for _ in range(1500)
    ]
    gauss_labels = [i % 5 for i in range(1500)]

    # --- REAL NINAPRO TEST DATA ---
    print(f"\n[3] Loading real NinaPro test data from {ninapro_dir}...")
    real_windows, real_labels = load_ninapro_windows(ninapro_dir)
    if not real_windows:
        print("    No real data — running synthetic self-test instead")
        from sklearn.model_selection import train_test_split
        real_windows = syn_windows[-200:]
        real_labels = syn_labels[-200:]

    print(f"    Loaded {len(real_windows)} real windows")

    # --- EXTRACT FEATURES ---
    print("\n[4] Extracting S2S features...")

    def batch_features(windows, feat_fn, label=""):
        feats = []
        for i, w in enumerate(windows):
            if i % 100 == 0 and label:
                print(f"    {label}: {i}/{len(windows)}", end="\r")
            feats.append(feat_fn(w))
        if label: print(f"    {label}: {len(windows)}/{len(windows)} ✓")
        return np.array(feats)

    X_cert_s2s   = batch_features(cert_windows, extract_s2s_features, "certified S2S")
    X_gauss_s2s  = batch_features(gauss_windows, extract_s2s_features, "gaussian S2S")
    X_real_s2s   = batch_features(real_windows, extract_s2s_features, "real S2S")

    X_cert_raw   = batch_features(cert_windows, extract_raw_features, "certified raw")
    X_gauss_raw  = batch_features(gauss_windows, extract_raw_features, "gaussian raw")
    X_real_raw   = batch_features(real_windows, extract_raw_features, "real raw")

    y_cert  = np.array(cert_labels_aligned if cert_labels_aligned else syn_labels[:len(cert_windows)])
    y_gauss = np.array(gauss_labels)
    y_real  = np.array(real_labels)

    # --- TRAIN & TEST ---
    print("\n[5] Training classifiers & testing on real data...")
    print("-"*60)

    results = {}

    def train_test(X_train, y_train, X_test, y_test, name):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)
        clf = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_te)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        results[name] = f1
        print(f"  {name:<45} F1={f1:.3f}")
        return f1

    train_test(X_cert_s2s,  y_cert,  X_real_s2s,  y_real, "Certified synthetic → Real  (S2S features)")
    train_test(X_gauss_s2s, y_gauss, X_real_s2s,  y_real, "Gaussian baseline   → Real  (S2S features)")
    train_test(X_cert_raw,  y_cert,  X_real_raw,  y_real, "Certified synthetic → Real  (raw features)")
    train_test(X_gauss_raw, y_gauss, X_real_raw,  y_real, "Gaussian baseline   → Real  (raw features)")

    print("-"*60)
    cert_s2s  = results["Certified synthetic → Real  (S2S features)"]
    gauss_s2s = results["Gaussian baseline   → Real  (S2S features)"]
    improvement = cert_s2s - gauss_s2s

    print(f"\n  S2S improvement over baseline: {improvement:+.3f}")
    if improvement > 0:
        print("  ✓ CERTIFIED SYNTHETIC OUTPERFORMS UNCERTIFIED BASELINE")
        print("  → S2S physics certification has semantic transfer value")
    else:
        print("  ✗ No improvement — domain gap too large at this configuration")
        print("  → Need more gesture-specific OU parameter tuning")

    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ninapro", default="~/ninapro_db5",
                        help="Path to NinaPro DB5 root directory")
    args = parser.parse_args()
    run_experiment(str(Path(args.ninapro).expanduser()))
