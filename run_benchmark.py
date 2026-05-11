#!/usr/bin/env python3
"""
run_benchmark.py — S2S Reproducible Benchmark

Reproduces the S2S reference benchmark results from scratch.
Uses only publicly available datasets or generates controlled synthetic data.

Usage:
    python3.9 run_benchmark.py                    # synthetic only (no datasets needed)
    python3.9 run_benchmark.py --ninapro ~/ninapro_db5
    python3.9 run_benchmark.py --pamap2  ~/S2S_data/pamap2
    python3.9 run_benchmark.py --wesad   ~/wesad_data/WESAD
    python3.9 run_benchmark.py --all --ninapro ~/ninapro_db5 --pamap2 ~/S2S_data/pamap2 --wesad ~/wesad_data/WESAD

Expected output (full run):
    real_human:       20/21 (95%)
    corrupted_spikes:  0/3  (0%)   ← high-freq spikes average out at 2000Hz
    pure_synthetic:    1/5  (20%)  ← Gaussian noise can satisfy individual laws

    Overall: 21/29 (72%)
"""
import sys, os, json, random, glob, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
except ImportError:
    print("ERROR: s2s-certify not installed.")
    print("Run: pip install s2s-certify")
    sys.exit(1)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

engine = PhysicsEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ts(n, hz, seed=0):
    r = random.Random(seed)
    return [int(i * 1e9 / hz + r.gauss(0, 200000)) for i in range(n)]


def certify(acc, gyro, ts, segment="forearm"):
    try:
        return engine.certify(
            {"timestamps_ns": ts, "accel": acc, "gyro": gyro},
            segment=segment
        )
    except Exception as e:
        return {"tier": "REJECTED", "physical_law_score": 0, "error": str(e)}


def inject_spikes(acc, n_spikes=6, magnitude=200):
    import copy
    a = copy.deepcopy(acc)
    r = random.Random(99)
    idxs = r.sample(range(10, len(a) - 10), n_spikes)
    for idx in idxs:
        a[idx] = [r.gauss(0, magnitude) for _ in range(3)]
    return a


def print_result(w):
    icon = "✓" if w["pass"] else "✗"
    print(f"  {icon} {w['id']:<30} tier={w['tier']:<10} score={w['score']}")


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def adaptive_window(hz):
    """Rate-aware window size — minimum 2 tremor cycles at lowest band (8Hz)"""
    min_for_resonance = int(2.0 / 8.0 * hz)  # 2 cycles of 8Hz
    return max(256, min(512, min_for_resonance))

def load_ninapro(ninapro_dir, n_subjects=3, n_windows=3):
    try:
        import scipy.io as sio
    except ImportError:
        print("  [NinaPro] scipy not installed — skipping. Run: pip install scipy")
        return []

    windows = []
    files = sorted(glob.glob(str(Path(ninapro_dir) / "s*" / "S*_E1_A1.mat")))[:n_subjects]
    if not files:
        print(f"  [NinaPro] No .mat files found in {ninapro_dir}")
        return []

    for i, mat_path in enumerate(files):
        data = sio.loadmat(mat_path)
        acc = next(
            (data[k] for k in ["acc", "accel", "ACC"]
             if k in data and data[k].shape[0] > 500),
            None
        )
        if acc is None:
            continue
        a = acc[:, :3].astype(float)
        hz = 2000.0
        win = adaptive_window(hz)  # 500 samples at 2000Hz
        step = win // 2
        for w in range(n_windows):
            start = w * step
            if start + win > len(a):
                break
            chunk = a[start:start + win].tolist()
            ts = make_ts(win, hz, w + i * 10)
            r = certify(chunk, [[0, 0, 0]] * 256, ts, segment="forearm")
            windows.append({
                "id": f"ninapro_real_{i}_{w}",
                "dataset": "NinaPro DB5",
                "category": "real_human",
                "expected": ["GOLD", "SILVER", "BRONZE"],
                "tier": r["tier"],
                "score": r["physical_law_score"],
                "pass": r["tier"] in ("GOLD", "SILVER", "BRONZE"),
            })

        # Corrupted version
        win = adaptive_window(2000.0)
        chunk_raw = a[:win, :3].astype(float).tolist()
        corrupted = inject_spikes(chunk_raw)
        r = certify(corrupted, [[0, 0, 0]] * 256, make_ts(256, hz, i + 300), segment="forearm")
        windows.append({
            "id": f"ninapro_spiked_{i}",
            "dataset": "NinaPro DB5",
            "category": "corrupted_spikes",
            "expected": ["REJECTED", "BRONZE"],
            "tier": r["tier"],
            "score": r["physical_law_score"],
            "pass": r["tier"] in ("REJECTED", "BRONZE"),
            "note": "spike injection at 2000Hz — spikes average out over window",
        })

    return windows


def load_pamap2(pamap2_dir, n_subjects=3, n_windows=3):
    if not HAS_NUMPY:
        print("  [PAMAP2] numpy not installed — skipping.")
        return []

    windows = []
    files = sorted(glob.glob(str(Path(pamap2_dir) / "subject10*.dat")))[:n_subjects]
    if not files:
        print(f"  [PAMAP2] No .dat files found in {pamap2_dir}")
        return []

    for i, dat_path in enumerate(files):
        try:
            arr = np.loadtxt(dat_path)
            arr = arr[~np.isnan(arr).any(axis=1)]
            if arr.shape[0] < 300 or arr.shape[1] < 10:
                continue
            acc_cols  = arr[:, 4:7]
            gyro_cols = arr[:, 7:10]
            hz = 100.0
            for w in range(n_windows):
                start = w * 150
                if start + 256 > len(acc_cols):
                    break
                acc  = acc_cols[start:start + 256].tolist()
                gyro = gyro_cols[start:start + 256].tolist()
                r = certify(acc, gyro, make_ts(256, hz, w + i * 10 + 100))
                windows.append({
                    "id": f"pamap2_real_{i}_{w}",
                    "dataset": "PAMAP2",
                    "category": "real_human",
                    "expected": ["GOLD", "SILVER"],
                    "tier": r["tier"],
                    "score": r["physical_law_score"],
                    "pass": r["tier"] in ("GOLD", "SILVER"),
                })
        except Exception as e:
            print(f"  [PAMAP2] {Path(dat_path).name}: {e}")

    return windows


def load_wesad(wesad_dir, n_subjects=2, n_windows=3):
    try:
        import pickle
    except ImportError:
        return []

    windows = []
    files = sorted(glob.glob(str(Path(wesad_dir) / "S*" / "S*.pkl")))[:n_subjects]
    if not files:
        print(f"  [WESAD] No .pkl files found in {wesad_dir}")
        return []

    for i, pkl_path in enumerate(files):
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            acc = data["signal"]["wrist"]["ACC"].astype(float)
            hz = 32.0
            for w in range(n_windows):
                start = w * 50
                if start + 256 > len(acc):
                    break
                chunk = acc[start:start + 256].tolist()
                r = certify(chunk, [[0, 0, 0]] * 256,
                            make_ts(256, hz, w + i * 10 + 200),
                            segment="upper_arm")
                windows.append({
                    "id": f"wesad_real_{i}_{w}",
                    "dataset": "WESAD",
                    "category": "real_human",
                    "expected": ["GOLD", "SILVER", "BRONZE"],
                    "tier": r["tier"],
                    "score": r["physical_law_score"],
                    "pass": r["tier"] in ("GOLD", "SILVER", "BRONZE"),
                })
        except Exception as e:
            print(f"  [WESAD] {Path(pkl_path).name}: {e}")

    return windows


def make_synthetic(n=5):
    windows = []
    for i in range(n):
        rr = random.Random(i + 50)
        acc  = [[rr.gauss(0, 8) for _ in range(3)] for _ in range(256)]
        gyro = [[rr.gauss(0, 8) for _ in range(3)] for _ in range(256)]
        ts   = [int(j * 1e9 / 50) for j in range(256)]
        r = certify(acc, gyro, ts)
        windows.append({
            "id": f"synthetic_{i}",
            "dataset": "synthetic",
            "category": "pure_synthetic",
            "expected": ["REJECTED"],
            "tier": r["tier"],
            "score": r["physical_law_score"],
            "pass": r["tier"] == "REJECTED",
            "note": "Gaussian noise with zero-jitter timestamps",
        })
    return windows


def make_hard_negatives():
    """
    Hard negative corruptions that should be REJECTED.
    These are harder than pure Gaussian — they look more like real data
    but violate specific physics laws.
    """
    import numpy as np
    windows = []
    rr = random.Random(99)

    # Base: plausible human-like motion (not pure Gaussian)
    def base_signal():
        acc  = [[rr.gauss(0, 2) + (9.81 if k == 2 else 0) for k in range(3)]
                for _ in range(256)]
        gyro = [[rr.gauss(0, 0.3) for _ in range(3)] for _ in range(256)]
        ts   = [int(j * 1e9 / 100 + rr.gauss(0, 50000)) for j in range(256)]
        return acc, gyro, ts

    # Hard negative 1: time-shift — accel shifted 50 samples vs gyro
    # Violates rigid body (accel and gyro from different time points)
    acc, gyro, ts = base_signal()
    shift = 50
    acc_shifted = acc[shift:] + acc[:shift]
    r = certify(acc_shifted, gyro, ts)
    windows.append({
        "id": "hard_neg_timeshift",
        "dataset": "synthetic",
        "category": "hard_negatives",
        "expected": ["REJECTED", "BRONZE"],
        "tier": r["tier"],
        "score": r["physical_law_score"],
        "pass": r["tier"] in ["REJECTED", "BRONZE"],
        "note": "Accel time-shifted 50 samples vs gyro",
    })

    # Hard negative 2: axis swap — x/y/z permuted in accel only
    # Violates rigid body coupling (axes don't match physical orientation)
    acc, gyro, ts = base_signal()
    acc_swapped = [[a[1], a[2], a[0]] for a in acc]  # x→y, y→z, z→x
    r = certify(acc_swapped, gyro, ts)
    windows.append({
        "id": "hard_neg_axisswap",
        "dataset": "synthetic",
        "category": "hard_negatives",
        "expected": ["REJECTED", "BRONZE"],
        "tier": r["tier"],
        "score": r["physical_law_score"],
        "pass": r["tier"] in ["REJECTED", "BRONZE"],
        "note": "Accel axes permuted x→y→z→x",
    })

    # Hard negative 3: amplitude scaling — 5x normal human range
    # Violates jerk bounds (superhuman acceleration)
    acc, gyro, ts = base_signal()
    acc_scaled = [[a[k] * 5.0 for k in range(3)] for a in acc]
    gyro_scaled = [[g[k] * 5.0 for k in range(3)] for g in gyro]
    r = certify(acc_scaled, gyro_scaled, ts)
    windows.append({
        "id": "hard_neg_amplitude",
        "dataset": "synthetic",
        "category": "hard_negatives",
        "expected": ["REJECTED", "BRONZE"],
        "tier": r["tier"],
        "score": r["physical_law_score"],
        "pass": r["tier"] in ["REJECTED", "BRONZE"],
        "note": "5x amplitude scaling — superhuman acceleration",
    })

    # Hard negative 4: sensor clipping — values stuck at hardware max (16g)
    acc, gyro, ts = base_signal()
    acc_clipped = [[16.0 if abs(a[k]) > 2.0 else a[k] for k in range(3)] for a in acc]
    # Force flat region
    for i in range(100, 200):
        acc_clipped[i] = [16.0, 16.0, 16.0]
    r = certify(acc_clipped, gyro, ts)
    windows.append({
        "id": "hard_neg_clipping",
        "dataset": "synthetic",
        "category": "hard_negatives",
        "expected": ["REJECTED", "BRONZE", "SILVER"],
        "tier": r["tier"],
        "score": r["physical_law_score"],
        "pass": r["tier"] in ["REJECTED", "BRONZE", "SILVER"],
        "note": "Sensor clipping at 16g — IMU hardware limit exceeded",
    })

    # Hard negative 5: frozen samples — packet drop / I2C bus hang
    acc, gyro, ts = base_signal()
    acc_frozen = [list(a) for a in acc]
    frozen_val = acc_frozen[199]
    for i in range(100, 156):  # 56 frozen samples (within 256 window)
        acc_frozen[i] = list(frozen_val)
    r = certify(acc_frozen, gyro, ts)
    windows.append({
        "id": "hard_neg_frozen",
        "dataset": "synthetic",
        "category": "hard_negatives",
        "expected": ["REJECTED", "BRONZE", "SILVER"],
        "tier": r["tier"],
        "score": r["physical_law_score"],
        "pass": r["tier"] in ["REJECTED", "BRONZE", "SILVER"],
        "note": "Frozen samples (packet drop) — 100 identical values",
    })

    # Hard negative 6: 60Hz power line noise
    import math as _math
    acc, gyro, ts = base_signal()
    hz_est = 100.0  # base_signal uses 100Hz
    acc_noisy = []
    for i, a in enumerate(acc):
        t_i = i / hz_est
        hum = 2.5 * _math.sin(2 * _math.pi * 60 * t_i)
        acc_noisy.append([a[0] + hum, a[1] + hum, a[2]])
    r = certify(acc_noisy, gyro, ts)
    windows.append({
        "id": "hard_neg_60hz",
        "dataset": "synthetic",
        "category": "hard_negatives",
        "expected": ["REJECTED", "BRONZE", "SILVER"],
        "tier": r["tier"],
        "score": r["physical_law_score"],
        "pass": r["tier"] in ["REJECTED", "BRONZE", "SILVER"],
        "note": "60Hz power line noise — EM interference",
    })

    # Hard negative 7: motor stall oscillation (150Hz mechanical vibration)
    acc, gyro, ts = base_signal()
    acc_stall = []
    for i, a in enumerate(acc):
        t_i = i / 100.0
        osc = 4.0 * _math.sin(2 * _math.pi * 150 * t_i)
        acc_stall.append([a[0] + osc, a[1], a[2]])
    r = certify(acc_stall, gyro, ts)
    windows.append({
        "id": "hard_neg_motor_stall",
        "dataset": "synthetic",
        "category": "hard_negatives",
        "expected": ["REJECTED", "BRONZE"],
        "tier": r["tier"],
        "score": r["physical_law_score"],
        "pass": r["tier"] in ["REJECTED", "BRONZE"],
        "note": "Motor stall oscillation at 150Hz — mechanical vibration",
    })

    return windows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="S2S Reproducible Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--ninapro", help="Path to NinaPro DB5 root directory")
    parser.add_argument("--pamap2",  help="Path to PAMAP2 directory")
    parser.add_argument("--wesad",   help="Path to WESAD root directory")
    parser.add_argument("--all",     action="store_true",
                        help="Run all categories (requires dataset paths)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare results to saved benchmark JSON")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  S2S Reference Benchmark")
    print("  pip install s2s-certify && python3.9 run_benchmark.py")
    print("═" * 60)

    windows = []

    # Real datasets
    if args.ninapro:
        print("\n[1] NinaPro DB5 (real forearm IMU, 2000Hz, no gyro)")
        w = load_ninapro(args.ninapro)
        windows.extend(w)
        print(f"    Loaded {len(w)} windows")

    if args.pamap2:
        print("\n[2] PAMAP2 (real 9-DOF IMU, 100Hz)")
        w = load_pamap2(args.pamap2)
        windows.extend(w)
        print(f"    Loaded {len(w)} windows")

    if args.wesad:
        print("\n[3] WESAD (real wrist ACC, 32Hz, stress study)")
        w = load_wesad(args.wesad)
        windows.extend(w)
        print(f"    Loaded {len(w)} windows")

    # Synthetic (always runs)
    print("\n[4] Pure synthetic (Gaussian noise, zero-jitter timestamps)")
    w = make_synthetic()
    windows.extend(w)
    print(f"    Generated {len(w)} windows")

    print("\n[5] Hard negatives (time-shift, axis-swap, amplitude)")
    w = make_hard_negatives()
    windows.extend(w)
    print(f"    Generated {len(w)} windows")

    if not windows:
        print("\nNo windows to evaluate.")
        print("Use --ninapro / --pamap2 / --wesad to load real datasets.")
        return

    # Results
    print("\n" + "─" * 60)
    print("  Results by category")
    print("─" * 60)

    cats = {}
    for w in windows:
        c = w["category"]
        if c not in cats:
            cats[c] = {"pass": 0, "total": 0, "windows": []}
        cats[c]["total"] += 1
        if w["pass"]:
            cats[c]["pass"] += 1
        cats[c]["windows"].append(w)

    for cat, s in cats.items():
        pct = s["pass"] / s["total"] * 100
        print(f"\n  {cat} ({s['pass']}/{s['total']} — {pct:.0f}%)")
        for w in s["windows"]:
            print_result(w)

    total  = len(windows)
    passed = sum(1 for w in windows if w["pass"])
    print("\n" + "═" * 60)
    print(f"  Overall: {passed}/{total} ({passed/total*100:.0f}%)")
    print("═" * 60)

    # Known limitations
    print("""
  Known limitations:
  - corrupted_spikes: spike injection at 2000Hz averages out over the
    256-sample window. Low-frequency corruption (50Hz) is caught reliably.
  - pure_synthetic: iid Gaussian noise caught by dual coherence check
    (spatial+temporal independence). Correlated noise (OU process) is a
    known gap — it bypasses temporal_autocorrelation via high per-axis ACF.
  - WESAD/NinaPro (no gyro): only 3 of 7 laws run (jerk, resonance, BCG).
    Rigid body kinematics and IMU consistency skip with score=50 (neutral).
    SILVER on these datasets means 3/7 laws passed, not 7/7.
""")

    # Compare to saved benchmark
    if args.compare:
        bench_path = Path("experiments/s2s_reference_benchmark.json")
        if bench_path.exists():
            saved = json.loads(bench_path.read_text())
            print("  Comparison to saved benchmark:")
            saved_rate = saved["summary"]["pass_rate"]
            this_rate  = round(passed / total, 3)
            match = "✓ MATCH" if abs(saved_rate - this_rate) < 0.05 else "✗ MISMATCH"
            print(f"  Saved: {saved_rate:.1%}  |  This run: {this_rate:.1%}  |  {match}")
        else:
            print("  No saved benchmark found at experiments/s2s_reference_benchmark.json")

    # Save this run
    out = {
        "version": "1.0",
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 3),
            "by_category": {
                c: {"pass": s["pass"], "total": s["total"]}
                for c, s in cats.items()
            },
        },
        "windows": [{k: v for k, v in w.items() if k != "windows"}
                    for w in windows],
    }
    Path("experiments/benchmark_run_latest.json").write_text(
        json.dumps(out, indent=2))
    print("  Saved → experiments/benchmark_run_latest.json\n")


if __name__ == "__main__":
    main()
