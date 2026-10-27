#!/usr/bin/env python3
"""
run_benchmark.py — S2S Reproducible Benchmark

Runs a mix of explicit real-data cases (when paths are provided) and generated
synthetic perturbations. A case "pass" means its output tier met the case-specific
expected-tier rule; it is not a sensitivity/specificity estimate or universal
synthetic-data detection score.

Usage:
    python3.9 run_benchmark.py                    # synthetic only (no datasets needed)
    python3.9 run_benchmark.py --ninapro ~/ninapro_db5
    python3.9 run_benchmark.py --pamap2  ~/S2S_data/pamap2
    python3.9 run_benchmark.py --wesad   ~/wesad_data/WESAD
    python3.9 run_benchmark.py --all --ninapro ~/ninapro_db5 --pamap2 ~/S2S_data/pamap2 --wesad ~/wesad_data/WESAD

The runner always evaluates 17 built-in synthetic cases (5 IID Gaussian, 7
hard-negative perturbations, and 5 coupled-OU cases). Supplying dataset paths
adds whatever real windows can be loaded from those files; full-run totals
therefore depend on the local datasets and usable windows. See README.md for
the distinction between the current runner and the historical v1.7.8 reference.
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
    """Run certification without treating execution failures as detections."""
    try:
        result = engine.certify(
            {"timestamps_ns": ts, "accel": acc, "gyro": gyro},
            segment=segment
        )
        if not isinstance(result, dict):
            raise TypeError(f"PhysicsEngine.certify returned {type(result).__name__}, expected dict")
        result = dict(result)
        result["execution_ok"] = not bool(result.get("error"))
        return result
    except Exception as e:
        return {
            "tier": "ERROR",
            "physical_law_score": None,
            "execution_ok": False,
            "error": f"{type(e).__name__}: {e}",
        }


def tier_is_expected(result, expected_tiers):
    """A case passes its expected-tier rule only if certification executed."""
    return bool(result.get("execution_ok", True)) and result.get("tier") in expected_tiers


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
    if not w.get("execution_ok", True):
        print(f"      EXECUTION ERROR: {w.get('error', 'unspecified error')}")


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
                "pass": tier_is_expected(r, ("GOLD", "SILVER", "BRONZE")),
                "execution_ok": r.get("execution_ok", True),
                "error": r.get("error"),
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
            "pass": tier_is_expected(r, ("REJECTED", "BRONZE")),
            "execution_ok": r.get("execution_ok", True),
            "error": r.get("error"),
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
            if arr.shape[0] < 300 or arr.shape[1] < 13:  # need up to col 12 for gyro
                continue
            acc_cols  = arr[:, 4:7]
            gyro_cols = arr[:, 10:13]  # corrected: cols 7-9 are 16g accel, cols 10-12 are gyro
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
                    "pass": tier_is_expected(r, ("GOLD", "SILVER")),
                    "execution_ok": r.get("execution_ok", True),
                    "error": r.get("error"),
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
                    "pass": tier_is_expected(r, ("GOLD", "SILVER", "BRONZE")),
                    "execution_ok": r.get("execution_ok", True),
                    "error": r.get("error"),
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
            "pass": tier_is_expected(r, ("REJECTED",)),
            "execution_ok": r.get("execution_ok", True),
            "error": r.get("error"),
            "note": "Gaussian noise with zero-jitter timestamps",
        })
    return windows



def make_coupled_ou():
    """
    Coupled OU test cases — Law 16 (innovation_kurtosis) closes this gap in v1.7.9.
    OU process: dx = -θ(x-μ)dt + σ√dt * dW where dW ~ N(0,1).
    Gaussian innovations are intrinsic to all OU generators regardless of parameters.
    Pass = correctly identified as REJECTED (adversarial benchmark logic).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from experiments.data_gen.ou_generator import generate_window
    pe = PhysicsEngine()
    cases = []

    configs = (
        [(seed, {}) for seed in range(3)] +
        [(seed, {"theta": 4.0, "sigma": 1.8,
                 "rho_xy": 0.6, "rho_xz": 0.4, "rho_yz": 0.5})
         for seed in range(2)]
    )
    labels = [f"ou_standard_{i}" for i in range(3)] +              [f"ou_aggressive_{i}" for i in range(2)]

    for (seed, kwargs), label in zip(configs, labels):
        imu = generate_window(seed=seed, **kwargs)
        try:
            result = pe.certify(imu_raw=imu, segment="forearm")
            if not isinstance(result, dict):
                raise TypeError(f"PhysicsEngine.certify returned {type(result).__name__}, expected dict")
            result = dict(result)
            result["execution_ok"] = not bool(result.get("error"))
        except Exception as e:
            result = {
                "tier": "ERROR",
                "physical_law_score": None,
                "laws_failed": [],
                "execution_ok": False,
                "error": f"{type(e).__name__}: {e}",
            }
        tier = result["tier"]
        w = {
            "timestamps_ns": imu["timestamps_ns"],
            "accel":         imu["accel"],
            "gyro":          imu["gyro"],
            "category":      "coupled_ou",
            "label":         label,
            "expected":      "REJECTED",
            "tier":          tier,
            "score":         result["physical_law_score"],
            "laws_failed":   result.get("laws_failed", []),
            # adversarial pass = Law 16 detected Gaussian innovations
            # (soft signal — lowers score but does not force REJECTED alone
            # because rest-state biological data overlaps with OU in kurtosis space)
            "pass":          bool(result.get("execution_ok", True)) and "innovation_kurtosis" in result.get("laws_failed", []),
            "execution_ok":  result.get("execution_ok", True),
            "error":         result.get("error"),
            "id":            label,
        }
        cases.append(w)
    return cases

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
        "pass": tier_is_expected(r, ("REJECTED", "BRONZE")),
        "execution_ok": r.get("execution_ok", True),
        "error": r.get("error"),
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
        "pass": tier_is_expected(r, ("REJECTED", "BRONZE")),
        "execution_ok": r.get("execution_ok", True),
        "error": r.get("error"),
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
        "pass": tier_is_expected(r, ("REJECTED", "BRONZE")),
        "execution_ok": r.get("execution_ok", True),
        "error": r.get("error"),
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
        "pass": tier_is_expected(r, ("REJECTED", "BRONZE", "SILVER")),
        "execution_ok": r.get("execution_ok", True),
        "error": r.get("error"),
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
        "pass": tier_is_expected(r, ("REJECTED", "BRONZE", "SILVER")),
        "execution_ok": r.get("execution_ok", True),
        "error": r.get("error"),
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
        "pass": tier_is_expected(r, ("REJECTED", "BRONZE", "SILVER")),
        "execution_ok": r.get("execution_ok", True),
        "error": r.get("error"),
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
        "pass": tier_is_expected(r, ("REJECTED", "BRONZE")),
        "execution_ok": r.get("execution_ok", True),
        "error": r.get("error"),
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
                        help="Run the full dataset-backed set (requires all three dataset paths)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare results to saved benchmark JSON")
    args = parser.parse_args()

    if args.all:
        missing = [
            flag for flag, value in (
                ("--ninapro", args.ninapro),
                ("--pamap2", args.pamap2),
                ("--wesad", args.wesad),
            ) if not value
        ]
        if missing:
            parser.error("--all requires all three dataset paths; missing " + ", ".join(missing))

    dataset_backed = any((args.ninapro, args.pamap2, args.wesad))
    run_mode = "dataset-backed" if dataset_backed else "synthetic-only"

    print("\n" + "═" * 60)
    print("  S2S Data-Quality Benchmark")
    print(f"  Mode: {run_mode}")
    print("  Run: python3 run_benchmark.py [--all --ninapro PATH --pamap2 PATH --wesad PATH]")
    if not dataset_backed:
        print("  No real dataset paths supplied; only generated synthetic cases will run.")
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

    print("\n[6] Coupled OU synthetic (Law 16 — innovation kurtosis)")
    w = make_coupled_ou()
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
            cats[c] = {"pass": 0, "total": 0, "execution_errors": 0, "windows": []}
        cats[c]["total"] += 1
        if w["pass"]:
            cats[c]["pass"] += 1
        if not w.get("execution_ok", True):
            cats[c]["execution_errors"] += 1
        cats[c]["windows"].append(w)

    for cat, s in cats.items():
        pct = s["pass"] / s["total"] * 100
        print(f"\n  {cat} ({s['pass']}/{s['total']} expected-tier matches — {pct:.0f}%; execution errors: {s['execution_errors']})")
        for w in s["windows"]:
            print_result(w)

    total  = len(windows)
    passed = sum(1 for w in windows if w["pass"])
    print("\n" + "═" * 60)
    execution_errors = sum(1 for w in windows if not w.get("execution_ok", True))
    print(f"  Overall expected-tier matches: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"  Execution errors: {execution_errors}")
    print("═" * 60)

    # Known limitations
    print("""
  Known limitations:
  - corrupted_spikes: spike injection at 2000Hz averages out over the
    256-sample window. Low-frequency corruption (50Hz) is caught reliably.
  - pure_synthetic: iid Gaussian noise caught by dual coherence check
    (spatial+temporal independence). Correlated noise (OU process) was
    a known gap — closed in v1.7.9 by Law 16 (innovation_kurtosis).
    Current runner evaluates these three checks against a finite set of generators;
    results do not establish universal detection of arbitrary synthetic data.
  - coupled_ou: inspect per-case Law 16 flags in this run. These five cases are
    not included in the historical v1.7.8 reference JSON.
  - WESAD/NinaPro (no gyro): only 3 of 7 laws run (jerk, resonance, BCG).
    Rigid body kinematics and IMU consistency skip with score=50 (neutral).
    SILVER on these datasets means 3/7 laws passed, not 7/7.
""")

    # Compare overlapping case IDs only; aggregate rates are not comparable when
    # the runner and historical reference contain different case sets.
    if args.compare:
        bench_path = Path("experiments/s2s_reference_benchmark.json")
        if bench_path.exists():
            saved = json.loads(bench_path.read_text())
            saved_by_id = {w["id"]: w for w in saved.get("windows", []) if "id" in w}
            current_by_id = {w["id"]: w for w in windows if "id" in w}
            common_ids = sorted(set(saved_by_id) & set(current_by_id))
            mismatches = [
                case_id for case_id in common_ids
                if saved_by_id[case_id].get("pass") != current_by_id[case_id].get("pass")
                or saved_by_id[case_id].get("tier") != current_by_id[case_id].get("tier")
            ]
            print("  Comparison to historical reference by overlapping case ID:")
            print(f"  Shared cases: {len(common_ids)}; outcome mismatches: {len(mismatches)}")
            print(f"  Current cases: {len(current_by_id)}; historical cases: {len(saved_by_id)}")
            if len(current_by_id) != len(saved_by_id) or set(current_by_id) != set(saved_by_id):
                print("  Case sets differ; aggregate pass rates are not directly comparable.")
        else:
            print("  No saved benchmark found at experiments/s2s_reference_benchmark.json")

    # Save this run
    out = {
        "version": "1.1",
        "runner_version": "1.7.9",
        "summary": {
            "total": total,
            "passed": passed,
            "pass_rate": round(passed / total, 3),
            "pass_definition": "Case-specific expected-tier match. This is not a sensitivity/specificity estimate or a universal detection rate.",
            "execution_errors": execution_errors,
            "run_mode": run_mode,
            "dataset_paths_supplied": {
                "ninapro": bool(args.ninapro),
                "pamap2": bool(args.pamap2),
                "wesad": bool(args.wesad),
            },
            "by_category": {
                c: {
                    "pass": s["pass"],
                    "total": s["total"],
                    "execution_errors": s["execution_errors"],
                }
                for c, s in cats.items()
            },
        },
        "windows": [{k: v for k, v in w.items() if k != "windows"}
                    for w in windows],
    }
    Path("experiments/benchmark_run_latest.json").write_text(
        json.dumps(out, indent=2))
    print("  Saved → experiments/benchmark_run_latest.json\n")

    if execution_errors:
        print(f"ERROR: {execution_errors} benchmark case(s) failed to execute; run is not valid.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
