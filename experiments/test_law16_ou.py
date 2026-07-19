#!/usr/bin/env python3
"""
Test Law 16 — Innovation Kurtosis
Verifies coupled OU (Cholesky) is caught by distributional check.

Usage: cd ~/S2S && python3 experiments/test_law16_ou.py
"""
import sys, os, math, random
sys.path.insert(0, os.path.expanduser("~/S2S"))

from experiments.data_gen.ou_generator import generate_window

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; W = "\033[97m"; X = "\033[0m"

def check_innovation_kurtosis(imu_raw):
    """
    Law 16 standalone — test before integrating into engine.
    OU uses Gaussian innovations by construction: excess kurtosis = 0.
    Real biological signals have leptokurtic innovations: excess kurtosis > 0.
    """
    accel = imu_raw.get("accel", [])
    n = len(accel)
    if n < 30:
        return True, 0, {"reason": "SKIP_SHORT", "excess_kurtosis": None}

    all_innovations = []

    for axis in range(3):
        signal = [accel[i][axis] for i in range(n)]
        mean_s = sum(signal) / n
        demeaned = [x - mean_s for x in signal]
        var = sum(x * x for x in demeaned)
        if var < 1e-10:
            return False, 100, {"reason": "FLAT", "excess_kurtosis": 0.0}

        # AR(1) coefficient via lag-1 ACF
        phi = sum(demeaned[i] * demeaned[i+1] for i in range(n-1)) / var
        phi = max(-0.99, min(0.99, phi))

        # Innovations: residuals after AR(1) removal
        innovations = [demeaned[i] - phi * demeaned[i-1] for i in range(1, n)]
        all_innovations.extend(innovations)

    m = len(all_innovations)
    mean_i = sum(all_innovations) / m
    centered = [x - mean_i for x in all_innovations]
    var_i = sum(x*x for x in centered) / m

    if var_i < 1e-12:
        return False, 100, {"reason": "ZERO_VAR", "excess_kurtosis": 0.0}

    kurt_raw = (sum(x**4 for x in centered) / m) / (var_i**2)
    excess_kurtosis = kurt_raw - 3.0

    THRESHOLD = 0.63
    passes = excess_kurtosis > THRESHOLD
    confidence = min(100, int(abs(excess_kurtosis) * 30))

    return passes, confidence, {
        "excess_kurtosis": round(excess_kurtosis, 4),
        "threshold": THRESHOLD,
        "reason": "PASS" if passes else "GAUSSIAN_INNOVATIONS_DETECTED"
    }


def make_real_like(n=256, hz=100.0, seed=0):
    """
    Impulse-modulated signal — mimics muscle burst non-Gaussianity.
    Not OU. Has leptokurtic innovations from occasional burst events.
    """
    random.seed(seed)
    state = [0.0, 0.0, 9.81]
    accel = []
    for i in range(n):
        row = []
        for axis in range(3):
            base = 0.92 * state[axis] + random.gauss(0, 0.4)
            if random.random() < 0.05:   # 5% burst probability
                base += random.choice([-1, 1]) * random.uniform(1.5, 4.0)
            state[axis] = base
            row.append(base)
        row[2] += 9.81
        accel.append(row)
    return {"timestamps_ns": [int(i*1e9/hz) for i in range(n)], "accel": accel}


print(f"\n{W}{'═'*60}")
print(f"  Law 16 — Innovation Kurtosis Calibration Test")
print(f"  Coupled OU (Gaussian) vs Real-like (leptokurtic)")
print(f"{'═'*60}{X}\n")

# --- Test 1: Coupled OU windows ---
print(f"  Testing 30 coupled OU windows (theta=5, rho=0.55)...")
ou_kurtosis = []
ou_passed = 0
for seed in range(30):
    w = generate_window(n_samples=256, hz=100.0, seed=seed)
    passes, conf, detail = check_innovation_kurtosis(w)
    ek = detail.get("excess_kurtosis", 0) or 0
    ou_kurtosis.append(ek)
    if passes:
        ou_passed += 1

ou_mean = sum(ou_kurtosis) / len(ou_kurtosis)
ou_min  = min(ou_kurtosis)
ou_max  = max(ou_kurtosis)
c = G if ou_passed == 0 else R
print(f"  excess kurtosis: mean={ou_mean:.4f}  min={ou_min:.4f}  max={ou_max:.4f}")
print(f"  {c}OU caught (FAIL): {30 - ou_passed}/30{X}")

# --- Test 2: Real-like windows ---
print(f"\n  Testing 30 real-like (impulse-modulated) windows...")
real_kurtosis = []
real_passed = 0
for seed in range(30):
    w = make_real_like(n=256, hz=100.0, seed=seed)
    passes, conf, detail = check_innovation_kurtosis(w)
    ek = detail.get("excess_kurtosis", 0) or 0
    real_kurtosis.append(ek)
    if passes:
        real_passed += 1

real_mean = sum(real_kurtosis) / len(real_kurtosis)
real_min  = min(real_kurtosis)
real_max  = max(real_kurtosis)
c = G if real_passed >= 24 else Y
print(f"  excess kurtosis: mean={real_mean:.4f}  min={real_min:.4f}  max={real_max:.4f}")
print(f"  {c}Real passed: {real_passed}/30{X}")

# --- Summary ---
print(f"\n{W}{'═'*60}  RESULT{X}")
print(f"  OU mean excess kurtosis:   {ou_mean:.4f}  (theoretical = 0.0)")
print(f"  Real mean excess kurtosis: {real_mean:.4f}  (expected > 0.4)")
print(f"  Separation gap:            {real_mean - ou_mean:.4f}")

if ou_passed == 0 and real_passed >= 24:
    print(f"\n  {G}✓ THRESHOLD 0.3 WORKS — Law 16 ready to integrate{X}")
elif ou_passed == 0 and real_passed < 24:
    suggested = round(real_min * 0.7, 2)
    print(f"\n  {Y}⚠ OU caught but too many real false positives")
    print(f"    Lower threshold to {suggested} and rerun{X}")
elif ou_passed > 0:
    suggested = round(ou_max + 0.05, 2)
    print(f"\n  {R}✗ Some OU windows passed — lower threshold to {suggested}{X}")

print()
