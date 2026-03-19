"""
S2S Module 2 — Frankenstein Mixer
===================================
WHERE IT FITS IN S2S:
  experiments/module2_frankenstein_mixer.py

PURPOSE:
  Mix certified GOLD/SILVER windows with REJECTED windows at every ratio.
  Find the exact physics boundary — the ratio where data crosses from
  valid to invalid. That boundary IS information.

ANALOGY:
  A child learns "too hot" by gradually touching warmer things.
  Not just "hot" or "cold" — they learn the exact threshold.
  This module finds the exact threshold for every physics law.

RUN:
  cd ~/S2S && python3 experiments/module2_frankenstein_mixer.py

OUTPUT:
  experiments/results_frankenstein_boundaries.json
"""

import numpy as np
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine


class FrankensteinMixer:
    def __init__(self, sample_rate_hz=50):
        self.engine = PhysicsEngine()
        self.sample_rate_hz = sample_rate_hz

    def certify_window(self, accel_window, gyro_window):
        n = len(accel_window)
        timestamps = np.arange(n) * (1e9 / self.sample_rate_hz)
        imu = {
            "timestamps_ns": timestamps.tolist(),
            "accel": accel_window.tolist(),
            "gyro": gyro_window.tolist(),
            "sample_rate_hz": self.sample_rate_hz,
        }
        try:
            return self.engine.certify(imu, segment="forearm")
        except Exception:
            return {"tier": "ERROR", "laws_failed": [], "physical_law_score": 0}

    def mix(self, good_accel, bad_accel, good_gyro, bad_gyro, ratio):
        """
        ratio=0.0 → pure good data
        ratio=1.0 → pure bad data
        ratio=0.5 → half and half (true Frankenstein)
        """
        mixed_accel = (1 - ratio) * good_accel + ratio * bad_accel
        mixed_gyro = (1 - ratio) * good_gyro + ratio * bad_gyro
        return mixed_accel, mixed_gyro

    def find_boundary(self, good_accel, bad_accel, good_gyro, bad_gyro, precision=0.01):
        """
        Binary search for the exact ratio where physics breaks.
        Returns: boundary ratio, laws that fail at boundary
        """
        lo, hi = 0.0, 1.0

        # Verify endpoints
        good_cert = self.certify_window(good_accel, good_gyro)
        bad_cert = self.certify_window(bad_accel, bad_gyro)

        if good_cert['tier'] not in ['GOLD', 'SILVER', 'BRONZE']:
            return None  # Good data isn't actually good
        if bad_cert['tier'] in ['GOLD', 'SILVER']:
            return None  # Bad data isn't actually bad

        # Binary search
        boundary_ratio = 1.0
        boundary_laws = []

        for _ in range(int(np.log2(1 / precision)) + 1):
            mid = (lo + hi) / 2
            mixed_a, mixed_g = self.mix(good_accel, bad_accel, good_gyro, bad_gyro, mid)
            cert = self.certify_window(mixed_a, mixed_g)

            if cert['tier'] in ['GOLD', 'SILVER', 'BRONZE']:
                lo = mid  # Still passing — push boundary higher
            else:
                hi = mid  # Failing — push boundary lower
                boundary_ratio = mid
                boundary_laws = cert.get('laws_failed', [])

        return {
            "boundary_ratio": boundary_ratio,
            "boundary_laws": boundary_laws,
            "good_tier": good_cert['tier'],
            "bad_tier": bad_cert['tier'],
            "good_score": good_cert.get('physical_law_score', 0),
            "bad_score": bad_cert.get('physical_law_score', 0),
            "interpretation": f"Data stays valid until {boundary_ratio:.1%} contamination"
        }

    def scan_mixing_curve(self, good_accel, bad_accel, good_gyro, bad_gyro, steps=20):
        """
        Scan the full mixing curve from 0% to 100% bad data.
        Returns score and tier at each ratio step.
        Reveals: which physics laws degrade gradually vs cliff-edge.
        """
        ratios = np.linspace(0, 1, steps)
        curve = []

        for ratio in ratios:
            mixed_a, mixed_g = self.mix(good_accel, bad_accel, good_gyro, bad_gyro, ratio)
            cert = self.certify_window(mixed_a, mixed_g)
            curve.append({
                "ratio": float(ratio),
                "tier": cert['tier'],
                "score": cert.get('physical_law_score', 0),
                "laws_failed": cert.get('laws_failed', []),
            })

        return curve

    def batch_find_boundaries(self, good_windows, bad_windows, n_pairs=50):
        """
        Find boundaries for multiple good/bad pairs.
        Reveals distribution of physics boundaries across the dataset.
        """
        results = []
        law_boundaries = {}  # law -> list of boundary ratios

        n_pairs = min(n_pairs, len(good_windows), len(bad_windows))
        print(f"Finding boundaries for {n_pairs} good/bad pairs...")

        for i in range(n_pairs):
            g_a, g_g = good_windows[i]
            b_a, b_g = bad_windows[i % len(bad_windows)]

            boundary = self.find_boundary(g_a, b_a, g_g, b_g)
            if boundary:
                results.append(boundary)
                for law in boundary['boundary_laws']:
                    if law not in law_boundaries:
                        law_boundaries[law] = []
                    law_boundaries[law].append(boundary['boundary_ratio'])

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{n_pairs} pairs")

        # Compute statistics per law
        law_stats = {}
        for law, ratios in law_boundaries.items():
            law_stats[law] = {
                "mean_boundary": float(np.mean(ratios)),
                "std_boundary": float(np.std(ratios)),
                "min_boundary": float(np.min(ratios)),
                "max_boundary": float(np.max(ratios)),
                "n_triggers": len(ratios),
                "interpretation": f"{law} breaks at {np.mean(ratios):.1%} ± {np.std(ratios):.1%} contamination"
            }

        return {
            "n_pairs_tested": len(results),
            "law_boundary_stats": law_stats,
            "individual_boundaries": results[:20],  # First 20 for inspection
        }


def generate_bad_window(n=256):
    """Generate a physically impossible window for testing"""
    t = np.linspace(0, n/50, n)
    # Violates jerk bounds — random high-frequency noise
    accel = np.random.randn(n, 3) * 50  # Way too noisy
    gyro = np.random.randn(n, 3) * 10
    return accel, gyro


def generate_good_window(n=256):
    """Generate a plausible clean forearm window for testing"""
    t = np.linspace(0, n/50, n)
    accel = np.column_stack([
        9.8 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n) * 0.1,
        np.random.randn(n) * 0.3,
        np.random.randn(n) * 0.3,
    ])
    gyro = np.column_stack([
        np.sin(2 * np.pi * 0.5 * t) * 0.5 + np.random.randn(n) * 0.01,
        np.random.randn(n) * 0.05,
        np.random.randn(n) * 0.05,
    ])
    return accel, gyro


def main():
    print("=" * 60)
    print("S2S Module 2 — Frankenstein Mixer")
    print("=" * 60)

    mixer = FrankensteinMixer(sample_rate_hz=50)

    # Generate test windows
    print("\nGenerating test windows...")
    good_windows = [generate_good_window() for _ in range(50)]
    bad_windows = [generate_bad_window() for _ in range(50)]

    # Try to load real data if available
    real_data_path = "data/training_X_raw.npy"
    if os.path.exists(real_data_path):
        print(f"Loading real data from {real_data_path}")
        X = np.load(real_data_path)
        # Use real windows as good, generate bad from corrupted versions
        for i in range(min(50, len(X))):
            w = X[i]  # (256, 3)
            good_windows[i] = (w, w * 0.1)
            bad_windows[i] = (w + np.random.randn(*w.shape) * 30, np.random.randn(256, 3) * 5)

    # Demo: scan one mixing curve
    print("\n--- Mixing Curve (0% to 100% bad data) ---")
    g_a, g_g = good_windows[0]
    b_a, b_g = bad_windows[0]
    curve = mixer.scan_mixing_curve(g_a, b_a, g_g, b_g, steps=11)

    print(f"{'Ratio':>8} {'Tier':>8} {'Score':>8} {'Laws Failed'}")
    print("-" * 60)
    for point in curve:
        laws = ", ".join(point['laws_failed']) if point['laws_failed'] else "none"
        print(f"{point['ratio']:>8.1%} {point['tier']:>8} {point['score']:>8.1f} {laws}")

    # Find boundary for one pair
    print("\n--- Finding Exact Physics Boundary ---")
    boundary = mixer.find_boundary(g_a, b_a, g_g, b_g)
    if boundary:
        print(f"Boundary ratio: {boundary['boundary_ratio']:.1%}")
        print(f"Laws that break: {boundary['boundary_laws']}")
        print(f"Interpretation: {boundary['interpretation']}")

    # Batch analysis
    print("\n--- Batch Boundary Analysis (50 pairs) ---")
    batch = mixer.batch_find_boundaries(good_windows, bad_windows, n_pairs=50)

    print(f"\nPairs tested: {batch['n_pairs_tested']}")
    print("\nPer-law boundary statistics:")
    for law, stats in batch['law_boundary_stats'].items():
        print(f"  {law}: {stats['interpretation']}")

    # Save results
    out_path = "experiments/results_frankenstein_boundaries.json"
    os.makedirs("experiments", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(batch, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
