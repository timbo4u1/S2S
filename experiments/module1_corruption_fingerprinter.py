"""
S2S Module 1 — Corruption Fingerprinter
========================================
WHERE IT FITS IN S2S:
  experiments/module1_corruption_fingerprinter.py

PURPOSE:
  Take certified GOLD/SILVER windows. Corrupt them systematically.
  Map which corruption type breaks which physics law at which intensity.
  Output: a corruption fingerprint library — S2S learning what bad data
  looks like from the inside, not just from what it was told.

ANALOGY:
  A child touches fire once and learns "hot = pain".
  This module touches every type of data corruption and learns
  "drift at 25% = rigid_body fails first".

RUN:
  cd ~/S2S && python3 experiments/module1_corruption_fingerprinter.py

OUTPUT:
  experiments/results_corruption_fingerprints.json
"""

import numpy as np
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine


# ─── CORRUPTION FUNCTIONS ────────────────────────────────────────────────────

def corrupt_dropped_packets(window, intensity):
    """Drop random samples — simulates sensor packet loss"""
    corrupted = window.copy()
    n = len(corrupted)
    n_drop = max(1, int(n * intensity * 0.3))
    drop_idx = np.random.choice(n, n_drop, replace=False)
    # Replace dropped samples with last valid value (common in real sensors)
    for idx in sorted(drop_idx):
        if idx > 0:
            corrupted[idx] = corrupted[idx - 1]
    return corrupted


def corrupt_sensor_drift(window, intensity):
    """Add linear drift — simulates sensor temperature drift"""
    corrupted = window.copy()
    drift = np.linspace(0, intensity * 5.0, len(corrupted))
    corrupted[:, 0] += drift  # drift on X axis
    return corrupted


def corrupt_spike_injection(window, intensity):
    """Inject random spikes — simulates electrical interference"""
    corrupted = window.copy()
    n_spikes = max(1, int(len(corrupted) * intensity * 0.1))
    spike_idx = np.random.choice(len(corrupted), n_spikes, replace=False)
    spike_magnitude = intensity * 50.0  # up to 50 m/s² spikes
    corrupted[spike_idx] += np.random.randn(n_spikes, 3) * spike_magnitude
    return corrupted


def corrupt_gravity_shift(window, intensity):
    """Add constant bias — simulates gravity misalignment"""
    corrupted = window.copy()
    bias = np.array([intensity * 9.8, 0, 0])  # shift gravity to wrong axis
    corrupted += bias
    return corrupted


def corrupt_clock_jitter(window, intensity):
    """Add timing noise — simulates unstable sampling clock"""
    corrupted = window.copy()
    # Randomly repeat or skip samples to simulate jitter
    jitter_samples = max(1, int(len(corrupted) * intensity * 0.05))
    for _ in range(jitter_samples):
        idx = np.random.randint(1, len(corrupted) - 1)
        # Either duplicate or interpolate
        if np.random.random() > 0.5:
            corrupted[idx] = corrupted[idx - 1]  # duplicate
        else:
            corrupted[idx] = (corrupted[idx - 1] + corrupted[idx + 1]) / 2  # interpolate
    return corrupted


def corrupt_synthetic_smoothing(window, intensity):
    """Over-smooth — simulates synthetic data generation artifact"""
    corrupted = window.copy()
    kernel_size = max(3, int(intensity * 20))
    kernel = np.ones(kernel_size) / kernel_size
    for axis in range(3):
        corrupted[:, axis] = np.convolve(corrupted[:, axis], kernel, mode='same')
    return corrupted


CORRUPTION_TYPES = {
    "dropped_packets": corrupt_dropped_packets,
    "sensor_drift": corrupt_sensor_drift,
    "spike_injection": corrupt_spike_injection,
    "gravity_shift": corrupt_gravity_shift,
    "clock_jitter": corrupt_clock_jitter,
    "synthetic_smoothing": corrupt_synthetic_smoothing,
}

INTENSITIES = [0.1, 0.25, 0.5, 0.75, 1.0]


# ─── FINGERPRINTER ───────────────────────────────────────────────────────────

class CorruptionFingerprinter:
    def __init__(self, sample_rate_hz=50):
        self.engine = PhysicsEngine()
        self.sample_rate_hz = sample_rate_hz

    def certify_window(self, accel_window, gyro_window):
        """Run physics certification on a single window"""
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
        except Exception as e:
            return {"tier": "ERROR", "laws_failed": [], "physical_law_score": 0}

    def fingerprint_window(self, accel_window, gyro_window):
        """
        For one clean window: apply every corruption at every intensity.
        Returns dict mapping (corruption, intensity) -> laws_failed
        """
        baseline = self.certify_window(accel_window, gyro_window)
        if baseline['tier'] not in ['GOLD', 'SILVER']:
            return None  # Skip — window wasn't clean to begin with

        results = {
            "baseline_tier": baseline['tier'],
            "baseline_score": baseline.get('physical_law_score', 0),
            "corruptions": {}
        }

        for corruption_name, corrupt_fn in CORRUPTION_TYPES.items():
            results["corruptions"][corruption_name] = {}
            for intensity in INTENSITIES:
                try:
                    corrupted_accel = corrupt_fn(accel_window.copy(), intensity)
                    cert = self.certify_window(corrupted_accel, gyro_window)
                    results["corruptions"][corruption_name][str(intensity)] = {
                        "tier": cert['tier'],
                        "score": cert.get('physical_law_score', 0),
                        "laws_failed": cert.get('laws_failed', []),
                        "tier_degraded": cert['tier'] != baseline['tier'],
                    }
                except Exception as e:
                    results["corruptions"][corruption_name][str(intensity)] = {
                        "tier": "ERROR", "score": 0, "laws_failed": [], "error": str(e)
                    }

        return results

    def fingerprint_dataset(self, accel_data, gyro_data, n_windows=100, window_size=256):
        """
        Fingerprint multiple windows from a dataset.
        accel_data: (N, 3) array
        gyro_data: (N, 3) array
        """
        n_samples = len(accel_data)
        results = []
        fingerprint_summary = {name: {str(i): {"tier_changes": 0, "law_failures": {}}
                                       for i in INTENSITIES}
                               for name in CORRUPTION_TYPES}

        windows_processed = 0
        windows_skipped = 0

        for start in range(0, min(n_samples - window_size, n_windows * window_size), window_size):
            accel_w = accel_data[start:start + window_size]
            gyro_w = gyro_data[start:start + window_size]

            fp = self.fingerprint_window(accel_w, gyro_w)
            if fp is None:
                windows_skipped += 1
                continue

            results.append(fp)
            windows_processed += 1

            # Accumulate summary stats
            for corruption_name, intensities in fp["corruptions"].items():
                for intensity_str, outcome in intensities.items():
                    if outcome["tier_degraded"]:
                        fingerprint_summary[corruption_name][intensity_str]["tier_changes"] += 1
                    for law in outcome["laws_failed"]:
                        law_counts = fingerprint_summary[corruption_name][intensity_str]["law_failures"]
                        law_counts[law] = law_counts.get(law, 0) + 1

            if windows_processed >= n_windows:
                break

        # Compute which law breaks first for each corruption type
        first_break = {}
        for corruption_name in CORRUPTION_TYPES:
            for intensity in INTENSITIES:
                key = str(intensity)
                law_failures = fingerprint_summary[corruption_name][key]["law_failures"]
                if law_failures:
                    first_law = max(law_failures, key=law_failures.get)
                    first_break[f"{corruption_name}@{intensity}"] = first_law

        return {
            "windows_processed": windows_processed,
            "windows_skipped": windows_skipped,
            "fingerprint_summary": fingerprint_summary,
            "first_law_to_break": first_break,
            "raw_results": results[:20],  # Save first 20 for inspection
        }


# ─── MAIN ────────────────────────────────────────────────────────────────────

def load_ptt_ppg_sample(data_dir="data"):
    """Load a sample of PTT-PPG data for fingerprinting"""
    accel_path = os.path.join(data_dir, "training_X_raw.npy")
    if os.path.exists(accel_path):
        X = np.load(accel_path)  # (N, 256, 3)
        print(f"Loaded PTT-PPG data: {X.shape}")
        # Use first channel as accel, create synthetic gyro
        accel = X[:500].reshape(-1, 3)
        gyro = accel * 0.1 + np.random.randn(*accel.shape) * 0.01
        return accel, gyro
    return None, None


def main():
    print("=" * 60)
    print("S2S Module 1 — Corruption Fingerprinter")
    print("=" * 60)

    fingerprinter = CorruptionFingerprinter(sample_rate_hz=50)

    # Try to load real data
    accel, gyro = load_ptt_ppg_sample()

    if accel is None:
        print("No data found — generating synthetic clean data for demo")
        # Synthetic clean forearm motion: 50Hz, 256 samples
        n = 10000
        t = np.linspace(0, n / 50, n)
        accel = np.column_stack([
            9.8 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(n) * 0.1,
            np.random.randn(n) * 0.5,
            np.random.randn(n) * 0.5,
        ])
        gyro = np.column_stack([
            np.random.randn(n) * 0.05,
            np.random.randn(n) * 0.05,
            np.random.randn(n) * 0.05,
        ])
        print(f"Generated synthetic data: {accel.shape}")

    print(f"\nFingerprinting {min(50, len(accel)//256)} windows...")
    print("This tests 6 corruption types × 5 intensities per window\n")

    results = fingerprinter.fingerprint_dataset(accel, gyro, n_windows=50)

    print(f"Windows processed: {results['windows_processed']}")
    print(f"Windows skipped (not clean): {results['windows_skipped']}")

    print("\n--- First Physics Law to Break per Corruption ---")
    for key, law in sorted(results['first_law_to_break'].items()):
        print(f"  {key:40s} → {law}")

    print("\n--- Tier Degradation Rate (% windows that drop tier) ---")
    for corruption_name in CORRUPTION_TYPES:
        row = []
        for intensity in INTENSITIES:
            changes = results['fingerprint_summary'][corruption_name][str(intensity)]['tier_changes']
            total = max(results['windows_processed'], 1)
            row.append(f"{intensity}: {changes/total*100:.0f}%")
        print(f"  {corruption_name:25s} | {' | '.join(row)}")

    # Save results
    out_path = "experiments/results_corruption_fingerprints.json"
    os.makedirs("experiments", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
