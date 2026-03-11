"""
S2S Module 3 — Curriculum Generator
======================================
WHERE IT FITS IN S2S:
  experiments/module3_curriculum_generator.py

PURPOSE:
  Generate training data at every quality level automatically.
  S2S stops waiting to be fed certified data — it generates its own
  curriculum from seed windows using corruption + mixing.

ANALOGY:
  An adult who learned what's safe to eat (certified data) can now
  teach a child by showing examples at every difficulty level:
  "this is definitely safe, this is borderline, this is dangerous."
  The child (downstream ML) learns from the full spectrum, not just
  the easy cases.

RUN:
  cd ~/S2S && python3 experiments/module3_curriculum_generator.py

OUTPUT:
  experiments/curriculum_dataset.npy  (features)
  experiments/curriculum_labels.npy   (quality scores 0-100)
  experiments/curriculum_tiers.npy    (GOLD/SILVER/BRONZE/REJECTED)
  experiments/results_curriculum_stats.json
"""

import numpy as np
import json
import sys
import os
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

# Import corruption functions from Module 1
sys.path.insert(0, str(Path(__file__).parent))
try:
    from module1_corruption_fingerprinter import (
        corrupt_dropped_packets, corrupt_sensor_drift, corrupt_spike_injection,
        corrupt_gravity_shift, corrupt_clock_jitter, corrupt_synthetic_smoothing,
        CORRUPTION_TYPES
    )
except ImportError:
    # Inline fallback if module1 not found
    def corrupt_spike_injection(w, i): 
        c = w.copy()
        c[np.random.choice(len(c), max(1, int(len(c)*i*0.1)), replace=False)] += np.random.randn(max(1,int(len(c)*i*0.1)), 3) * i * 50
        return c
    def corrupt_sensor_drift(w, i):
        c = w.copy(); c[:,0] += np.linspace(0, i*5, len(c)); return c
    CORRUPTION_TYPES = {"spike": corrupt_spike_injection, "drift": corrupt_sensor_drift}


TIER_ORDER = {'GOLD': 3, 'SILVER': 2, 'BRONZE': 1, 'REJECTED': 0, 'ERROR': -1}


class CurriculumGenerator:
    def __init__(self, sample_rate_hz=50, window_size=256):
        self.engine = PhysicsEngine()
        self.sample_rate_hz = sample_rate_hz
        self.window_size = window_size

    def certify_window(self, accel, gyro):
        n = len(accel)
        timestamps = np.arange(n) * (1e9 / self.sample_rate_hz)
        imu = {
            "timestamps_ns": timestamps.tolist(),
            "accel": accel.tolist(),
            "gyro": gyro.tolist(),
            "sample_rate_hz": self.sample_rate_hz,
        }
        try:
            return self.engine.certify(imu, segment="forearm")
        except Exception:
            return {"tier": "ERROR", "laws_failed": [], "physical_law_score": 0}

    def extract_features(self, accel, gyro):
        """Extract 13 physics-inspired features from corrupted window"""
        # Handle NaN values in input
        accel = np.array(accel)
        accel = accel[~np.isnan(accel).any(axis=1)] if len(accel.shape) > 1 else accel[~np.isnan(accel)]
        gyro = np.array(gyro)
        gyro = gyro[~np.isnan(gyro).any(axis=1)] if len(gyro.shape) > 1 else gyro[~np.isnan(gyro)]
        
        if len(accel) == 0:  # Fallback if all NaN
            return np.zeros(13, dtype=np.float32)
        
        features = []
        
        # Basic statistics
        features.extend([
            np.mean(accel), np.std(accel), np.max(np.abs(accel)),
            np.mean(gyro), np.std(gyro), np.max(np.abs(gyro))
        ])
        
        # Signal quality metrics
        signal_energy = np.sum(accel**2)
        zero_crossings = np.sum(np.diff(np.sign(accel[:, 0])) != 0)
        features.extend([float(signal_energy), float(zero_crossings)])
        
        # Jerk (3rd derivative of position = 1st derivative of accel)
        if len(accel) > 3:
            jerk = np.diff(accel, n=1, axis=0) * self.sample_rate_hz
            jerk_clean = jerk[~np.isnan(jerk).any(axis=1)] if len(jerk.shape) > 1 else jerk[~np.isnan(jerk)]
            if len(jerk_clean) > 0:
                features.append(np.sqrt(np.mean(jerk_clean**2)))        # jerk RMS
                features.append(np.percentile(np.abs(jerk_clean), 95))  # jerk P95
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        # Frequency domain
        for axis in range(min(3, accel.shape[1])):
            axis_data = accel[:, axis]
            axis_clean = axis_data[~np.isnan(axis_data)]
            if len(axis_clean) > 0:
                fft = np.abs(np.fft.rfft(axis_clean))
                freqs = np.fft.rfftfreq(len(axis_clean), 1/self.sample_rate_hz)
                peak_freq = freqs[np.argmax(fft)] if len(fft) > 0 else 0
                features.append(float(peak_freq))
            else:
                features.append(0.0)
        
        # Cross-axis coupling (rigid body)
        if accel.shape[1] >= 2:
            try:
                coupling = np.corrcoef(accel[:, 0], accel[:, 1])[0, 1]
                features.append(float(coupling) if not np.isnan(coupling) else 0.0)
            except:
                features.append(0.0)
        else:
            features.append(0.0)
        
        # Gravity component
        mean_accel = np.mean(accel, axis=0)
        gravity_magnitude = np.linalg.norm(mean_accel)
        features.append(float(gravity_magnitude) if not np.isnan(gravity_magnitude) else 0.0)
        
        # Entropy (complexity measure)
        accel_flat = accel.flatten()
        accel_flat = accel_flat[~np.isnan(accel_flat)]  # Remove NaN values
        if len(accel_flat) == 0:  # Fallback if all NaN
            entropy = 0.0
        else:
            hist, _ = np.histogram(accel_flat, bins=20)
            hist = hist / (hist.sum() + 1e-10)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
        features.append(float(entropy))
        
        return np.array(features, dtype=np.float32)

    def generate_from_seed(self, seed_accel, seed_gyro, sample_rate=50, n_samples=100):
        """
        From one clean seed window, generate n_samples at varying quality levels.
        Uses corruption at random types and intensities.
        """
        samples = []

        for _ in range(n_samples):
            # Pick random corruption strategy
            strategy = random.choice(list(CORRUPTION_TYPES.keys()))
            # Use higher intensity range for more challenging curriculum
            intensity = random.uniform(0.7, 1.0)  # Final adjustment for target distribution
            corrupt_fn = CORRUPTION_TYPES[strategy]

            # Apply corruption
            corrupted_accel = corrupt_fn(seed_accel.copy(), intensity)
            cert = self.certify_window(corrupted_accel, seed_gyro)

            features = self.extract_features(corrupted_accel, seed_gyro)
            samples.append({
                "features": features,
                "tier": cert['tier'],
                "score": cert.get('physical_law_score', 0),
                "laws_failed": cert.get('laws_failed', []),
                "corruption_type": strategy,
                "intensity": intensity,
            })

        return samples

    def generate_curriculum(self, seed_windows, sample_rates=None, n_per_seed=50, target_distribution=None):
        """
        Generate full curriculum from seed windows.

        target_distribution: desired tier ratios e.g.
          {'GOLD': 0.1, 'SILVER': 0.3, 'BRONZE': 0.4, 'REJECTED': 0.2}
        If None, takes natural distribution from corruption.

        Returns arrays ready for ML training.
        """
        if target_distribution is None:
            target_distribution = {
                'GOLD': 0.10, 'SILVER': 0.30, 'BRONZE': 0.40, 'REJECTED': 0.20
            }

        all_samples = []
        print(f"Generating curriculum from {len(seed_windows)} seed windows...")
        print(f"Target: {n_per_seed} samples per seed = {len(seed_windows)*n_per_seed} total")

        for i, (seed_accel, seed_gyro) in enumerate(seed_windows):
            sample_rate = sample_rates[i] if sample_rates and i < len(sample_rates) else 50
            samples = self.generate_from_seed(seed_accel, seed_gyro, sample_rate, n_per_seed)
            all_samples.extend(samples)
            if (i + 1) % 10 == 0:
                print(f"  Seed {i+1}/{len(seed_windows)} processed")

        # Convert to arrays
        X = np.array([s['features'] for s in all_samples])
        y_score = np.array([s['score'] for s in all_samples])
        y_tier = np.array([s['tier'] for s in all_samples])

        # Tier distribution
        unique, counts = np.unique(y_tier, return_counts=True)
        dist = dict(zip(unique, counts))

        print(f"\nGenerated {len(all_samples)} curriculum samples")
        print("Tier distribution:")
        for tier in ['GOLD', 'SILVER', 'BRONZE', 'REJECTED', 'ERROR']:
            count = dist.get(tier, 0)
            pct = count / len(all_samples) * 100
            print(f"  {tier:8s}: {count:5d} ({pct:.1f}%)")

        return X, y_score, y_tier, all_samples

    def build_difficulty_stages(self, X, y_score, n_stages=5):
        """
        Sort curriculum into difficulty stages for progressive training.
        Stage 1: Only clear GOLD vs REJECTED (easy)
        Stage 5: Borderline cases near tier boundaries (hard)

        This is curriculum learning — start simple, increase difficulty.
        Like a child learning with easy examples first.
        """
        stages = []
        stage_size = len(X) // n_stages

        # Sort by score — easy cases at extremes, hard at middle
        score_range = np.max(y_score) - np.min(y_score)
        difficulty = np.abs(y_score - (np.min(y_score) + score_range / 2))  # Distance from middle
        easy_to_hard = np.argsort(-difficulty)  # Most extreme (easy) first

        for stage in range(n_stages):
            start = stage * stage_size
            end = start + stage_size * (stage + 1)
            indices = easy_to_hard[start:end]
            stages.append({
                "stage": stage + 1,
                "n_samples": len(indices),
                "avg_score": float(np.mean(y_score[indices])),
                "description": f"Stage {stage+1}: {'easy extremes' if stage == 0 else 'harder borderlines'}"
            })

        return stages


def make_seed_windows_from_data(data_path, n_seeds=100, window_size=256):
    """Load seed windows from existing certified data with auto-discovery of local datasets"""
    
    # Define potential dataset paths for auto-discovery
    dataset_paths = [
        "~/ninapro_db5/",           # NinaPro DB5
        "~/S2S_Project/EMG_Amputee/",  # EMG Amputee
        "~/S2S_Project/HuGaDB/",        # HuGaDB
        "data/training_X_raw.npy",       # PTT-PPG
    ]
    
    all_seeds = []
    sample_rates = []
    
    # Try each dataset in order of preference
    for path in dataset_paths:
        expanded_path = os.path.expanduser(path)
        
        if os.path.exists(expanded_path):
            if path.endswith(".npy"):  # PTT-PPG .npy file
                print(f"Auto-discovered PTT-PPG data: {expanded_path}")
                X = np.load(expanded_path)
                seeds = []
                for i in range(min(n_seeds//2, len(X))):  # Take half from PTT-PPG
                    accel = X[i]  # (256, 3)
                    gyro = accel * 0.1 + np.random.randn(*accel.shape) * 0.01
                    seeds.append((accel, gyro))
                    sample_rates.append(500)  # PTT-PPG is 500Hz
                all_seeds.extend(seeds)
                
            elif os.path.isdir(expanded_path):  # Directory-based datasets
                # Look for subject directories
                subjects = [d for d in os.listdir(expanded_path) 
                           if os.path.isdir(os.path.join(expanded_path, d))]
                if subjects:
                    print(f"Auto-discovered dataset: {expanded_path} with subjects: {subjects[:5]}...")
                    
                    # Prioritize low-rejection NinaPro subjects (s4, s5)
                    priority_subjects = []
                    other_subjects = []
                    
                    for subject in subjects:
                        if subject in ['s4', 's5']:  # Low rejection subjects
                            priority_subjects.append(subject)
                        else:
                            other_subjects.append(subject)
                    
                    # Use priority subjects first
                    selected_subjects = priority_subjects + other_subjects[:3]
                    
                    # Try to load from selected subjects
                    seeds_loaded = 0
                    sessions_skipped = 0
                    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine as _PE
                    for subject in selected_subjects:
                        if seeds_loaded >= n_seeds//2:  # Take half from NinaPro
                            break
                        subject_path = os.path.join(expanded_path, subject)
                        try:
                            # Look for common data files
                            import glob
                            mat_files = glob.glob(os.path.join(subject_path, "*.mat"))
                            if mat_files:
                                # Load first .mat file
                                import scipy.io
                                mat = scipy.io.loadmat(mat_files[0])
                                
                                # Find accelerometer data
                                for key in mat.keys():
                                    if not key.startswith('_'):
                                        data = mat[key]
                                        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 3:
                                            accel = data[:, :3]
                                            
                                            # --- BFS BIOLOGICAL ORIGIN FILTER ---
                                            # Use 2000-sample windows for BFS check (proven at 2000Hz)
                                            # Seed windows stay at 256 — BFS check is separate
                                            _pe = _PE()
                                            _hz = 2000
                                            _dt_ns = int(1e9 / _hz)
                                            _bfs_window = 2000  # 1s at 2000Hz — satisfies resonance minimum
                                            _bfs_step = 1000
                                            _n_check = (len(accel) - _bfs_window) // _bfs_step  # use all windows — Hurst needs 100+ for stable estimate
                                            for _i in range(_n_check):
                                                _start = _i * _bfs_step
                                                _w = accel[_start:_start+_bfs_window, :3]
                                                _ts = [_j * _dt_ns for _j in range(_bfs_window)]
                                                _pe.certify(imu_raw={"timestamps_ns": _ts, "accel": _w.tolist()}, segment="forearm")
                                            _session = _pe.certify_session()
                                            _grade = _session.get("biological_grade")
                                            if _grade == "NOT_BIOLOGICAL":
                                                sessions_skipped += 1
                                                print(f"  SKIPPED {subject}: NOT_BIOLOGICAL (H={_session.get('hurst')})")
                                                break
                                            else:
                                                print(f"  ACCEPTED {subject}: {_grade} (H={_session.get('hurst')}, BFS={_session.get('bfs')})")
                                            # --- END BFS FILTER ---

                                            # Extract windows
                                            max_windows = min(10, (n_seeds//2 - seeds_loaded))
                                            for i in range(max_windows):
                                                if i * window_size + window_size <= len(accel):
                                                    window_accel = accel[i*window_size:(i+1)*window_size, :3]
                                                    # Create synthetic gyro
                                                    window_gyro = window_accel * 0.1 + np.random.randn(*window_accel.shape) * 0.01
                                                    all_seeds.append((window_accel, window_gyro))
                                                    sample_rates.append(2000)  # NinaPro is 2000Hz
                                                    seeds_loaded += 1
                                                    break
                                            break
                        except Exception as e:
                            print(f"  Could not load {subject}: {e}")
                            continue
                    print(f"  BFS filter: {sessions_skipped} sessions skipped as NOT_BIOLOGICAL, {seeds_loaded} sessions accepted")
                    
                    if all_seeds:
                        print(f"  Loaded {seeds_loaded} seed windows from {expanded_path}")
    
    if all_seeds:
        print(f"Total seeds loaded: {len(all_seeds)} (PTT-PPG: {sample_rates.count(500)}, NinaPro: {sample_rates.count(2000)})")
        return all_seeds, sample_rates
    
    print("No local datasets found with auto-discovery")
    return None, None


def make_synthetic_seeds(n_seeds=50, window_size=256, sample_rate=50):
    """Generate synthetic clean seeds when real data unavailable"""
    seeds = []
    t = np.linspace(0, window_size/sample_rate, window_size)
    for _ in range(n_seeds):
        freq = random.uniform(0.3, 3.0)
        accel = np.column_stack([
            9.8 * np.sin(2*np.pi*freq*t) + np.random.randn(window_size)*0.1,
            np.random.randn(window_size)*0.3,
            np.random.randn(window_size)*0.3,
        ])
        gyro = np.column_stack([
            np.sin(2*np.pi*freq*t)*0.5 + np.random.randn(window_size)*0.01,
            np.random.randn(window_size)*0.05,
            np.random.randn(window_size)*0.05,
        ])
        seeds.append((accel, gyro))
    return seeds


def main():
    print("=" * 60)
    print("S2S Module 3 — Curriculum Generator")
    print("=" * 60)

    generator = CurriculumGenerator(sample_rate_hz=50)

    # Load seed windows
    seeds, sample_rates = make_seed_windows_from_data("data/training_X_raw.npy", n_seeds=100)
    if seeds is None:
        print("No real data found — using synthetic seeds")
        seeds = make_synthetic_seeds(n_seeds=50)
        sample_rates = [50] * len(seeds)
    else:
        print(f"Loaded {len(seeds)} real seed windows from mixed datasets")

    # Generate curriculum
    X, y_score, y_tier, raw = generator.generate_curriculum(seeds, sample_rates, n_per_seed=20)

    # Build difficulty stages
    stages = generator.build_difficulty_stages(X, y_score, n_stages=5)
    print("\nDifficulty stages for progressive training:")
    for s in stages:
        print(f"  {s['description']} — {s['n_samples']} samples, avg score {s['avg_score']:.1f}")

    # Save curriculum
    os.makedirs("experiments", exist_ok=True)
    np.save("experiments/curriculum_dataset.npy", X)
    np.save("experiments/curriculum_labels.npy", y_score)
    np.save("experiments/curriculum_tiers.npy", y_tier)

    stats = {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "score_mean": float(np.mean(y_score)),
        "score_std": float(np.std(y_score)),
        "tier_distribution": {t: int(np.sum(y_tier == t)) for t in ['GOLD','SILVER','BRONZE','REJECTED']},
        "difficulty_stages": stages,
        "ready_for_cloud_training": True,
        "cloud_training_file": "module4_cloud_trainer.py",
    }
    with open("experiments/results_curriculum_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nCurriculum saved:")
    print(f"  experiments/curriculum_dataset.npy  ({X.shape})")
    print(f"  experiments/curriculum_labels.npy")
    print(f"  experiments/curriculum_tiers.npy")
    print(f"\nNext: run module4_cloud_trainer.py on Google Colab with these files")


if __name__ == "__main__":
    main()
