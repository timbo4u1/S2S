"""
S2S Adapter — Transradial Amputee EMG + IMU Dataset
Kaggle: alihaltimemy/transradial-amputee-emg-myo-armband-dataset

Dataset structure:
  EMG_Amputee/
    AMP1/AMP1/
      mov 1/
        accelerometer-<timestamp>.csv  — timestamp,x,y,z (m/s²)
        emg-<timestamp>.csv            — timestamp,emg_0..emg_7
      mov 2/ ...
    AMP2/AMP2/ ...

Usage:
    python3 amputee_adapter.py \
        --input ~/S2S_Project/EMG_Amputee/ \
        --out   ~/S2S_Project/s2s_dataset/ \
        --window 200

Output: certified .json records in s2s_dataset/PRECISION/ (hand gestures)
"""

import os, sys, json, math, csv, time, argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
try:
    from review_queue import ReviewQueue
    _review_queue = ReviewQueue()
except ImportError:
    _review_queue = None

# ── CONFIG ────────────────────────────────────────────────────────────────────

SAMPLE_RATE_HZ   = 200      # Myo armband accelerometer rate
WINDOW_SAMPLES   = 200      # 1 second windows
OVERLAP          = 0        # no overlap (clean windows)
DOMAIN           = "PRECISION"  # hand/finger movements
DATASET_SOURCE   = "amputee_myo_armband"

# Movement labels from Myo armband protocol
MOV_LABELS = {
    1: "wrist_flexion", 2: "wrist_extension", 3: "wrist_pronation",
    4: "wrist_supination", 5: "hand_open", 6: "hand_close",
    7: "pointer", 8: "pinch_index", 9: "pinch_middle",
    10: "pinch_ring", 11: "pinch_little", 12: "thumb_up",
    13: "lateral_grasp", 14: "tripod_grasp", 15: "power_grasp",
    16: "rest", 17: "elbow_flexion",
}

# ── CSV READERS ───────────────────────────────────────────────────────────────

def read_accel_csv(path):
    """Returns list of (timestamp_ns, x, y, z)."""
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append((
                        int(float(row['timestamp'])),
                        float(row['x']),
                        float(row['y']),
                        float(row['z']),
                    ))
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        pass
    return rows

def read_gyro_csv(path):
    """Returns list of (timestamp_ns, x, y, z) in rad/s."""
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append((
                        int(float(row['timestamp'])),
                        float(row.get('x', row.get('gyro_x', 0))) * 0.01745,
                        float(row.get('y', row.get('gyro_y', 0))) * 0.01745,
                        float(row.get('z', row.get('gyro_z', 0))) * 0.01745,
                    ))
                except (ValueError, KeyError):
                    continue
    except Exception:
        pass
    return rows

def read_emg_csv(path):
    """Returns list of (timestamp_ns, [emg_0..emg_7])."""
    rows = []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts = int(float(row.get('timestamp', 0)))
                    emg = []
                    for i in range(8):
                        key = f'emg{i+1}'
                        if key in row:
                            emg.append(float(row[key]))
                    if len(emg) == 8:
                        rows.append((ts, emg))
                except (ValueError, KeyError):
                    continue
    except Exception:
        pass
    return rows

# ── SYNC EMG + ACCEL ──────────────────────────────────────────────────────────

def sync_sensors(accel_rows, emg_rows, window_size):
    """
    Align EMG and accelerometer by timestamp.
    Returns list of (timestamps_ns, accel_xyz, emg_envelope) windows.
    Accel and EMG have different sample rates — interpolate EMG to accel timestamps.
    """
    if len(accel_rows) < window_size or len(emg_rows) < 10:
        return []

    # Build EMG lookup: timestamp → envelope (RMS of 8 channels)
    emg_by_ts = {}
    for ts, channels in emg_rows:
        rms = math.sqrt(sum(c**2 for c in channels) / len(channels))
        emg_by_ts[ts] = rms

    emg_timestamps = sorted(emg_by_ts.keys())

    def nearest_emg(ts):
        """Find nearest EMG sample to accel timestamp."""
        if not emg_timestamps:
            return 0.0
        lo, hi = 0, len(emg_timestamps) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if emg_timestamps[mid] < ts:
                lo = mid + 1
            else:
                hi = mid
        return emg_by_ts[emg_timestamps[lo]]

    windows = []
    i = 0
    while i + window_size <= len(accel_rows):
        chunk = accel_rows[i:i + window_size]
        timestamps = [r[0] for r in chunk]
        accel = [[r[1], r[2], r[3]] for r in chunk]
        emg_env = [nearest_emg(ts) for ts in timestamps]

        windows.append({
            'timestamps_ns': timestamps,
            'accel': accel,
            'emg_envelope': emg_env,
        })
        i += window_size  # non-overlapping

    return windows

# ── MAIN ADAPTER ──────────────────────────────────────────────────────────────

def process_dataset(input_dir, output_dir, window_size=200):
    engine = PhysicsEngine()

    stats = {
        'total_windows': 0,
        'certified': 0,
        'rejected': 0,
        'by_tier': defaultdict(int),
        'by_subject': defaultdict(int),
        'by_movement': defaultdict(int),
        'errors': 0,
    }

    print(f"S2S Amputee Adapter")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Window: {window_size} samples ({window_size/SAMPLE_RATE_HZ:.1f}s)")
    print("="*50)

    for subject_dir in sorted(os.listdir(input_dir)):
        subject_path = os.path.join(input_dir, subject_dir)
        if not os.path.isdir(subject_path):
            continue

        # Handle nested structure: AMP1/AMP1/
        inner = os.path.join(subject_path, subject_dir)
        if os.path.isdir(inner):
            subject_path = inner

        subject_id = subject_dir  # AMP1, AMP2, ...
        print(f"\nSubject: {subject_id}")

        for mov_dir in sorted(os.listdir(subject_path)):
            mov_path = os.path.join(subject_path, mov_dir)
            if not os.path.isdir(mov_path):
                continue

            # Parse movement number
            try:
                mov_num = int(mov_dir.replace('mov', '').strip())
            except ValueError:
                continue

            action = MOV_LABELS.get(mov_num, f"movement_{mov_num}")

            # Find accel and EMG files
            accel_files = sorted([f for f in os.listdir(mov_path)
                                   if f.startswith('accelerometer') and f.endswith('.csv')])
            gyro_files  = sorted([f for f in os.listdir(mov_path)
                                   if f.startswith('gyro') and f.endswith('.csv')])
            emg_files   = sorted([f for f in os.listdir(mov_path)
                                   if f.startswith('emg') and f.endswith('.csv')])

            if not accel_files or not emg_files:
                continue

            # Load all accel and EMG for this movement
            all_accel = []
            for af in accel_files:
                all_accel.extend(read_accel_csv(os.path.join(mov_path, af)))

            all_gyro = []
            for gf in gyro_files:
                all_gyro.extend(read_accel_csv(os.path.join(mov_path, gf)))
            all_gyro.sort(key=lambda r: r[0])

            all_emg = []
            for ef in emg_files:
                all_emg.extend(read_emg_csv(os.path.join(mov_path, ef)))

            if len(all_accel) < window_size:
                continue

            # Sort by timestamp
            all_accel.sort(key=lambda r: r[0])
            all_emg.sort(key=lambda r: r[0])

            # Create windows
            windows = sync_sensors(all_accel, all_emg, window_size)

            for w_idx, window in enumerate(windows):
                stats['total_windows'] += 1

                # Build imu_raw for PhysicsEngine
                # Generate synthetic gyro (not in dataset) — zeros
                # This limits rigid_body and imu_coupling laws but allows:
                # jerk_bounds, resonance_frequency, BCG, Newton F=ma (with EMG)
                n = len(window['timestamps_ns'])
                # Sync gyro to accel timestamps
                gyro_by_ts = {r[0]: [r[1],r[2],r[3]] for r in all_gyro}
                gyro_ts = sorted(gyro_by_ts.keys())
                def nearest_gyro(ts):
                    if not gyro_ts: return [0.0,0.0,0.0]
                    lo,hi = 0,len(gyro_ts)-1
                    while lo<hi:
                        mid=(lo+hi)//2
                        if gyro_ts[mid]<ts: lo=mid+1
                        else: hi=mid
                    return gyro_by_ts[gyro_ts[lo]]
                gyro_win = [nearest_gyro(ts) for ts in window['timestamps_ns']]
                imu_raw = {
                    'timestamps_ns': window['timestamps_ns'],
                    'accel': window['accel'],
                    'gyro': gyro_win,
                    'emg': window['emg_envelope'],
                }

                try:
                    result = engine.certify(imu_raw=imu_raw, segment="forearm")
                except Exception as e:
                    stats['errors'] += 1
                    continue

                tier  = result.get('physics_tier',  result.get('tier',  'REJECTED'))
                score = result.get('physics_score', result.get('physical_law_score', 0))

                stats['by_tier'][tier] += 1
                stats['by_subject'][subject_id] += 1
                stats['by_movement'][action] += 1

                if tier != 'REJECTED':
                    stats['certified'] += 1
                else:
                    stats['rejected'] += 1

                # Auto-queue BRONZE for human review
                if tier == 'BRONZE' and _review_queue is not None:
                    _review_queue.add({
                        'action': action,
                        'domain': DOMAIN,
                        'person_id': subject_id,
                        'dataset_source': DATASET_SOURCE,
                        'physics_tier': tier,
                        'physics_score': score,
                        'physics_laws_passed': result.get('laws_passed', []),
                        'physics_laws_failed': result.get('laws_failed', []),
                        'jerk_p95_ms3': result.get('jerk_p95_ms3', 0),
                        'imu_coupling_r': result.get('imu_coupling_r', 0),
                        'n_samples': window.get('n_samples', len(window.get('timestamps_ns', []))),
                        'duration_s': len(window.get('timestamps_ns', [])) / SAMPLE_RATE_HZ,
                    })

                # Build output record
                record = {
                    'schema': 's2s_v1.4_amputee',
                    'action': action,
                    'domain': DOMAIN,
                    'domain_description': 'Precision hand/finger movements — prosthetic training data',
                    'robot_use': 'myoelectric_prosthetic_hand',
                    'person_id': subject_id,
                    'dataset_source': DATASET_SOURCE,
                    'movement_id': mov_num,
                    'window_idx': w_idx,
                    'duration_s': round(n / SAMPLE_RATE_HZ, 3),
                    'n_samples': n,
                    'sample_rate_hz': SAMPLE_RATE_HZ,
                    'has_emg': True,
                    'has_gyro': True,
                    'physics_tier': tier,
                    'physics_score': score,
                    'physics_laws_passed': result.get('laws_passed', []),
                    'physics_laws_failed': result.get('laws_failed', []),
                    'jerk_p95_ms3': result.get('jerk_p95_ms3', 0),
                    'imu_coupling_r': result.get('imu_coupling_r', 0),
                    # Store raw data for future benchmark
                    'accel_raw': window['accel'],
                    'emg_envelope': window['emg_envelope'],
                    '_signed_at_ns': int(time.time() * 1e9),
                }

                # Save
                out_domain_dir = os.path.join(output_dir, DOMAIN, action)
                os.makedirs(out_domain_dir, exist_ok=True)
                fname = f"{action}_{subject_id}_w{w_idx:04d}.json"
                with open(os.path.join(out_domain_dir, fname), 'w') as f:
                    json.dump(record, f, separators=(',', ':'))

            print(f"  mov {mov_num:2d} {action:<20} "
                  f"accel={len(all_accel)} emg={len(all_emg)} "
                  f"windows={len(windows)}")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total windows:  {stats['total_windows']}")
    print(f"Certified:      {stats['certified']} "
          f"({100*stats['certified']/max(stats['total_windows'],1):.1f}%)")
    print(f"Rejected:       {stats['rejected']}")
    print(f"Errors:         {stats['errors']}")
    print(f"\nBy tier:")
    for tier in ['GOLD','SILVER','BRONZE','REJECTED']:
        n = stats['by_tier'][tier]
        if n: print(f"  {tier:<10} {n}")
    print(f"\nBy subject:")
    for subj, n in sorted(stats['by_subject'].items()):
        print(f"  {subj}: {n} windows")

    return stats


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="S2S Amputee Dataset Adapter")
    p.add_argument('--input',  required=True, help='Path to EMG_Amputee/')
    p.add_argument('--out',    required=True, help='Output s2s_dataset/ dir')
    p.add_argument('--window', type=int, default=200, help='Window size in samples')
    args = p.parse_args()
    process_dataset(args.input, args.out, args.window)