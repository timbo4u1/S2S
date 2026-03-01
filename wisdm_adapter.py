#!/usr/bin/env python3
"""
WISDM Dataset Adapter for S2S
51 subjects, 18 activities, phone + watch accel+gyro at 20Hz
Adds PRECISION and SPORT domains to the certified dataset.

Usage:
    python3 wisdm_adapter.py --input wisdm-dataset/ --out s2s_dataset/
"""

import os, sys, json, math, hashlib, argparse, statistics
from pathlib import Path

# Activity code → (label, domain)
ACTIVITY_MAP = {
    "A": ("walking",         "LOCOMOTION"),
    "B": ("jogging",         "LOCOMOTION"),
    "C": ("stairs",          "LOCOMOTION"),
    "D": ("sitting",         "DAILY_LIVING"),
    "E": ("standing",        "DAILY_LIVING"),
    "F": ("typing",          "PRECISION"),
    "G": ("teeth_brushing",  "PRECISION"),
    "H": ("eating_soup",     "PRECISION"),
    "I": ("eating_chips",    "PRECISION"),
    "J": ("eating_pasta",    "PRECISION"),
    "K": ("drinking",        "PRECISION"),
    "L": ("eating_sandwich", "PRECISION"),
    "M": ("kicking",         "SPORT"),
    "O": ("catch",           "SPORT"),
    "P": ("dribbling",       "SPORT"),
    "Q": ("writing",         "PRECISION"),
    "R": ("clapping",        "SOCIAL"),
    "S": ("folding_clothes", "DAILY_LIVING"),
}

WINDOW_SIZE = 100   # 5 seconds at 20Hz
STEP_SIZE   = 50    # 50% overlap

def parse_raw_file(filepath):
    """Parse WISDM raw file → list of (subject, activity_code, ts_ns, ax, ay, az)"""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip().rstrip(';')
            if not line: continue
            parts = line.split(',')
            if len(parts) < 6: continue
            try:
                subj = parts[0].strip()
                act  = parts[1].strip()
                ts   = int(parts[2].strip())
                ax   = float(parts[3].strip())
                ay   = float(parts[4].strip())
                az   = float(parts[5].strip())
                rows.append((subj, act, ts, ax, ay, az))
            except (ValueError, IndexError):
                continue
    return rows

def simple_certify(accel_window, gyro_window, dt=0.05):
    """Lightweight physics check — no deps, matches S2S scoring logic"""
    n = len(accel_window)
    if n < 20:
        return 50, []

    score = 0
    passed = []

    # 1. Jerk bounds (windowed)
    jerks = []
    for axis in range(3):
        sig = [accel_window[i][axis] for i in range(n)]
        for i in range(1, n-1):
            j = abs((sig[i+1] - 2*sig[i] + sig[i-1]) / (dt**2))
            jerks.append(j)
    if jerks:
        jerks.sort()
        p95 = jerks[int(len(jerks)*0.95)]
        if p95 <= 500:
            score += 25
            passed.append("jerk_bounds")

    # 2. IMU consistency (accel-gyro coupling)
    if gyro_window and len(gyro_window) == n:
        accel_mag = [math.sqrt(sum(a[i]**2 for i in range(3))) for a in accel_window]
        gyro_mag  = [math.sqrt(sum(g[i]**2 for i in range(3))) for g in gyro_window]
        if len(set(accel_mag)) > 1 and len(set(gyro_mag)) > 1:
            mean_a = statistics.mean(accel_mag)
            mean_g = statistics.mean(gyro_mag)
            std_a  = statistics.stdev(accel_mag) or 1e-9
            std_g  = statistics.stdev(gyro_mag)  or 1e-9
            corr   = statistics.mean(
                [(a - mean_a) * (g - mean_g) for a, g in zip(accel_mag, gyro_mag)]
            ) / (std_a * std_g)
            if corr > 0.05:
                score += 25
                passed.append("imu_consistency")

    # 3. Rigid body plausibility (accel variance non-zero)
    vars_ = []
    for axis in range(3):
        sig = [accel_window[i][axis] for i in range(n)]
        if len(set(sig)) > 1:
            vars_.append(statistics.variance(sig))
    if vars_ and max(vars_) > 0.01:
        score += 25
        passed.append("rigid_body_kinematics")

    # 4. Sensor noise floor (not all zeros, not clipped)
    magnitudes = [math.sqrt(sum(a[i]**2 for i in range(3))) for a in accel_window]
    mean_mag = statistics.mean(magnitudes)
    if 0.5 < mean_mag < 50:
        score += 25
        passed.append("sensor_validity")

    tier = "GOLD" if score >= 90 else "SILVER" if score >= 60 else "BRONZE" if score >= 30 else "REJECTED"
    return score, passed, tier

def process_subject(subject_id, accel_data, gyro_data, device, out_dir):
    """Window accel+gyro data and certify each window"""
    # Group by activity
    by_activity = {}
    for (subj, act, ts, ax, ay, az) in accel_data:
        if act not in by_activity:
            by_activity[act] = []
        by_activity[act].append((ts, ax, ay, az))

    # Build gyro lookup by timestamp (nearest)
    gyro_by_ts = {}
    for (subj, act, ts, gx, gy, gz) in gyro_data:
        gyro_by_ts[ts] = (gx, gy, gz)

    records = []
    for act_code, rows in by_activity.items():
        if act_code not in ACTIVITY_MAP:
            continue
        label, domain = ACTIVITY_MAP[act_code]
        rows.sort(key=lambda x: x[0])

        for w_start in range(0, len(rows) - WINDOW_SIZE, STEP_SIZE):
            window = rows[w_start:w_start + WINDOW_SIZE]
            accel_w = [[ax, ay, az] for (ts, ax, ay, az) in window]
            ts_list  = [ts for (ts, ax, ay, az) in window]

            # Match gyro
            gyro_w = []
            for ts in ts_list:
                if ts in gyro_by_ts:
                    gyro_w.append(list(gyro_by_ts[ts]))

            dt = (ts_list[-1] - ts_list[0]) / (len(ts_list) - 1) * 1e-9 if len(ts_list) > 1 else 0.05

            score, laws, tier = simple_certify(accel_w, gyro_w if len(gyro_w) == WINDOW_SIZE else [], dt)

            if tier == "REJECTED":
                continue

            # Compute top-level features for classifier
            jerks = []
            for axis in range(3):
                sig = [accel_w[i][axis] for i in range(len(accel_w))]
                for i in range(1, len(sig)-1):
                    j = abs((sig[i+1] - 2*sig[i] + sig[i-1]) / (dt**2))
                    jerks.append(j)
            jerks.sort()
            jerk_p95 = jerks[int(len(jerks)*0.95)] if jerks else 100.0

            # IMU coupling r
            accel_mag = [math.sqrt(sum(accel_w[i][k]**2 for k in range(3))) for i in range(len(accel_w))]
            gyro_w_full = gyro_w if len(gyro_w) == len(accel_w) else []
            if gyro_w_full:
                gyro_mag = [math.sqrt(sum(gyro_w_full[i][k]**2 for k in range(3))) for i in range(len(gyro_w_full))]
                mean_a = statistics.mean(accel_mag); mean_g = statistics.mean(gyro_mag)
                std_a = statistics.stdev(accel_mag) or 1e-9; std_g = statistics.stdev(gyro_mag) or 1e-9
                coupling_r = statistics.mean([(a-mean_a)*(g-mean_g) for a,g in zip(accel_mag,gyro_mag)]) / (std_a*std_g)
            else:
                coupling_r = 0.0

            rec = {
                "format": "s2s_v1.3",
                "dataset": "WISDM_2019",
                "device": device,
                "subject_id": subject_id,
                "action": label,
                "domain": domain,
                "segment": "wrist",
                "sample_rate_hz": 20,
                "window_size": WINDOW_SIZE,
                "jerk_p95_ms3": round(jerk_p95, 2),
                "imu_coupling_r": round(coupling_r, 4),
                "physics_score": score,
                "imu_raw": {
                    "timestamps_ns": ts_list,
                    "accel": accel_w,
                    "gyro": gyro_w if len(gyro_w) == WINDOW_SIZE else []
                },
                "certification": {
                    "tier": tier,
                    "physical_law_score": score,
                    "laws_passed": laws,
                    "certifier": "S2S_v1.3_wisdm_adapter"
                }
            }

            # Fingerprint
            fingerprint = hashlib.sha256(
                json.dumps(rec["imu_raw"]["accel"], separators=(',', ':')).encode()
            ).hexdigest()[:16]
            rec["fingerprint"] = fingerprint
            records.append((domain, label, rec))

    return records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to wisdm-dataset/")
    parser.add_argument("--out",   required=True, help="Output s2s_dataset/ directory")
    parser.add_argument("--device", default="phone", choices=["phone", "watch"])
    parser.add_argument("--max_subjects", type=int, default=51)
    args = parser.parse_args()

    base = Path(args.input) / "raw" / args.device
    accel_dir = base / "accel"
    gyro_dir  = base / "gyro"
    out_base  = Path(args.out)

    if not accel_dir.exists():
        print(f"ERROR: {accel_dir} not found")
        sys.exit(1)

    # Find all subject IDs
    subjects = set()
    for f in accel_dir.glob("data_*_accel_*.txt"):
        subjects.add(f.stem.split("_")[1])
    subjects = sorted(subjects)[:args.max_subjects]
    print(f"Found {len(subjects)} subjects, device={args.device}")

    total = 0
    skipped = 0
    domain_counts = {}

    for subj_id in subjects:
        accel_file = accel_dir / f"data_{subj_id}_accel_{args.device}.txt"
        gyro_file  = gyro_dir  / f"data_{subj_id}_gyro_{args.device}.txt"

        if not accel_file.exists():
            continue

        accel_data = parse_raw_file(accel_file)
        gyro_data  = parse_raw_file(gyro_file) if gyro_file.exists() else []

        records = process_subject(subj_id, accel_data, gyro_data, args.device, out_base)

        for (domain, label, rec) in records:
            domain_dir = out_base / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            fname = f"wisdm_{subj_id}_{label}_{rec['fingerprint']}.json"
            with open(domain_dir / fname, 'w') as f:
                json.dump(rec, f, separators=(',', ':'))
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            total += 1

        if total % 500 == 0 and total > 0:
            print(f"  {total} records so far...")

    print(f"\nDone! {total} records certified")
    print(f"Skipped {skipped} REJECTED windows")
    print("\nDomain breakdown:")
    for d, c in sorted(domain_counts.items()):
        print(f"  {d:15s}: {c}")

if __name__ == "__main__":
    main()
