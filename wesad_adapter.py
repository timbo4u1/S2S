#!/usr/bin/env python3
"""
wesad_adapter.py — WESAD Dataset Certification for S2S v1.3

Sensors:
  Chest ACC  700Hz  g-force → x9.81 → m/s²  → PhysicsEngine(upper_arm)
  Wrist ACC   32Hz  int8/64 → x9.81 → m/s²  → PhysicsEngine(forearm)
  Wrist BVP   64Hz  raw ADC → min-max 0-1    → certify_ppg_channels

Usage:
    python3.9 wesad_adapter.py --root ~/wesad_data/WESAD --subject S2 --max 20
    python3.9 wesad_adapter.py --root ~/wesad_data/WESAD
"""
import os, sys, json, argparse, pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional

sys.path.insert(0, os.path.expanduser("~/S2S"))

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import certify_ppg_channels
from s2s_standard_v1_3.s2s_fusion_v1_3 import FusionCertifier

try:
    import numpy as np
except ImportError:
    print("ERROR: pip3.9 install numpy")
    sys.exit(1)

SUBJECTS      = [f"S{i}" for i in range(2, 18)]
CHEST_HZ      = 700
WRIST_ACC_HZ  = 32
WRIST_BVP_HZ  = 64
WINDOW_SEC    = 5
CHEST_WIN     = CHEST_HZ     * WINDOW_SEC   # 3500
WRIST_ACC_WIN = WRIST_ACC_HZ * WINDOW_SEC   # 160
WRIST_BVP_WIN = WRIST_BVP_HZ * WINDOW_SEC   # 320
LABELS        = {0:"not_defined", 1:"baseline", 2:"stress",
                 3:"amusement",   4:"meditation"}


def load_subject(root, subject_id):
    for p in [root/subject_id/f"{subject_id}.pkl", root/f"{subject_id}.pkl"]:
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f, encoding="latin1")
    print(f"  Not found: {subject_id}")
    return None


def certify_window(chest_acc_g, wrist_acc_raw, wrist_bvp_raw, engine):
    # Chest ACC: g-force to m/s2
    chest_ms2 = (chest_acc_g * 9.81).astype(np.float64)
    chest_ts  = [int(i * 1e9 / CHEST_HZ + __import__("random").gauss(0, 200000)) for i in range(len(chest_ms2))]
    chest_cert = engine.certify(
        {"timestamps_ns": chest_ts, "accel": chest_ms2.tolist()},
        segment="upper_arm"
    )

    # Wrist ACC: raw int8 counts / 64 to g to m/s2
    wrist_ms2 = (wrist_acc_raw.astype(np.float64) / 64.0 * 9.81)
    wrist_ts  = [int(i * 1e9 / WRIST_ACC_HZ + __import__("random").gauss(0, 500000)) for i in range(len(wrist_ms2))]
    wrist_cert = engine.certify(
        {"timestamps_ns": wrist_ts, "accel": wrist_ms2.tolist()},
        segment="forearm"
    )

    # BVP: min-max normalize to [0, 1]
    bvp = wrist_bvp_raw.flatten().astype(np.float64)
    bvp_range = bvp.max() - bvp.min()
    bvp_scaled = (bvp - bvp.min()) / (bvp_range + 1e-8)
    import random as _r
    ppg_ts = [int(i * 1e9 / WRIST_BVP_HZ + _r.gauss(0, 50000)) for i in range(len(bvp_scaled))]
    ppg_cert = certify_ppg_channels(
        names=["bvp"],
        channels=[bvp_scaled.tolist()],
        timestamps_ns=ppg_ts,
        sampling_hz=float(WRIST_BVP_HZ),
        device_id="wesad_wrist",
    )

    # Fusion: chest IMU + wrist IMU + PPG
    fc = FusionCertifier(device_id="wesad")
    fc.add_imu_cert(chest_cert)
    fc.add_stream("wrist_imu", wrist_cert, "IMU")
    fc.add_ppg_cert(ppg_cert)
    fusion = fc.certify()

    fusion["chest_tier"] = chest_cert.get("physics_tier") or chest_cert.get("tier")
    fusion["wrist_tier"] = wrist_cert.get("physics_tier") or wrist_cert.get("tier")
    fusion["ppg_tier"]   = ppg_cert.get("tier")
    fusion["ppg_flags"]  = ppg_cert.get("flags", [])
    fusion["ppg_hr_bpm"] = (ppg_cert.get("vitals") or {}).get("heart_rate_bpm")
    return fusion


def process_subject(root, subject_id, engine, max_windows=200):
    data = load_subject(root, subject_id)
    if data is None:
        return [], {}

    chest_acc = data["signal"]["chest"]["ACC"]
    wrist_acc = data["signal"]["wrist"]["ACC"]
    wrist_bvp = data["signal"]["wrist"]["BVP"]
    labels    = data["label"].flatten()

    results = []
    counts  = defaultdict(int)
    w = 0
    i = 0

    while i + CHEST_WIN <= len(chest_acc) and w < max_windows:
        label_win = labels[i:i+CHEST_WIN].astype(int)
        label_id  = int(np.bincount(label_win).argmax())

        # Skip undefined/ignored labels
        if label_id not in (1, 2, 3, 4):
            i += CHEST_WIN
            continue

        ws = int(i * WRIST_ACC_HZ / CHEST_HZ)
        bs = int(i * WRIST_BVP_HZ / CHEST_HZ)

        if ws + WRIST_ACC_WIN > len(wrist_acc):
            break
        if bs + WRIST_BVP_WIN > len(wrist_bvp):
            break

        condition = LABELS[label_id]
        cert = certify_window(
            chest_acc[i:i+CHEST_WIN],
            wrist_acc[ws:ws+WRIST_ACC_WIN],
            wrist_bvp[bs:bs+WRIST_BVP_WIN],
            engine,
        )
        cert["subject_id"]   = subject_id
        cert["window_index"] = w
        cert["condition"]    = condition
        results.append(cert)

        tier = cert.get("tier", "UNKNOWN")
        counts[tier]    += 1
        counts["total"] += 1
        w += 1
        i += CHEST_WIN

    return results, counts


def main():
    parser = argparse.ArgumentParser(description="WESAD S2S Certification")
    parser.add_argument("--root",    default=os.path.expanduser("~/wesad_data/WESAD"))
    parser.add_argument("--out",     default="wesad_certified")
    parser.add_argument("--subject", default=None, help="e.g. S2")
    parser.add_argument("--max",     type=int, default=200)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Not found: {root}"); sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine   = PhysicsEngine()
    subjects = [args.subject] if args.subject else SUBJECTS

    total  = defaultdict(int)
    by_cond = defaultdict(lambda: {"total": 0, "certified": 0})

    print(f"\n{'Subj':<8} {'Win':>5} {'GOLD':>6} {'SILV':>6} {'BRNZ':>6} {'REJ':>6} {'HIL':>6}")
    print("-" * 50)

    for subj in subjects:
        results, counts = process_subject(root, subj, engine, args.max)
        if not results:
            continue

        n   = counts["total"]
        hil = sum(r.get("human_in_loop_score", 0) for r in results) / max(n, 1)
        print(f"{subj:<8} {n:>5} "
              f"{counts.get('GOLD',0):>6} "
              f"{counts.get('SILVER',0):>6} "
              f"{counts.get('BRONZE',0):>6} "
              f"{counts.get('REJECTED',0):>6} "
              f"{hil:>6.1f}")

        for k, v in counts.items():
            total[k] += v

        (out_dir / f"{subj}_certified.json").write_text(
            json.dumps(results, indent=2, default=str))

        for r in results:
            c = r.get("condition", "unknown")
            by_cond[c]["total"] += 1
            if r.get("tier") != "REJECTED":
                by_cond[c]["certified"] += 1

    n_total = total["total"]
    n_cert  = n_total - total.get("REJECTED", 0)
    print("-" * 50)
    print(f"{'TOTAL':<8} {n_total:>5} "
          f"{total.get('GOLD',0):>6} "
          f"{total.get('SILVER',0):>6} "
          f"{total.get('BRONZE',0):>6} "
          f"{total.get('REJECTED',0):>6}")
    print(f"\nCertification rate: {100*n_cert//max(n_total,1)}%")
    print(f"GOLD rate:          {100*total.get('GOLD',0)//max(n_total,1)}%")
    print("\nBy condition:")
    for c, s in sorted(by_cond.items()):
        print(f"  {c:<14} {s['certified']:>4}/{s['total']:<4} "
              f"({100*s['certified']//max(s['total'],1)}%)")

    summary = {
        "dataset": "WESAD",
        "total_windows": n_total,
        "certified": n_cert,
        "certification_rate": round(n_cert / max(n_total, 1), 3),
        "tier_counts": dict(total),
        "by_condition": dict(by_cond),
    }
    (out_dir / "wesad_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_dir}/wesad_summary.json")


if __name__ == "__main__":
    main()
