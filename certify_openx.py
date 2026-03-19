#!/usr/bin/env python3
"""
S2S Audit — Open-X Embodiment (NYU Rotation Dataset)
=====================================================
Certifies robot action trajectories using S2S physics engine.
Actions: 7-DOF end-effector (x,y,z,rx,ry,rz,gripper) at ~10Hz
Method: finite-difference acceleration from position trajectory
"""

import os, sys, json, pickle, glob, time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

DATA_DIR = os.path.expanduser("~/S2S/openx_data")
HZ       = 10    # Open-X control frequency ~10Hz
SEGMENT  = "forearm"

G  = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
W  = "\033[97m"; D = "\033[2m";  X = "\033[0m"
C  = "\033[96m"

def load_episode(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    steps = data.get('steps', [])
    actions = []
    for step in steps:
        if isinstance(step, dict) and 'action' in step:
            actions.append(np.array(step['action'], dtype=np.float32))
    return np.array(actions) if actions else None

def compute_accel(positions, hz):
    """Finite difference acceleration from position trajectory."""
    if len(positions) < 3:
        return None
    vel   = np.diff(positions[:, :3], axis=0) * hz        # velocity m/s
    accel = np.diff(vel, axis=0) * hz                      # accel m/s²
    return accel

def certify_episode(engine, accel, hz):
    n = len(accel)
    if n < 20:
        return None

    # Use 20-sample windows (2 seconds at 10Hz)
    window_size = 20
    results = []
    for i in range(0, n - window_size, window_size // 2):
        chunk = accel[i:i+window_size]
        ts    = [int(j * 1e9 / hz) for j in range(len(chunk))]
        gyro  = [[0.0, 0.0, 0.0]] * len(chunk)  # no gyro in Open-X

        r = engine.certify(
            imu_raw={"timestamps_ns": ts,
                     "accel": chunk.tolist(),
                     "gyro":  gyro},
            segment=SEGMENT
        )
        results.append(r)
    return results

def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "sample_*.data.pickle")))
    print(f"\n{W}{'═'*60}")
    print(f"  S2S AUDIT — Open-X Embodiment (NYU Rotation Dataset)")
    print(f"  {len(files)} episodes | {HZ}Hz control | 7-DOF end-effector")
    print(f"{'═'*60}{X}\n")

    engine  = PhysicsEngine()
    stats   = defaultdict(int)
    failing = defaultdict(int)
    episode_results = []

    for ep_path in files:
        ep_name = os.path.basename(ep_path)
        actions = load_episode(ep_path)

        if actions is None or len(actions) < 3:
            print(f"  {R}✗{X} {ep_name} — no actions")
            continue

        # Get language instruction
        with open(ep_path, 'rb') as f:
            raw = pickle.load(f)
        steps = raw.get('steps', [])
        instruction = ""
        if steps and isinstance(steps[0], dict):
            instr = steps[0].get('language_instruction', b'')
            instruction = instr.decode() if isinstance(instr, bytes) else str(instr)

        accel = compute_accel(actions, HZ)
        if accel is None:
            continue

        window_results = certify_episode(engine, accel, HZ)
        if not window_results:
            continue

        tiers  = [r.get('tier', 'REJECTED') for r in window_results]
        scores = [r.get('physical_law_score', 0) for r in window_results]
        laws_failed = []
        for r in window_results:
            laws_failed.extend(r.get('laws_failed', []))

        for t in tiers:
            stats[t] += 1
        stats['total'] += len(tiers)
        for law in laws_failed:
            failing[law] += 1

        gold   = tiers.count('GOLD')
        silver = tiers.count('SILVER')
        bronze = tiers.count('BRONZE')
        rej    = tiers.count('REJECTED')
        avg_sc = np.mean(scores)

        # Color by pass rate
        pass_rate = (gold + silver) / len(tiers)
        pc = G if pass_rate > 0.8 else (Y if pass_rate > 0.5 else R)

        print(f"  {pc}●{X} {ep_name.replace('.data.pickle',''):<35} "
              f"{len(actions):>3} steps  "
              f"score={avg_sc:>5.1f}  "
              f"{G}G:{gold}{X} {C}S:{silver}{X} {Y}B:{bronze}{X} {R}R:{rej}{X}  "
              f"{D}{instruction[:30]}{X}")

        episode_results.append({
            "episode":     ep_name,
            "instruction": instruction,
            "n_steps":     len(actions),
            "n_windows":   len(tiers),
            "avg_score":   round(float(avg_sc), 1),
            "tiers":       {t: tiers.count(t) for t in set(tiers)},
            "laws_failed": list(set(laws_failed)),
        })

    # Summary
    total = stats['total']
    gold  = stats['GOLD']
    silv  = stats['SILVER']
    bron  = stats['BRONZE']
    rej   = stats['REJECTED']
    pass_rate = 100 * (gold + silv) / max(total, 1)

    print(f"\n{W}{'═'*60}")
    print(f"  AUDIT RESULTS — Open-X NYU Rotation Dataset")
    print(f"{'═'*60}{X}")
    print(f"  Episodes   : {len(episode_results)}")
    print(f"  Windows    : {total}")
    print(f"  Pass rate  : {pass_rate:.1f}% (GOLD+SILVER)")
    print(f"")
    print(f"  {G}GOLD    {gold:>4} ({100*gold/max(total,1):>5.1f}%){X}")
    print(f"  {C}SILVER  {silv:>4} ({100*silv/max(total,1):>5.1f}%){X}")
    print(f"  {Y}BRONZE  {bron:>4} ({100*bron/max(total,1):>5.1f}%){X}")
    print(f"  {R}REJECTED{rej:>4} ({100*rej/max(total,1):>5.1f}%){X}")

    if failing:
        print(f"\n  Top failing physics laws:")
        for law, count in sorted(failing.items(), key=lambda x: -x[1])[:5]:
            pct = 100 * count / max(total, 1)
            print(f"  {R}  {law:<35}{X} {count:>4} windows ({pct:.1f}%)")

    print(f"\n  Interpretation:")
    if rej / max(total, 1) > 0.3:
        print(f"  {R}  ⚠ HIGH VIOLATION RATE — robot motion violates physics laws{X}")
        print(f"  {R}    Training on this data teaches physically impossible motion{X}")
    elif rej / max(total, 1) > 0.1:
        print(f"  {Y}  ⚡ MODERATE violations — flag these windows before training{X}")
    else:
        print(f"  {G}  ✓ Low violation rate — data is physically consistent{X}")

    # Save report
    report = {
        "audit":        "S2S Open-X Embodiment Audit",
        "dataset":      "nyu_rot_dataset (Open-X Embodiment)",
        "date":         time.strftime("%Y-%m-%d"),
        "s2s_version":  "v1.5.0",
        "hz":           HZ,
        "segment":      SEGMENT,
        "n_episodes":   len(episode_results),
        "n_windows":    total,
        "pass_rate_pct": round(pass_rate, 1),
        "tiers": {
            "GOLD":     gold,
            "SILVER":   silv,
            "BRONZE":   bron,
            "REJECTED": rej,
        },
        "top_failing_laws": dict(sorted(failing.items(), key=lambda x: -x[1])[:10]),
        "episodes": episode_results,
    }

    out = os.path.expanduser("~/S2S/experiments/openx_audit.json")
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report → {out}")
    print(f"{W}{'═'*60}{X}\n")

if __name__ == "__main__":
    main()
