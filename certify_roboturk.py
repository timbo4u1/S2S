#!/usr/bin/env python3
"""
S2S Audit — RoboTurk (Open-X Embodiment)
=========================================
RoboTurk: 260 human-teleoperated robot demonstrations.
Humans used smartphone controllers — world_vector encodes
human-commanded end-effector motion (x,y,z meters/step).
rotation_delta encodes human wrist rotation (rx,ry,rz rad/step).

This is the closest Open-X gets to human motion data.
S2S certifies whether human motor control patterns are
preserved in the teleoperation signal.

Rate: RoboTurk ~15Hz control frequency
Signal: world_vector integrated → position → accel via finite diff
"""

import os, sys, json, pickle, glob, time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

DATA_DIR   = os.path.expanduser("~/S2S/openx_data")
HZ         = 15
SEGMENT    = "forearm"
NYU_COUNT  = 14   # skip first 14 (nyu_rot episodes)

G  = "\033[92m"; Y = "\033[93m"; R = "\033[91m"
W  = "\033[97m"; D = "\033[2m";  X = "\033[0m"
C  = "\033[96m"

def load_roboturk_episode(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    steps = data.get('steps', [])
    world_vecs = []
    rot_deltas = []
    instruction = ""
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        action = step.get('action', {})
        wv = action.get('world_vector')
        rd = action.get('rotation_delta')
        if wv is not None:
            world_vecs.append(np.array(wv, dtype=np.float32))
        if rd is not None:
            rot_deltas.append(np.array(rd, dtype=np.float32))
        if i == 0:
            obs = step.get('observation', {})
            instr = obs.get('natural_language_instruction', b'')
            instruction = instr.decode() if isinstance(instr, bytes) else str(instr)
    
    if not world_vecs:
        return None, None, instruction
    
    # Integrate deltas → position trajectory
    positions = np.cumsum(np.array(world_vecs), axis=0)
    rotations = np.cumsum(np.array(rot_deltas), axis=0) if rot_deltas else None
    return positions, rotations, instruction

def compute_accel_from_positions(positions, hz):
    if len(positions) < 3:
        return None
    vel   = np.diff(positions, axis=0) * hz
    accel = np.diff(vel, axis=0) * hz
    return accel

def certify_episode(engine, accel, gyro_accel, hz):
    n = len(accel)
    if n < 10:
        return []
    
    window_size = max(10, int(2.0 * hz))  # 2-second windows
    results = []
    
    for i in range(0, n - window_size + 1, window_size // 2):
        chunk_a = accel[i:i+window_size]
        ts = [int(j * 1e9 / hz) for j in range(len(chunk_a))]
        
        # Use rotation as gyro proxy if available
        if gyro_accel is not None and i+window_size <= len(gyro_accel):
            chunk_g = gyro_accel[i:i+window_size].tolist()
        else:
            chunk_g = [[0.0, 0.0, 0.0]] * len(chunk_a)
        
        r = engine.certify(
            imu_raw={
                "timestamps_ns": ts,
                "accel": chunk_a.tolist(),
                "gyro":  chunk_g,
            },
            segment=SEGMENT
        )
        results.append(r)
    return results

def main():
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "sample_*.data.pickle")))
    roboturk_files = all_files[NYU_COUNT:]  # skip nyu_rot

    print(f"\n{W}{'═'*62}")
    print(f"  S2S AUDIT — RoboTurk (Open-X Embodiment)")
    print(f"  {len(roboturk_files)} human-teleoperated episodes | {HZ}Hz | 7-DOF")
    print(f"  Signal: integrated world_vector → end-effector trajectory")
    print(f"{'═'*62}{X}\n")

    engine  = PhysicsEngine()
    stats   = defaultdict(int)
    failing = defaultdict(int)
    scores_all = []
    episode_results = []

    for ep_path in roboturk_files:
        ep_name = os.path.basename(ep_path).replace('.data.pickle','')
        positions, rotations, instruction = load_roboturk_episode(ep_path)

        if positions is None or len(positions) < 5:
            continue

        accel = compute_accel_from_positions(positions, HZ)
        if accel is None:
            continue

        # Use rotation delta acceleration as gyro proxy
        gyro_proxy = None
        if rotations is not None and len(rotations) >= 3:
            gyro_vel   = np.diff(rotations, axis=0) * HZ
            gyro_proxy = np.diff(gyro_vel, axis=0) * HZ
            if len(gyro_proxy) != len(accel):
                gyro_proxy = None

        window_results = certify_episode(engine, accel, gyro_proxy, HZ)
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
        scores_all.extend(scores)

        gold   = tiers.count('GOLD')
        silver = tiers.count('SILVER')
        bronze = tiers.count('BRONZE')
        rej    = tiers.count('REJECTED')
        avg_sc = np.mean(scores)

        pass_rate = (gold + silver) / len(tiers)
        pc = G if pass_rate > 0.8 else (Y if pass_rate > 0.5 else R)

        print(f"  {pc}●{X} {ep_name:<35} "
              f"{len(positions):>3}steps "
              f"sc={avg_sc:>5.1f} "
              f"{G}G:{gold}{X}{C}S:{silver}{X}{Y}B:{bronze}{X}{R}R:{rej}{X} "
              f"{D}{instruction[:28]}{X}")

        episode_results.append({
            "episode":     ep_name,
            "instruction": instruction,
            "n_steps":     len(positions),
            "n_windows":   len(tiers),
            "avg_score":   round(float(avg_sc), 1),
            "tiers":       {t: tiers.count(t) for t in set(tiers)},
            "laws_failed": sorted(set(laws_failed)),
        })

    total = stats['total']
    gold  = stats['GOLD']
    silv  = stats['SILVER']
    bron  = stats['BRONZE']
    rej   = stats['REJECTED']
    pass_rate = 100 * (gold + silv) / max(total, 1)
    avg_score = float(np.mean(scores_all)) if scores_all else 0

    print(f"\n{W}{'═'*62}")
    print(f"  AUDIT RESULTS — RoboTurk")
    print(f"{'═'*62}{X}")
    print(f"  Episodes   : {len(episode_results)}")
    print(f"  Windows    : {total}")
    print(f"  Avg score  : {avg_score:.1f} / 100")
    print(f"  Pass rate  : {pass_rate:.1f}% (GOLD+SILVER)")
    print(f"")
    print(f"  {G}GOLD    {gold:>4} ({100*gold/max(total,1):>5.1f}%){X}")
    print(f"  {C}SILVER  {silv:>4} ({100*silv/max(total,1):>5.1f}%){X}")
    print(f"  {Y}BRONZE  {bron:>4} ({100*bron/max(total,1):>5.1f}%){X}")
    print(f"  {R}REJECTED{rej:>4} ({100*rej/max(total,1):>5.1f}%){X}")

    if failing:
        print(f"\n  Top failing physics laws:")
        for law, count in sorted(failing.items(), key=lambda x: -x[1])[:7]:
            pct = 100 * count / max(total, 1)
            print(f"  {R}  {law:<40}{X} {count:>4} ({pct:.1f}%)")

    # Save
    report = {
        "audit":         "S2S RoboTurk Audit — Open-X Embodiment",
        "dataset":       "roboturk (Open-X Embodiment, Stanford)",
        "description":   "Human teleoperation via smartphone VR controllers",
        "date":          time.strftime("%Y-%m-%d"),
        "s2s_version":   "v1.5.0",
        "hz":            HZ,
        "segment":       SEGMENT,
        "n_episodes":    len(episode_results),
        "n_windows":     total,
        "avg_score":     round(avg_score, 1),
        "pass_rate_pct": round(pass_rate, 1),
        "tiers":         {"GOLD": gold, "SILVER": silv, "BRONZE": bron, "REJECTED": rej},
        "top_failing_laws": dict(sorted(failing.items(), key=lambda x: -x[1])[:10]),
        "episodes":      episode_results,
    }

    out = os.path.expanduser("~/S2S/experiments/roboturk_audit.json")
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report → {out}")
    print(f"{W}{'═'*62}{X}\n")

if __name__ == "__main__":
    main()
