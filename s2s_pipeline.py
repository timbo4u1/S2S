#!/usr/bin/env python3
"""
S2S Pipeline — Raw sensor data to certified prediction.

Usage:
    python3 s2s_pipeline.py --data ~/ninapro_db5/s5
    python3 s2s_pipeline.py --data ~/ninapro_db5/ --all-subjects
    python3 s2s_pipeline.py --demo

Pipeline:
    RAW DATA → Physics Certification → Biological Origin → Curriculum → Report
"""

import os, sys, glob, json, argparse, time
import scipy.io
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

# ── Constants ─────────────────────────────────────────────────────────────────
HZ        = 2000
WINDOW    = 2000   # 1s at 2000Hz
STEP      = 1000
BFS_MIN   = 10     # minimum windows before certify_session() is meaningful

# ── Colours (terminal) ────────────────────────────────────────────────────────
G  = "\033[92m"   # green
Y  = "\033[93m"   # yellow
R  = "\033[91m"   # red
B  = "\033[94m"   # blue
W  = "\033[97m"   # white bold
DIM= "\033[2m"
X  = "\033[0m"    # reset

def bar(value, total, width=30, colour=G):
    filled = int(width * value / total) if total else 0
    return colour + "█" * filled + DIM + "░" * (width - filled) + X

def tier_colour(tier):
    return {
        "GOLD": "\033[93m", "SILVER": "\033[97m",
        "BRONZE": "\033[33m", "REJECTED": "\033[91m"
    }.get(tier, X)

def grade_colour(grade):
    return G if grade == "HUMAN" else (Y if "LOW" in grade else R)

# ── Load subject ──────────────────────────────────────────────────────────────
def load_subject(path):
    """Load accelerometer data — concatenates ALL mat files for a subject directory."""
    name = None
    if os.path.isdir(path):
        mats = sorted(glob.glob(os.path.join(path, "*.mat")))
        name = os.path.basename(path)
        if not mats:
            return None, None
        # Concatenate all exercises (E1, E2, E3) for stable Hurst estimate
        segments = []
        for mat_path in mats:
            try:
                mat = scipy.io.loadmat(mat_path)
                acc = mat.get('acc', mat.get('ACC'))
                if acc is not None and acc.ndim == 2 and acc.shape[1] >= 3:
                    segments.append(acc[:, :3])
            except Exception:
                continue
        if not segments:
            return None, None
        acc = np.concatenate(segments, axis=0)
        return acc, name
    try:
        mat = scipy.io.loadmat(path)
        acc = mat.get('acc', mat.get('ACC'))
        if acc is None:
            return None, None
        return acc[:, :3] if acc.ndim == 2 else None, os.path.basename(os.path.dirname(path))
    except Exception as e:
        print(f"  {R}Error loading {path}: {e}{X}")
        return None, None

# ── Run full pipeline on one subject ─────────────────────────────────────────
def run_subject(subject_path, verbose=True):
    acc, name = load_subject(subject_path)
    if acc is None:
        return None

    pe     = PhysicsEngine()
    scores, tiers, results = [], [], []
    t0     = time.time()

    n_windows = (len(acc) - WINDOW) // STEP
    for i in range(n_windows):
        w  = acc[i*STEP : i*STEP+WINDOW, :3]
        ts = [j * int(1e9/HZ) for j in range(WINDOW)]
        r  = pe.certify(imu_raw={"timestamps_ns": ts, "accel": w.tolist()}, segment="forearm")
        scores.append(r["physical_law_score"])
        tiers.append(r["tier"])
        results.append(r)

    session = pe.certify_session()
    elapsed = time.time() - t0

    from collections import Counter
    tc     = Counter(tiers)
    n      = len(tiers)
    grade  = session.get("biological_grade", "UNKNOWN")
    bfs    = session.get("bfs", 0)
    hurst  = session.get("hurst", 0)
    rec    = session.get("recommendation", "?")
    avg_sc = sum(scores)/n if n else 0

    if verbose:
        gc = grade_colour(grade)
        print(f"\n  {W}Subject: {name or subject_path}{X}")
        print(f"  {DIM}{'─'*50}{X}")
        print(f"  Windows processed : {n}  ({elapsed:.1f}s)")
        print(f"  Avg physics score : {avg_sc:.1f}/100")
        print(f"")
        print(f"  Tier distribution :")
        for tier in ["GOLD","SILVER","BRONZE","REJECTED"]:
            count = tc.get(tier, 0)
            pct   = 100*count//n if n else 0
            print(f"    {tier_colour(tier)}{tier:<10}{X}  {bar(count, n, 20, tier_colour(tier))}  {pct:3d}%  ({count})")
        print(f"")
        print(f"  Biological origin :")
        print(f"    Grade       : {gc}{grade}{X}")
        print(f"    Hurst (H)   : {hurst}  {'✅' if hurst and hurst >= 0.7 else '❌'}")
        print(f"    BFS         : {bfs}")
        print(f"    Decision    : {gc}{rec}{X}")
        print(f"  {DIM}{'─'*50}{X}")

    return {
        "name":     name,
        "n":        n,
        "avg_score": round(avg_sc, 1),
        "tiers":    dict(tc),
        "grade":    grade,
        "hurst":    hurst,
        "bfs":      bfs,
        "recommendation": rec,
        "accepted": rec == "ACCEPT",
        "elapsed":  round(elapsed, 2)
    }

# ── Full pipeline summary ─────────────────────────────────────────────────────
def run_pipeline(data_path, all_subjects=False):
    print(f"\n{W}{'═'*55}")
    print(f"  S2S PIPELINE — Physics + Biological Certification")
    print(f"{'═'*55}{X}")
    print(f"  Input  : {data_path}")
    print(f"  Levels : Physics (1-4) → Biological Origin → Curriculum")

    subjects = []
    if all_subjects and os.path.isdir(data_path):
        subdirs = sorted([
            os.path.join(data_path, d)
            for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        ])
        subjects = subdirs
    else:
        subjects = [data_path]

    print(f"  Subjects found : {len(subjects)}\n")

    all_results = []
    for s in subjects:
        r = run_subject(s, verbose=True)
        if r:
            all_results.append(r)

    if len(all_results) > 1:
        accepted = [r for r in all_results if r["accepted"]]
        rejected = [r for r in all_results if not r["accepted"]]

        print(f"\n{W}{'═'*55}")
        print(f"  PIPELINE SUMMARY")
        print(f"{'═'*55}{X}")
        print(f"  Sessions processed : {len(all_results)}")
        print(f"  {G}Accepted (HUMAN)   : {len(accepted)}{X}")
        print(f"  {R}Rejected           : {len(rejected)}{X}")
        if rejected:
            for r in rejected:
                print(f"    {R}↳ {r['name']} — {r['grade']} (H={r['hurst']}){X}")
        total_windows   = sum(r["n"] for r in all_results)
        accepted_windows= sum(r["n"] for r in accepted)
        print(f"")
        print(f"  Total windows      : {total_windows}")
        print(f"  {G}Curriculum windows : {accepted_windows}{X}  ({100*accepted_windows//total_windows if total_windows else 0}% of total)")
        print(f"  {R}Filtered out       : {total_windows - accepted_windows}{X}  windows from non-biological sessions")

    # Save report
    report = {
        "pipeline": "S2S v1.5.0",
        "input": data_path,
        "n_sessions": len(all_results),
        "n_accepted": len([r for r in all_results if r["accepted"]]),
        "sessions": all_results
    }
    out = "experiments/pipeline_report.json"
    os.makedirs("experiments", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved → {out}")
    print(f"{W}{'═'*55}{X}\n")
    return report

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="S2S Pipeline — raw data to certified prediction")
    p.add_argument("--data",         type=str, help="Path to subject folder or dataset root")
    p.add_argument("--all-subjects", action="store_true", help="Run all subject subdirectories")
    p.add_argument("--demo",         action="store_true", help="Demo on NinaPro DB5 all subjects")
    args = p.parse_args()

    if args.demo or not args.data:
        run_pipeline(os.path.expanduser("~/ninapro_db5"), all_subjects=True)
    else:
        run_pipeline(os.path.expanduser(args.data), all_subjects=args.all_subjects)
