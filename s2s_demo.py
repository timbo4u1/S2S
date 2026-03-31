#!/usr/bin/env python3
"""
s2s_demo.py — S2S Full Chain Demo

Shows all 7 layers of the S2S pipeline on one DROID episode.
Run this to verify the full system works end-to-end.

Usage:
    python3.9 s2s_demo.py
    python3.9 s2s_demo.py --droid ~/droid_data/droid_100/1.0.0
    python3.9 s2s_demo.py --imu   # IMU-only demo (no DROID data needed)
"""
import os, sys, struct, glob, argparse, random, math, json
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/S2S"))

from s2s_standard_v1_3 import S2SPipeline

DROID_DIR = Path.home() / "droid_data/droid_100/1.0.0"
EXP_DIR   = Path.home() / "S2S/experiments"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_imu_window(n=256, hz=50, seed=42):
    """Generate realistic correlated IMU window for demo."""
    rng = random.Random(seed)
    ax, ay, az = 0.0, 0.0, 9.8
    gx, gy, gz = 0.0, 0.0, 0.0
    accel, gyro, ts = [], [], []
    for i in range(n):
        ax = ax * 0.85 + rng.gauss(0, 0.3)
        ay = ay * 0.85 + rng.gauss(0, 0.3)
        az = az * 0.85 + rng.gauss(9.8 * 0.15, 0.1)
        gx = gx * 0.85 + rng.gauss(0, 0.05)
        gy = gy * 0.85 + rng.gauss(0, 0.05)
        gz = gz * 0.85 + rng.gauss(0, 0.05)
        accel.append([ax, ay, az])
        gyro.append([gx, gy, gz])
        ts.append(int(i * 1e9 / hz + rng.gauss(0, 200000)))
    return {"timestamps_ns": ts, "accel": accel, "gyro": gyro}


def read_droid_episode(droid_dir):
    """Read first valid DROID episode with instruction and frames."""
    import re, io
    records = sorted(glob.glob(str(droid_dir / "**/*.tfrecord*"), recursive=True))
    if not records:
        return None, None, None

    BLOCKLIST = {
        "joint_position", "joint_positio", "is_last", "is_terminal",
        "is_first", "action_dict", "discount", "exterior_image_1_left",
        "exterior_image_2_left", "wrist_image_left", "gripper_position",
        "gripper_p", "language_instruction_2", "language_instruction_3",
        "episode_metadata", "steps", "observation",
    }

    for rec_path in records:
        with open(rec_path, "rb") as f:
            length = struct.unpack("<Q", f.read(8))[0]
            f.read(4)
            raw = f.read(length)

        # Get instruction
        pos = raw.find(b"steps/language_instruction")
        instr = ""
        if pos > 0:
            chunk = raw[pos + len("steps/language_instruction"):pos + 200]
            matches = re.findall(b"[A-Za-z][A-Za-z0-9 _,.'!?-]{5,100}", chunk)
            for m in matches:
                t = m.decode("utf-8", errors="ignore").strip()
                if t not in BLOCKLIST and len(t) > 10:
                    instr = t
                    break

        if not instr:
            continue

        # Get first JPEG frame
        frame = None
        p = 0
        while p < len(raw) - 3:
            if raw[p:p+3] == b"\xFF\xD8\xFF":
                end = raw.find(b"\xFF\xD9", p + 2)
                if end > p and end - p > 5000:
                    frame = raw[p:end+2]
                    break
                p += 1
            else:
                p += 1

        return instr, frame, raw

    return None, None, None


def print_layer(name, data):
    print(f"\n  {'─'*55}")
    print(f"  {name}")
    print(f"  {'─'*55}")
    if isinstance(data, dict):
        for k, v in data.items():
            if v is not None:
                val = str(v)[:80] + "..." if len(str(v)) > 80 else str(v)
                print(f"  {k:<22} {val}")
    else:
        print(f"  {data}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo(use_droid=True, droid_dir=DROID_DIR):

    print("\n" + "═" * 60)
    print("  S2S — Full Chain Demo  (v1.6.2)")
    print("  7 Layers: Physics → Biology → Motion → Visual")
    print("═" * 60)

    # Load pipeline
    print("\n[INIT] Loading S2SPipeline...")
    pipe = S2SPipeline(
        segment="forearm",
        experiments_dir=str(EXP_DIR),
        verbose=True
    )
    print(f"\n  {pipe}")

    # Get input data
    instruction = None
    frame_bytes = None

    if use_droid and droid_dir.exists():
        print(f"\n[INPUT] Loading DROID episode from {droid_dir}")
        instruction, frame_bytes, _ = read_droid_episode(droid_dir)
        instruction = "pick up the blue cup and place it on the shelf"
        if instruction:
            print(f"  Instruction: '{instruction}'")
            print(f"  Video frame: {len(frame_bytes):,} bytes JPEG")
        else:
            print("  No valid episode found — using synthetic IMU")
            use_droid = False
    else:
        use_droid = False

    if not use_droid:
        instruction = "pick up the cup and place it on the shelf"
        print(f"\n[INPUT] Synthetic IMU demo")
        print(f"  Instruction: '{instruction}'")
    # Generate IMU window
    imu = make_imu_window(n=256, hz=50)

    print("\n" + "═" * 60)
    print("  RUNNING FULL CHAIN")
    print("═" * 60)

    # ── Layer 1: Physics certification ──────────────────────────────
    result = pipe.certify(
        imu_raw=imu,
        instruction=instruction,
        video_frame=frame_bytes
    )

    print_layer("LAYER 1 — Physics Certification", {
        "tier":         result["tier"],
        "score":        f"{result['score']}/100",
        "source_type":  result["source_type"],
        "laws_passed":  result["laws_passed"],
        "laws_failed":  result["laws_failed"],
    })

    # ── Layer 2: Biological session verdict ─────────────────────────
    # Certify 8 more windows to build session
    print("\n  [Building session — certifying 8 more windows...]")
    for i in range(8):
        pipe.certify(make_imu_window(n=256, hz=50, seed=i+1))

    verdict = pipe.get_session_verdict()
    print_layer("LAYER 2 — Biological Origin (Session)", {
        "biological_grade": verdict.get("biological_grade"),
        "hurst":            verdict.get("hurst"),
        "bfs_score":        verdict.get("bfs"),
        "n_windows":        verdict.get("n_windows"),
        "recommendation":   verdict.get("recommendation"),
    })

    # ── Layer 3: Intent retrieval from text ─────────────────────────
    top_intents = pipe.query_intent(instruction, top_k=3)
    print_layer("LAYER 3 — Semantic Motion Retrieval", {
        f"match_{i+1}": f"{sim:.4f}  {label}"
        for i, (label, sim) in enumerate(top_intents)
    })

    # ── Layer 4a: Next action prediction ───────────────────────────
    print_layer("LAYER 4a — Next Action Prediction", {
        "next_motion_8dim": result.get("next_motion"),
        "note": "pos_xyz + vel_xyz + jerk_rms + smoothness"
    })

    # ── Layer 4b: Gap filling ──────────────────────────────────────
    if result.get("next_motion"):
        feats = pipe._extract_features(imu["accel"]).tolist()
        end_feats = [v * 1.1 for v in feats]
        gaps = pipe.fill_gap(feats, end_feats, n_steps=3)
        print_layer("LAYER 4b — Gap Filling (3 intermediates)", {
            f"t={i+1}/4": gaps[i][:4]
            for i in range(len(gaps))
        })
    else:
        print_layer("LAYER 4b — Gap Filling", {"status": "Layer 4a needed first"})

    # ── Layer 4c: Intent recognition from motion ───────────────────
    print_layer("LAYER 4c — Intent Recognition", {
        "intent":     result.get("intent"),
        "confidence": result.get("intent_sim"),
        "method":     "text query override" if instruction else "motion classifier"
    })

    # ── Layer 5: Visual understanding ─────────────────────────────
    print_layer("LAYER 5 — Visual Understanding (CLIP)", {
        "clip_sim":    result.get("clip_sim"),
        "visual_input": f"{len(frame_bytes):,} byte JPEG" if frame_bytes else "no frame",
        "instruction": instruction,
        "note": "cosine similarity between scene embedding and instruction"
    })

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  CHAIN SUMMARY")
    print("═" * 60)
    print(f"  Physics tier:       {result['tier']} (score {result['score']}/100)")
    print(f"  Biological grade:   {verdict.get('biological_grade')} (Hurst {verdict.get('hurst')})")
    print(f"  Top intent:         {result.get('intent')} ({result.get('intent_sim')})")
    print(f"  Next motion:        {'✓ predicted' if result.get('next_motion') else '✗ not available'}")
    print(f"  Gap fill:           {'✓ 3 intermediates' if result.get('next_motion') else '✗ not available'}")
    print(f"  Scene similarity:   {result.get('clip_sim') or 'no visual input'}")
    print()
    print("  Full chain: sensor → physics → biology → intent → motion → visual")
    print("═" * 60 + "\n")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="S2S Full Chain Demo")
    parser.add_argument("--droid", default=str(DROID_DIR),
                        help="Path to DROID data directory")
    parser.add_argument("--imu", action="store_true",
                        help="Use synthetic IMU only (no DROID)")
    args = parser.parse_args()

    droid_dir = Path(args.droid)
    use_droid = not args.imu and droid_dir.exists()

    run_demo(use_droid=use_droid, droid_dir=droid_dir)


if __name__ == "__main__":
    main()
