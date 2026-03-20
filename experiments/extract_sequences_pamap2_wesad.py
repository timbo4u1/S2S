#!/usr/bin/env python3
"""
extract_sequences_pamap2_wesad.py
==================================
Adds PAMAP2 (source=3) and WESAD (source=4) to sequences_real.npz.

PAMAP2: .dat files 100Hz, hand+chest IMU with gyro
WESAD:  PKL files, chest 700Hz accel-only

Usage:
    python3.9 experiments/extract_sequences_pamap2_wesad.py
"""
import os, sys, glob, pickle, random
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

PAMAP2_DIR   = Path.home() / "S2S_data/pamap2"
WESAD_DIR    = Path.home() / "wesad_data/WESAD"
NPZ_PATH     = Path("experiments/sequences_real.npz")

WINDOW_PAMAP2      = 500   # 5s at 100Hz
WINDOW_WESAD       = 3500  # 5s at 700Hz
PAMAP2_HZ          = 100
WESAD_HZ           = 700


def extract_features(accel, gyro, hz):
    accel = np.array(accel, dtype=np.float64)
    gyro  = np.array(gyro,  dtype=np.float64)
    if accel.ndim == 1: accel = accel.reshape(-1, 1)
    if gyro.ndim  == 1: gyro  = gyro.reshape(-1, 1)
    while accel.shape[1] < 3: accel = np.hstack([accel, np.zeros((len(accel),1))])
    while gyro.shape[1]  < 3: gyro  = np.hstack([gyro,  np.zeros((len(gyro),1))])
    f = []
    f.append(float(np.sqrt(np.mean(accel**2))))
    f.append(float(np.std(accel)))
    f.append(float(np.max(np.abs(accel))))
    f.append(float(np.sqrt(np.mean(gyro**2))))
    f.append(float(np.std(gyro)))
    if len(accel) > 3:
        jerk = np.diff(accel, n=1, axis=0) * hz
        f.append(float(np.sqrt(np.mean(jerk**2))))
        f.append(float(np.percentile(np.abs(jerk), 95)))
    else:
        f.extend([0.0, 0.0])
    for axis in range(3):
        fft   = np.abs(np.fft.rfft(accel[:, axis]))
        freqs = np.fft.rfftfreq(len(accel), 1/hz)
        f.append(float(freqs[np.argmax(fft)] if len(fft) > 0 else 0))
    c = np.corrcoef(accel[:,0], accel[:,1])[0,1]
    f.append(float(c) if not np.isnan(c) else 0.0)
    f.append(float(np.linalg.norm(np.mean(accel, axis=0))))
    hist, _ = np.histogram(accel.flatten(), bins=20)
    hist = hist / (hist.sum() + 1e-10)
    f.append(float(-np.sum(hist * np.log(hist + 1e-10))))
    return f


def window_summary(accel, gyro, hz):
    accel = np.array(accel, dtype=np.float32)
    vel   = np.cumsum(accel, axis=0) / hz
    pos   = np.cumsum(vel,   axis=0) / hz
    jerk  = np.diff(accel, axis=0) * hz
    smoothness = float(1.0 / (1.0 + np.sqrt(np.mean(jerk**2))))
    return {
        "mean_pos":   pos.mean(axis=0).tolist(),
        "final_vel":  vel[-1].tolist(),
        "jerk_rms":   float(np.sqrt(np.mean(jerk**2))),
        "smoothness": smoothness,
    }


def certify(engine, accel, gyro, hz, segment, jitter_ns=0):
    n  = len(accel)
    ts = [int(i*1e9/hz + (random.gauss(0, jitter_ns) if jitter_ns else 0))
          for i in range(n)]
    r  = engine.certify(
        {"timestamps_ns": ts, "accel": accel.tolist(), "gyro": gyro.tolist()},
        segment=segment,
    )
    return r.get("tier", "REJECTED")


def process_chunks(engine, chunks_a, chunks_g, hz, source_id, segment,
                   pairs_X, pairs_Y, pairs_T, pairs_S, jitter_ns=0):
    results = []
    for a, g in zip(chunks_a, chunks_g):
        tier = certify(engine, a, g, hz, segment, jitter_ns)
        results.append((tier, a, g))

    count = 0
    for i in range(len(results)-1):
        tier_i, a_i, g_i = results[i]
        tier_j, a_j, g_j = results[i+1]
        if tier_i in ("SILVER","BRONZE","GOLD") and tier_j in ("SILVER","BRONZE","GOLD"):
            feats = extract_features(a_i, g_i, hz)
            summ  = window_summary(a_j, g_j, hz)
            y = summ["mean_pos"] + summ["final_vel"] + [summ["jerk_rms"], summ["smoothness"]]
            pairs_X.append(feats)
            pairs_Y.append(y)
            pairs_T.append(tier_i)
            pairs_S.append(source_id)
            count += 1
    return count


def main():
    engine = PhysicsEngine()

    # Load existing
    d = np.load(str(NPZ_PATH), allow_pickle=True)
    pairs_X = d["X"].tolist()
    pairs_Y = d["Y"].tolist()
    pairs_T = d["tiers"].tolist()
    pairs_S = d["sources"].tolist()
    print(f"Existing: {len(pairs_X)} pairs  "
          f"(NinaPro={pairs_S.count(0)} Amputee={pairs_S.count(1)} "
          f"RoboTurk={pairs_S.count(2)})")

    # ------------------------------------------------------------------
    # PAMAP2
    # ------------------------------------------------------------------
    print("\n[1/2] PAMAP2...")
    n_pamap2 = 0
    if not PAMAP2_DIR.exists():
        print(f"  Not found: {PAMAP2_DIR}")
    else:
        dat_files = sorted(PAMAP2_DIR.glob("*.dat"))
        print(f"  Found {len(dat_files)} .dat files")
        for dat_path in dat_files:
            try:
                data = np.genfromtxt(str(dat_path), delimiter=" ", filling_values=0)
                if data.ndim < 2 or data.shape[1] < 50:
                    continue
                data = np.nan_to_num(data, nan=0.0)
                n    = len(data)

                # Hand: accel cols 4:7, gyro cols 10:13
                a_hand = data[:, 4:7].astype(np.float64)
                g_hand = data[:, 10:13].astype(np.float64)
                chunks_a = [a_hand[i:i+WINDOW_PAMAP2] for i in range(0, n-WINDOW_PAMAP2*2, WINDOW_PAMAP2)]
                chunks_g = [g_hand[i:i+WINDOW_PAMAP2] for i in range(0, n-WINDOW_PAMAP2*2, WINDOW_PAMAP2)]
                c = process_chunks(engine, chunks_a, chunks_g,
                                   PAMAP2_HZ, 3, "forearm",
                                   pairs_X, pairs_Y, pairs_T, pairs_S)
                n_pamap2 += c

                # Chest: accel cols 22:25, gyro cols 28:31
                a_chest = data[:, 22:25].astype(np.float64)
                g_chest = data[:, 28:31].astype(np.float64)
                chunks_a = [a_chest[i:i+WINDOW_PAMAP2] for i in range(0, n-WINDOW_PAMAP2*2, WINDOW_PAMAP2)]
                chunks_g = [g_chest[i:i+WINDOW_PAMAP2] for i in range(0, n-WINDOW_PAMAP2*2, WINDOW_PAMAP2)]
                c = process_chunks(engine, chunks_a, chunks_g,
                                   PAMAP2_HZ, 3, "upper_arm",
                                   pairs_X, pairs_Y, pairs_T, pairs_S)
                n_pamap2 += c

            except Exception as e:
                print(f"  {dat_path.name} error: {e}")
    print(f"  → {n_pamap2} PAMAP2 pairs")

    # ------------------------------------------------------------------
    # WESAD
    # ------------------------------------------------------------------
    print("\n[2/2] WESAD chest sequences...")
    n_wesad = 0
    if not WESAD_DIR.exists():
        print(f"  Not found: {WESAD_DIR}")
    else:
        for subj in [f"S{i}" for i in range(2,18)]:
            pkl = WESAD_DIR / subj / f"{subj}.pkl"
            if not pkl.exists():
                continue
            try:
                with open(pkl,"rb") as f:
                    data = pickle.load(f, encoding="latin1")
                chest = (data["signal"]["chest"]["ACC"] * 9.81).astype(np.float64)
                labels = data["label"].flatten()
                zeros  = np.zeros_like(chest)
                n      = len(chest)

                chunks_a, chunks_g = [], []
                i = 0
                while i + WINDOW_WESAD <= n:
                    win_labels = labels[i:i+WINDOW_WESAD].astype(int)
                    label_id   = int(np.bincount(win_labels).argmax())
                    if label_id in (1,2,3,4):
                        chunks_a.append(chest[i:i+WINDOW_WESAD])
                        chunks_g.append(zeros[i:i+WINDOW_WESAD])
                    i += WINDOW_WESAD

                c = process_chunks(engine, chunks_a, chunks_g,
                                   WESAD_HZ, 4, "upper_arm",
                                   pairs_X, pairs_Y, pairs_T, pairs_S,
                                   jitter_ns=200000)
                n_wesad += c
                print(f"  {subj}: +{c}")
            except Exception as e:
                print(f"  {subj} error: {e}")
    print(f"  → {n_wesad} WESAD pairs")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    X = np.array(pairs_X, dtype=np.float32)
    Y = np.array(pairs_Y, dtype=np.float32)
    T = np.array(pairs_T, dtype=str)
    S = np.array(pairs_S, dtype=np.int8)

    np.savez(str(NPZ_PATH), X=X, Y=Y, tiers=T, sources=S)

    print(f"\n{'='*50}")
    print(f"Total pairs: {len(X)}")
    print(f"  NinaPro  (0): {int(np.sum(S==0))}")
    print(f"  Amputee  (1): {int(np.sum(S==1))}")
    print(f"  RoboTurk (2): {int(np.sum(S==2))}")
    print(f"  PAMAP2   (3): {int(np.sum(S==3))}")
    print(f"  WESAD    (4): {int(np.sum(S==4))}")
    print(f"Saved → {NPZ_PATH}")


if __name__ == "__main__":
    main()
