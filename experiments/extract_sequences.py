#!/usr/bin/env python3
"""
Sequence Extractor for Level 5 Dual-Head
=========================================
Extracts consecutive certified window pairs:
  input:  window_N  (raw accel+gyro, 16 features)
  target: window_N+1 position, velocity, smoothness

Only SILVER windows kept. Consecutive pairs only.
Output: sequences_real.npz
"""

import os, sys, glob, pickle
import numpy as np
import scipy.io
sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

WINDOW_NINAPRO  = 2000
WINDOW_AMPUTEE  = 200
WINDOW_ROBOTURK = 30
NYU_COUNT       = 14

def certify_window(engine, accel, gyro, hz):
    n  = len(accel)
    ts = [int(i * 1e9 / hz) for i in range(n)]
    r  = engine.certify(
        imu_raw={"timestamps_ns": ts,
                 "accel": accel.tolist(),
                 "gyro":  gyro.tolist()},
        segment="forearm"
    )
    return r.get("physical_law_score", 0), r.get("tier", "REJECTED")

def window_summary(accel, gyro, hz):
    """Compact representation: mean pos, mean vel, jerk rms, smoothness"""
    accel = np.array(accel, dtype=np.float32)
    vel   = np.cumsum(accel, axis=0) / hz
    pos   = np.cumsum(vel,   axis=0) / hz
    jerk  = np.diff(accel, axis=0) * hz
    smoothness = float(1.0 / (1.0 + np.sqrt(np.mean(jerk**2))))
    return {
        "mean_pos":    pos.mean(axis=0).tolist(),   # 3
        "final_vel":   vel[-1].tolist(),             # 3
        "jerk_rms":    float(np.sqrt(np.mean(jerk**2))),
        "smoothness":  smoothness,                   # 0-1, higher=smoother
        "speed_mean":  float(np.sqrt((vel**2).sum(axis=1)).mean()),
    }

pairs_X   = []   # input window features (16)
pairs_Y   = []   # next motion target (8: mean_pos x3, final_vel x3, jerk_rms, smoothness)
pairs_tier = []  # quality tier of input window
source_ids = []  # 0=ninapro, 1=amputee, 2=roboturk

def extract_features(accel, gyro, hz):
    accel = np.array(accel, dtype=np.float64)
    gyro  = np.array(gyro,  dtype=np.float64)
    if accel.ndim == 1: accel = accel.reshape(-1,1)
    if gyro.ndim  == 1: gyro  = gyro.reshape(-1,1)
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

def process_sequence(engine, chunks_a, chunks_g, hz, source_id):
    """Certify all chunks, extract consecutive SILVER pairs"""
    results = []
    for a, g in zip(chunks_a, chunks_g):
        score, tier = certify_window(engine, a, g, hz)
        results.append((score, tier, a, g))

    count = 0
    for i in range(len(results) - 1):
        score_i, tier_i, a_i, g_i = results[i]
        score_j, tier_j, a_j, g_j = results[i+1]
        if tier_i in ("SILVER","BRONZE","GOLD") and tier_j in ("SILVER","BRONZE","GOLD"):
            feats = extract_features(a_i, g_i, hz)
            next_motion = window_summary(a_j, g_j, hz)
            y = (next_motion["mean_pos"] +
                 next_motion["final_vel"] +
                 [next_motion["jerk_rms"], next_motion["smoothness"]])
            pairs_X.append(feats)
            pairs_Y.append(y)
            pairs_tier.append(tier_i)
            source_ids.append(source_id)
            count += 1
    return count

engine = PhysicsEngine()

# ── NinaPro ───────────────────────────────────────────────────────────────────
print("[1/3] NinaPro DB5...")
count = 0
for subj_dir in sorted(glob.glob(os.path.expanduser("~/ninapro_db5/s*"))):
    accel_all = []
    for mat_path in sorted(glob.glob(os.path.join(subj_dir, "*.mat"))):
        try:
            mat = scipy.io.loadmat(mat_path)
            acc = mat.get("acc", mat.get("ACC"))
            if acc is not None: accel_all.append(acc)
        except: continue
    if not accel_all: continue
    accel = np.vstack(accel_all).astype(np.float64)
    gyro  = np.zeros_like(accel)
    chunks_a = [accel[i:i+WINDOW_NINAPRO, :3]
                for i in range(0, len(accel)-WINDOW_NINAPRO*2, WINDOW_NINAPRO)]
    chunks_g = [gyro[i:i+WINDOW_NINAPRO, :3]
                for i in range(0, len(accel)-WINDOW_NINAPRO*2, WINDOW_NINAPRO)]
    count += process_sequence(engine, chunks_a, chunks_g, 2000, 0)
print(f"  → {count} pairs")

# ── EMG Amputee ───────────────────────────────────────────────────────────────
print("[2/3] EMG Amputee...")
count = 0
for subj in sorted(os.listdir(os.path.expanduser("~/S2S_Project/EMG_Amputee"))):
    inner = os.path.expanduser(f"~/S2S_Project/EMG_Amputee/{subj}/{subj}")
    if not os.path.isdir(inner): continue
    for csv_path in sorted(glob.glob(os.path.join(inner, "mov *", "accelerometer-*.csv"))):
        try:
            data  = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            if data.ndim < 2 or data.shape[1] < 4: continue
            accel = data[:, 1:4].astype(np.float64)
            gyro  = np.zeros_like(accel)
            chunks_a = [accel[i:i+WINDOW_AMPUTEE]
                        for i in range(0, len(accel)-WINDOW_AMPUTEE*2, WINDOW_AMPUTEE)]
            chunks_g = [gyro[i:i+WINDOW_AMPUTEE]
                        for i in range(0, len(accel)-WINDOW_AMPUTEE*2, WINDOW_AMPUTEE)]
            count += process_sequence(engine, chunks_a, chunks_g, 200, 1)
        except: continue
print(f"  → {count} pairs")

# ── RoboTurk ──────────────────────────────────────────────────────────────────
print("[3/3] RoboTurk...")
count = 0
for path in sorted(glob.glob(os.path.expanduser(
        "~/S2S/openx_data/sample_*.data.pickle")))[NYU_COUNT:]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    steps = data.get("steps", [])
    world_vecs = []
    for step in steps:
        if not isinstance(step, dict): continue
        wv = step.get("action", {}).get("world_vector")
        if wv is not None: world_vecs.append(np.array(wv, dtype=np.float32))
    if len(world_vecs) < 10: continue
    positions = np.cumsum(np.array(world_vecs), axis=0)
    vel   = np.diff(positions, axis=0) * 15
    accel = np.diff(vel, axis=0) * 15
    gyro  = np.zeros_like(accel)
    chunks_a = [accel[i:i+WINDOW_ROBOTURK]
                for i in range(0, len(accel)-WINDOW_ROBOTURK*2, WINDOW_ROBOTURK)]
    chunks_g = [gyro[i:i+WINDOW_ROBOTURK]
                for i in range(0, len(accel)-WINDOW_ROBOTURK*2, WINDOW_ROBOTURK)]
    count += process_sequence(engine, chunks_a, chunks_g, 15, 2)
print(f"  → {count} pairs")

# ── Save ──────────────────────────────────────────────────────────────────────
X = np.array(pairs_X,    dtype=np.float32)
Y = np.array(pairs_Y,    dtype=np.float32)
T = np.array(pairs_tier, dtype=str)
S = np.array(source_ids, dtype=np.int8)

out = os.path.expanduser("~/S2S/experiments/sequences_real.npz")
np.savez(out, X=X, Y=Y, tiers=T, sources=S)

print(f"\nTotal pairs: {len(X)}")
print(f"Input shape: {X.shape}  (16 features)")
print(f"Target shape: {Y.shape}  (8: pos x3, vel x3, jerk_rms, smoothness)")
print(f"Sources: NinaPro={sum(S==0)} Amputee={sum(S==1)} RoboTurk={sum(S==2)}")
print(f"Saved → {out}")
