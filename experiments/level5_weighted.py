#!/usr/bin/env python3
"""
Level 5 — Weighted Semi-Supervised Training
============================================
Trains on ALL data (raw + certified).
Certified windows = full weight (1.0)
Raw uncertified = weight by predicted quality score

Loop:
  1. Train motion predictor on weighted data
  2. Quality predictor scores all raw windows
  3. Update weights
  4. Repeat — model improves without new data
"""

import os, sys, json, time, glob, pickle
import numpy as np
import scipy.io
sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

WINDOW_NINAPRO  = 2000
WINDOW_AMPUTEE  = 200
WINDOW_ROBOTURK = 30
NYU_COUNT       = 14
HZ_MAP          = {0: 2000, 1: 200, 2: 15}

# ── Feature extractor (same as before) ───────────────────────────────────────
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

def window_motion_target(accel, hz):
    accel = np.array(accel, dtype=np.float32)
    vel   = np.cumsum(accel, axis=0) / hz
    pos   = np.cumsum(vel,   axis=0) / hz
    jerk  = np.diff(accel, axis=0) * hz
    smoothness = float(1.0 / (1.0 + np.sqrt(np.mean(jerk**2))))
    return (pos.mean(axis=0).tolist() +
            vel[-1].tolist() +
            [float(np.sqrt(np.mean(jerk**2))), smoothness])

# ── Load certified sequences ──────────────────────────────────────────────────
print("Loading certified sequence data...")
cert = np.load(os.path.expanduser("~/S2S/experiments/sequences_real.npz"),
               allow_pickle=True)
X_cert = cert["X"].astype(np.float32)
Y_cert = cert["Y"].astype(np.float32)
T_cert = cert["tiers"]
# Quality label: SILVER=1, BRONZE=0
Q_cert = np.array([1 if t=="SILVER" else 0 for t in T_cert], dtype=np.int64)
W_cert = np.ones(len(X_cert), dtype=np.float32)  # full weight
print(f"  Certified pairs: {len(X_cert)}")

# ── Load raw uncertified data (RoboTurk only — fast) ─────────────────────────
print("Loading raw uncertified data (RoboTurk)...")
engine = PhysicsEngine()
X_raw, Y_raw, Q_raw = [], [], []

files = sorted(glob.glob(os.path.expanduser(
    "~/S2S/openx_data/sample_*.data.pickle")))[NYU_COUNT:]

for path in files:  # first 50 episodes
    with open(path,"rb") as f:
        data = pickle.load(f)
    steps = data.get("steps",[])
    wvs = [np.array(s.get("action",{}).get("world_vector"),dtype=np.float32)
           for s in steps if isinstance(s,dict)
           and s.get("action",{}).get("world_vector") is not None]
    if len(wvs) < 10: continue
    positions = np.cumsum(np.array(wvs), axis=0)
    vel   = np.diff(positions, axis=0) * 15
    accel = np.diff(vel, axis=0) * 15
    gyro  = np.zeros_like(accel)
    chunks = [(accel[i:i+WINDOW_ROBOTURK], gyro[i:i+WINDOW_ROBOTURK])
              for i in range(0, len(accel)-WINDOW_ROBOTURK*2, WINDOW_ROBOTURK)]
    for j in range(len(chunks)-1):
        ca, cg = chunks[j]
        na, _  = chunks[j+1]
        if len(ca) < 10 or len(na) < 10: continue
        feats  = extract_features(ca, cg, 15)
        target = window_motion_target(na, 15)
        X_raw.append(feats)
        Y_raw.append(target)
        Q_raw.append(0)  # unknown quality — starts at 0

X_raw = np.array(X_raw, dtype=np.float32)
Y_raw = np.array(Y_raw, dtype=np.float32)
Q_raw = np.array(Q_raw, dtype=np.int64)
W_raw = np.full(len(X_raw), 0.1, dtype=np.float32)  # low initial weight

print(f"  Raw pairs: {len(X_raw)}")

# ── Combine all data ──────────────────────────────────────────────────────────
X_all = np.vstack([X_cert, X_raw])
Y_all = np.vstack([Y_cert, Y_raw])
Q_all = np.concatenate([Q_cert, Q_raw])
W_all = np.concatenate([W_cert, W_raw])
IS_CERTIFIED = np.array([True]*len(X_cert) + [False]*len(X_raw))

print(f"\nTotal: {len(X_all)} pairs")
print(f"  Certified: {len(X_cert)} (weight=1.0)")
print(f"  Raw:       {len(X_raw)} (weight=0.1 → updates each cycle)")

# Normalize Y
Y_mean = Y_all.mean(axis=0)
Y_std  = Y_all.std(axis=0) + 1e-8
Y_norm = (Y_all - Y_mean) / Y_std

# ── Dataset ───────────────────────────────────────────────────────────────────
class WeightedMotionDataset(Dataset):
    def __init__(self, X, Yq, Ym, weights):
        self.X = torch.FloatTensor(X)
        self.Yq = torch.LongTensor(Yq)
        self.Ym = torch.FloatTensor(Ym)
        self.W  = torch.FloatTensor(weights)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Yq[i], self.Ym[i], self.W[i]

# ── Model ─────────────────────────────────────────────────────────────────────
class DualHeadPhysicalAI(nn.Module):
    def __init__(self, n_feat, n_cls, n_mot):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 128),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),   nn.ReLU(),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_cls))
        self.motion_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_mot))
    def forward(self, x):
        e = self.encoder(x)
        return self.quality_head(e), self.motion_head(e)

device = "mps" if torch.backends.mps.is_available() else "cpu"
n_feat = X_all.shape[1]
model  = DualHeadPhysicalAI(n_feat, 2, Y_all.shape[1]).to(device)
print(f"\nDevice: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")

criterion_q = nn.CrossEntropyLoss(reduction="none")
criterion_m = nn.MSELoss(reduction="none")
optimizer   = optim.Adam(model.parameters(), lr=1e-3)

# ── Weighted training cycle ───────────────────────────────────────────────────
CYCLES = 5
EPOCHS_PER_CYCLE = 20
history = []

print(f"\nTraining: {CYCLES} cycles × {EPOCHS_PER_CYCLE} epochs")
print("="*60)

t0 = time.time()

for cycle in range(CYCLES):
    print(f"\n[Cycle {cycle+1}/{CYCLES}]")
    print(f"  Raw window weight: {W_all[~IS_CERTIFIED].mean():.3f}")

    # Build dataset with current weights
    dataset = WeightedMotionDataset(X_all, Q_all, Y_norm, W_all)

    # Weighted sampler — certified windows sampled more often
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(W_all),
        num_samples=min(len(X_all), 2000),
        replacement=True
    )
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Train for N epochs
    for epoch in range(EPOCHS_PER_CYCLE):
        model.train()
        total_loss = 0
        for xb, yq, ym, wb in loader:
            xb,yq,ym,wb = xb.to(device),yq.to(device),ym.to(device),wb.to(device)
            optimizer.zero_grad()
            pred_q, pred_m = model(xb)
            loss_q = (criterion_q(pred_q, yq) * wb).mean()
            loss_m = (criterion_m(pred_m, ym).mean(dim=1) * wb).mean()
            loss   = 0.5*loss_q + 0.5*loss_m
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Evaluate on certified val set
    model.eval()
    X_c_t = torch.FloatTensor(X_cert).to(device)
    with torch.no_grad():
        pq, pm = model(X_c_t)
    q_acc = accuracy_score(Q_cert, pq.argmax(1).cpu().numpy())

    # Predict smoothness on certified data
    pm_np = pm.cpu().numpy() * Y_std + Y_mean
    true_smooth = Y_cert[:, -1]
    pred_smooth = pm_np[:, -1]
    smooth_corr = float(np.corrcoef(pred_smooth, true_smooth)[0,1])

    # ── Update weights for raw data ──────────────────────────────────────────
    # Model scores every raw window — higher predicted quality = higher weight
    if len(X_raw) > 0:
        X_raw_t = torch.FloatTensor(X_raw).to(device)
        with torch.no_grad():
            pq_raw, _ = model(X_raw_t)
        # Softmax probability of being SILVER (class 1)
        probs = torch.softmax(pq_raw, dim=1)[:, 1].cpu().numpy()
        # New weight = min_weight + quality_prob * (1 - min_weight)
        new_raw_weights = 0.1 + probs * 0.9
        W_all[~IS_CERTIFIED] = new_raw_weights.astype(np.float32)

    cycle_result = {
        "cycle": cycle+1,
        "quality_acc": round(q_acc, 4),
        "smoothness_corr": round(smooth_corr, 3),
        "raw_weight_mean": round(float(W_all[~IS_CERTIFIED].mean()), 3),
        "raw_weight_max":  round(float(W_all[~IS_CERTIFIED].max()),  3),
    }
    history.append(cycle_result)
    print(f"  quality_acc={q_acc:.3f} | smoothness_corr={smooth_corr:.3f} | "
          f"raw_weight={W_all[~IS_CERTIFIED].mean():.3f}→{W_all[~IS_CERTIFIED].max():.3f}")

# ── Final results ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"Done in {elapsed:.0f}s")
print(f"\nCycle progression:")
print(f"{'Cycle':>6} {'Quality Acc':>12} {'Smooth Corr':>12} {'Raw Weight':>11}")
for r in history:
    print(f"{r['cycle']:>6} {r['quality_acc']:>12.3f} "
          f"{r['smoothness_corr']:>12.3f} {r['raw_weight_mean']:>11.3f}")

# How many raw windows got promoted to high quality?
promoted = sum(W_all[~IS_CERTIFIED] > 0.7)
print(f"\nRaw windows promoted (weight>0.7): {promoted}/{len(X_raw)}")
print(f"  = model found {promoted} raw windows that look like certified data")

torch.save(model.state_dict(),
           os.path.expanduser("~/S2S/experiments/level5_weighted_best.pt"))

results = {
    "experiment":    "Level 5 Weighted Semi-Supervised",
    "date":          time.strftime("%Y-%m-%d"),
    "certified_pairs": len(X_cert),
    "raw_pairs":     len(X_raw),
    "cycles":        CYCLES,
    "cycle_history": history,
    "final_quality_acc":    history[-1]["quality_acc"],
    "final_smoothness_corr": history[-1]["smoothness_corr"],
    "raw_promoted":  int(promoted),
    "raw_total":     len(X_raw),
    "description":   "Trains on all data, weights certified=1.0, raw=0.1→quality_score. Each cycle model re-evaluates raw data and promotes high-quality windows."
}

out = os.path.expanduser("~/S2S/experiments/results_level5_weighted.json")
with open(out,"w") as f: json.dump(results, f, indent=2)
print(f"\nResults → {out}")
