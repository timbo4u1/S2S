#!/usr/bin/env python3
"""
Level 5 — Full System
======================
1. All data sources (NinaPro + Amputee + RoboTurk)
2. 20 cycles weighted semi-supervised
3. Command encoder — "pick up cup" → best certified windows
4. Saves command→motion map for animation demo
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
from sklearn.metrics import accuracy_score

WINDOW_NINAPRO  = 2000
WINDOW_AMPUTEE  = 200
WINDOW_ROBOTURK = 30
NYU_COUNT       = 14

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

def motion_target(accel, hz):
    accel = np.array(accel, dtype=np.float32)
    if accel.ndim == 1: accel = accel.reshape(-1,1)
    while accel.shape[1] < 3: accel = np.hstack([accel, np.zeros((len(accel),1))])
    vel   = np.cumsum(accel, axis=0) / hz
    pos   = np.cumsum(vel,   axis=0) / hz
    jerk  = np.diff(accel, axis=0) * hz if len(accel) > 1 else np.zeros_like(accel)
    smoothness = float(1.0 / (1.0 + np.sqrt(np.mean(jerk**2) + 1e-8)))
    speed = float(np.sqrt((vel**2).sum(axis=1)).mean())
    return (pos.mean(axis=0).tolist() +
            vel[-1].tolist() +
            [float(np.sqrt(np.mean(jerk**2) + 1e-8)), smoothness, speed])

engine = PhysicsEngine()

def certify(accel, gyro, hz):
    n  = len(accel)
    ts = [int(i * 1e9 / hz) for i in range(n)]
    r  = engine.certify(
        imu_raw={"timestamps_ns": ts,
                 "accel": accel.tolist(),
                 "gyro":  gyro.tolist()},
        segment="forearm"
    )
    return r.get("physical_law_score", 0), r.get("tier", "REJECTED")

# ── CERTIFIED DATA ────────────────────────────────────────────────────────────
print("="*60)
print("LOADING ALL DATA")
print("="*60)

X_cert, Y_cert, Q_cert, LABELS_cert = [], [], [], []

# NinaPro
print("\n[1/3] NinaPro DB5 (certified)...")
count = 0
for subj_dir in sorted(glob.glob(os.path.expanduser("~/ninapro_db5/s*"))):
    accel_all = []
    for mat_path in sorted(glob.glob(os.path.join(subj_dir,"*.mat"))):
        try:
            mat = scipy.io.loadmat(mat_path)
            acc = mat.get("acc", mat.get("ACC"))
            if acc is not None: accel_all.append(acc)
        except: continue
    if not accel_all: continue
    accel = np.vstack(accel_all).astype(np.float64)
    gyro  = np.zeros_like(accel)
    chunks = [(accel[i:i+WINDOW_NINAPRO,:3], gyro[i:i+WINDOW_NINAPRO,:3])
              for i in range(0, len(accel)-WINDOW_NINAPRO*2, WINDOW_NINAPRO)]
    results = []
    for ca, cg in chunks:
        score, tier = certify(ca, cg, 2000)
        results.append((score, tier, ca, cg))
    for i in range(len(results)-1):
        s_i, t_i, ca_i, cg_i = results[i]
        s_j, t_j, ca_j, cg_j = results[i+1]
        if t_i in ("SILVER","BRONZE","GOLD") and t_j in ("SILVER","BRONZE","GOLD"):
            X_cert.append(extract_features(ca_i, cg_i, 2000))
            Y_cert.append(motion_target(ca_j, 2000))
            Q_cert.append(1 if t_i=="SILVER" else 0)
            LABELS_cert.append("ninapro_hand_motion")
            count += 1
print(f"  → {count} pairs")

# EMG Amputee
print("\n[2/3] EMG Amputee (certified)...")
count = 0
for subj in sorted(os.listdir(os.path.expanduser("~/S2S_Project/EMG_Amputee"))):
    inner = os.path.expanduser(f"~/S2S_Project/EMG_Amputee/{subj}/{subj}")
    if not os.path.isdir(inner): continue
    for csv_path in sorted(glob.glob(os.path.join(inner,"mov *","accelerometer-*.csv"))):
        try:
            data  = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            if data.ndim < 2 or data.shape[1] < 4: continue
            accel = data[:,1:4].astype(np.float64)
            gyro  = np.zeros_like(accel)
            chunks = [(accel[i:i+WINDOW_AMPUTEE], gyro[i:i+WINDOW_AMPUTEE])
                      for i in range(0, len(accel)-WINDOW_AMPUTEE*2, WINDOW_AMPUTEE)]
            results = []
            for ca, cg in chunks:
                score, tier = certify(ca, cg, 200)
                results.append((score, tier, ca, cg))
            for i in range(len(results)-1):
                s_i,t_i,ca_i,cg_i = results[i]
                s_j,t_j,ca_j,cg_j = results[i+1]
                if t_i in ("SILVER","BRONZE","GOLD") and t_j in ("SILVER","BRONZE","GOLD"):
                    X_cert.append(extract_features(ca_i,cg_i,200))
                    Y_cert.append(motion_target(ca_j,200))
                    Q_cert.append(1 if t_i=="SILVER" else 0)
                    LABELS_cert.append("amputee_hand_motion")
                    count += 1
        except: continue
print(f"  → {count} pairs")

# RoboTurk certified
print("\n[3/3] RoboTurk (certified + raw)...")
count_cert, count_raw = 0, 0
X_raw, Y_raw, Q_raw, LABELS_raw, INSTRS_raw = [], [], [], [], []

files = sorted(glob.glob(os.path.expanduser(
    "~/S2S/openx_data/sample_*.data.pickle")))[NYU_COUNT:]

for path in files:
    with open(path,"rb") as f:
        data = pickle.load(f)
    steps = data.get("steps",[])
    instr = "unknown"
    for step in steps:
        if not isinstance(step, dict): continue
        raw_instr = step.get("observation",{}).get("natural_language_instruction","")
        if raw_instr:
            instr = raw_instr.decode() if isinstance(raw_instr,bytes) else str(raw_instr)
            break

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

    results = []
    for ca, cg in chunks:
        if len(ca) < 5:
            results.append((0,"REJECTED",ca,cg))
            continue
        score, tier = certify(ca, cg, 15)
        results.append((score, tier, ca, cg))

    for i in range(len(results)-1):
        s_i,t_i,ca_i,cg_i = results[i]
        s_j,t_j,ca_j,cg_j = results[i+1]
        feats  = extract_features(ca_i,cg_i,15)
        target = motion_target(ca_j,15)
        if t_i in ("SILVER","BRONZE","GOLD") and t_j in ("SILVER","BRONZE","GOLD"):
            X_cert.append(feats)
            Y_cert.append(target)
            Q_cert.append(1 if t_i=="SILVER" else 0)
            LABELS_cert.append(instr)
            count_cert += 1
        elif t_i != "REJECTED":
            X_raw.append(feats)
            Y_raw.append(target)
            Q_raw.append(0)
            LABELS_raw.append(instr)
            INSTRS_raw.append(instr)
            count_raw += 1

print(f"  → certified: {count_cert} | raw: {count_raw}")

# ── Combine ───────────────────────────────────────────────────────────────────
X_cert = np.array(X_cert, dtype=np.float32)
Y_cert = np.array(Y_cert, dtype=np.float32)
Q_cert = np.array(Q_cert, dtype=np.int64)

X_raw  = np.array(X_raw,  dtype=np.float32) if X_raw else np.zeros((0,X_cert.shape[1]),dtype=np.float32)
Y_raw  = np.array(Y_raw,  dtype=np.float32) if Y_raw else np.zeros((0,Y_cert.shape[1]),dtype=np.float32)
Q_raw  = np.array(Q_raw,  dtype=np.int64)

X_all  = np.vstack([X_cert, X_raw])
Y_all  = np.vstack([Y_cert, Y_raw])
Q_all  = np.concatenate([Q_cert, Q_raw])
W_all  = np.concatenate([
    np.ones(len(X_cert), dtype=np.float32),
    np.full(len(X_raw), 0.1, dtype=np.float32)
])
IS_CERT = np.array([True]*len(X_cert) + [False]*len(X_raw))

Y_mean = Y_all.mean(axis=0)
Y_std  = Y_all.std(axis=0) + 1e-8
Y_norm = (Y_all - Y_mean) / Y_std

print(f"\nTOTAL: {len(X_all)} pairs")
print(f"  Certified: {len(X_cert)}")
print(f"  Raw:       {len(X_raw)}")
print(f"  Features:  {X_all.shape[1]}")
print(f"  Motion targets: {Y_all.shape[1]}")

# ── Model ─────────────────────────────────────────────────────────────────────
class DualHeadAI(nn.Module):
    def __init__(self, n_feat, n_cls, n_mot):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_feat,128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,256),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128),   nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128,64),    nn.ReLU(),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(), nn.Linear(32,n_cls))
        self.motion_head = nn.Sequential(
            nn.Linear(64,32), nn.ReLU(), nn.Linear(32,n_mot))
    def forward(self, x):
        e = self.encoder(x)
        return self.quality_head(e), self.motion_head(e)
    def encode(self, x):
        return self.encoder(x)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model  = DualHeadAI(X_all.shape[1], 2, Y_all.shape[1]).to(device)
print(f"\nDevice: {device} | Params: {sum(p.numel() for p in model.parameters()):,}")

crit_q = nn.CrossEntropyLoss(reduction="none")
crit_m = nn.MSELoss(reduction="none")
opt    = optim.Adam(model.parameters(), lr=1e-3)
sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400)

class WDS(Dataset):
    def __init__(self, X, Yq, Ym, W):
        self.X=torch.FloatTensor(X); self.Yq=torch.LongTensor(Yq)
        self.Ym=torch.FloatTensor(Ym); self.W=torch.FloatTensor(W)
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i],self.Yq[i],self.Ym[i],self.W[i]

# ── Training Cycles ───────────────────────────────────────────────────────────
CYCLES           = 20
EPOCHS_PER_CYCLE = 20
history          = []
best_smooth      = 0.0

print(f"\nTraining: {CYCLES} cycles × {EPOCHS_PER_CYCLE} epochs = {CYCLES*EPOCHS_PER_CYCLE} total")
print("="*60)
t0 = time.time()

for cycle in range(CYCLES):
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(W_all),
        num_samples=min(len(X_all), 3000),
        replacement=True
    )
    loader = DataLoader(WDS(X_all,Q_all,Y_norm,W_all),
                        batch_size=64, sampler=sampler)
    model.train()
    for epoch in range(EPOCHS_PER_CYCLE):
        for xb,yq,ym,wb in loader:
            xb,yq,ym,wb = xb.to(device),yq.to(device),ym.to(device),wb.to(device)
            opt.zero_grad()
            pq,pm = model(xb)
            loss  = (0.4*crit_q(pq,yq) + 0.6*crit_m(pm,ym).mean(1)) * wb
            loss.mean().backward()
            opt.step()
            sched.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        pq,pm = model(torch.FloatTensor(X_cert).to(device))
    q_acc  = accuracy_score(Q_cert, pq.argmax(1).cpu().numpy())
    pm_np  = pm.cpu().numpy() * Y_std + Y_mean
    smooth_corr = float(np.corrcoef(pm_np[:,-2], Y_cert[:,-2])[0,1])
    speed_corr  = float(np.corrcoef(pm_np[:,-1], Y_cert[:,-1])[0,1])

    # Update raw weights
    if len(X_raw) > 0:
        with torch.no_grad():
            pq_r,_ = model(torch.FloatTensor(X_raw).to(device))
        probs = torch.softmax(pq_r,dim=1)[:,1].cpu().numpy()
        W_all[~IS_CERT] = (0.1 + probs * 0.9).astype(np.float32)

    promoted = int(sum(W_all[~IS_CERT] > 0.7)) if len(X_raw) > 0 else 0

    h = {"cycle":cycle+1, "quality_acc":round(q_acc,4),
         "smoothness_corr":round(smooth_corr,3),
         "speed_corr":round(speed_corr,3),
         "raw_weight_mean":round(float(W_all[~IS_CERT].mean()),3) if len(X_raw)>0 else 1.0,
         "promoted":promoted}
    history.append(h)

    if smooth_corr > best_smooth:
        best_smooth = smooth_corr
        torch.save(model.state_dict(),
                   os.path.expanduser("~/S2S/experiments/level5_full_best.pt"))

    if (cycle+1) % 5 == 0 or cycle == 0:
        print(f"Cycle {cycle+1:2d} | quality={q_acc:.3f} smooth={smooth_corr:.3f} "
              f"speed={speed_corr:.3f} | promoted={promoted}/{len(X_raw)}")

# ── Command → Motion Map ──────────────────────────────────────────────────────
print("\nBuilding command → motion map...")
model.eval()

# Get all unique commands from RoboTurk
all_labels = LABELS_cert + LABELS_raw
unique_commands = list(set(
    l for l in all_labels
    if l not in ("ninapro_hand_motion","amputee_hand_motion","unknown")
))
print(f"Commands found: {unique_commands}")

command_map = {}
for cmd in unique_commands:
    # Find certified windows with this command label
    cert_indices = [i for i,l in enumerate(LABELS_cert) if l==cmd]
    if not cert_indices: continue

    cmd_X = X_cert[cert_indices]
    cmd_Y = Y_cert[cert_indices]

    with torch.no_grad():
        _, pm = model(torch.FloatTensor(cmd_X).to(device))
    pm_np = pm.cpu().numpy() * Y_std + Y_mean

    # Best windows = highest predicted smoothness
    smooth_scores = pm_np[:, -2]
    top_idx = np.argsort(smooth_scores)[-10:][::-1]

    command_map[cmd] = {
        "n_examples": len(cert_indices),
        "avg_smoothness": round(float(smooth_scores.mean()), 3),
        "best_smoothness": round(float(smooth_scores[top_idx[0]]), 3),
        "best_window_features": cmd_X[top_idx[0]].tolist(),
        "predicted_motion": {
            "pos_x": round(float(pm_np[top_idx[0],0]),4),
            "pos_y": round(float(pm_np[top_idx[0],1]),4),
            "pos_z": round(float(pm_np[top_idx[0],2]),4),
            "vel_x": round(float(pm_np[top_idx[0],3]),4),
            "vel_y": round(float(pm_np[top_idx[0],4]),4),
            "vel_z": round(float(pm_np[top_idx[0],5]),4),
            "jerk_rms":  round(float(pm_np[top_idx[0],6]),4),
            "smoothness":round(float(pm_np[top_idx[0],7]),4),
            "speed":     round(float(pm_np[top_idx[0],8]),4),
        },
        "top10_smoothness": [round(float(s),3) for s in smooth_scores[top_idx]]
    }
    print(f"  '{cmd}': {len(cert_indices)} examples | "
          f"best smoothness={command_map[cmd]['best_smoothness']}")

# ── Final results ─────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"Done in {elapsed:.0f}s")
print(f"\nProgression (every 5 cycles):")
print(f"{'Cycle':>6} {'Quality':>9} {'Smooth':>8} {'Speed':>8} {'Promoted':>10}")
for r in history[::5]:
    print(f"{r['cycle']:>6} {r['quality_acc']:>9.3f} "
          f"{r['smoothness_corr']:>8.3f} {r['speed_corr']:>8.3f} "
          f"{r['promoted']:>8}/{len(X_raw)}")

results = {
    "experiment": "Level 5 Full System",
    "date": time.strftime("%Y-%m-%d"),
    "data": {
        "certified_pairs": len(X_cert),
        "raw_pairs":       len(X_raw),
        "total_pairs":     len(X_all),
        "features":        int(X_all.shape[1]),
        "motion_targets":  int(Y_all.shape[1]),
    },
    "training": {
        "cycles":          CYCLES,
        "epochs_per_cycle":EPOCHS_PER_CYCLE,
        "total_epochs":    CYCLES * EPOCHS_PER_CYCLE,
        "elapsed_s":       round(elapsed,1),
    },
    "final": {
        "quality_acc":      history[-1]["quality_acc"],
        "smoothness_corr":  history[-1]["smoothness_corr"],
        "speed_corr":       history[-1]["speed_corr"],
        "best_smoothness":  round(best_smooth,3),
        "raw_promoted":     history[-1]["promoted"],
        "raw_total":        len(X_raw),
    },
    "commands": list(command_map.keys()),
    "cycle_history": history,
}

# Save results
out_results = os.path.expanduser("~/S2S/experiments/results_level5_full.json")
with open(out_results,"w") as f: json.dump(results, f, indent=2)

# Save command map for animation demo
out_cmds = os.path.expanduser("~/S2S/experiments/command_motion_map.json")
with open(out_cmds,"w") as f: json.dump(command_map, f, indent=2)

print(f"\nResults      → {out_results}")
print(f"Command map  → {out_cmds}")
print(f"\nCommands ready for animation:")
for cmd, v in command_map.items():
    print(f"  '{cmd}' → smoothness={v['best_smoothness']} speed={v['predicted_motion']['speed']}")
