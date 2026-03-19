#!/usr/bin/env python3
"""
Level 5 — Dual-Head Physical AI
=================================
Head 1: Quality prediction (SILVER/BRONZE)
Head 2: Motion prediction (next pos, vel, smoothness)

Same encoder. Two outputs. One training pass.
"""

import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.expanduser("~/S2S"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ── Load data ─────────────────────────────────────────────────────────────────
data = np.load(os.path.expanduser("~/S2S/experiments/sequences_real.npz"),
               allow_pickle=True)
X       = data["X"].astype(np.float32)        # (692, 13)
Y_motion = data["Y"].astype(np.float32)       # (692, 8)
tiers   = data["tiers"]
sources = data["sources"]

le = LabelEncoder()
Y_quality = le.fit_transform(tiers).astype(np.int64)
print(f"Classes: {le.classes_}")
print(f"Pairs: {len(X)} | Features: {X.shape[1]} | Motion targets: {Y_motion.shape[1]}")
print(f"Sources: NinaPro={sum(sources==0)} Amputee={sum(sources==1)} RoboTurk={sum(sources==2)}")

# Normalize motion targets
Y_mean = Y_motion.mean(axis=0)
Y_std  = Y_motion.std(axis=0) + 1e-8
Y_norm = (Y_motion - Y_mean) / Y_std

# Train/val split
idx = np.arange(len(X))
tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                    stratify=Y_quality)

X_tr,  X_val  = X[tr_idx],        X[val_idx]
Yq_tr, Yq_val = Y_quality[tr_idx], Y_quality[val_idx]
Ym_tr, Ym_val = Y_norm[tr_idx],    Y_norm[val_idx]

# ── Dataset ───────────────────────────────────────────────────────────────────
class MotionDataset(Dataset):
    def __init__(self, X, Yq, Ym):
        self.X  = torch.FloatTensor(X)
        self.Yq = torch.LongTensor(Yq)
        self.Ym = torch.FloatTensor(Ym)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Yq[i], self.Ym[i]

train_loader = DataLoader(MotionDataset(X_tr, Yq_tr, Ym_tr),
                          batch_size=32, shuffle=True)
val_loader   = DataLoader(MotionDataset(X_val, Yq_val, Ym_val),
                          batch_size=32)

# ── Dual-Head Model ───────────────────────────────────────────────────────────
class DualHeadPhysicalAI(nn.Module):
    def __init__(self, n_features, n_classes, n_motion):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Head 1: quality classification
        self.quality_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
        # Head 2: motion prediction
        self.motion_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_motion)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return self.quality_head(encoded), self.motion_head(encoded)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nDevice: {device}")

model = DualHeadPhysicalAI(
    n_features=X.shape[1],
    n_classes=len(le.classes_),
    n_motion=Y_motion.shape[1]
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

# ── Training ──────────────────────────────────────────────────────────────────
criterion_quality = nn.CrossEntropyLoss()
criterion_motion  = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

EPOCHS = 100
ALPHA  = 0.5   # weight: 0.5 quality + 0.5 motion

history = {"train_loss": [], "val_loss": [],
           "val_quality_acc": [], "val_motion_mse": []}

best_val_loss = float("inf")
t0 = time.time()

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for xb, yq, ym in train_loader:
        xb, yq, ym = xb.to(device), yq.to(device), ym.to(device)
        optimizer.zero_grad()
        pred_q, pred_m = model(xb)
        loss = ALPHA * criterion_quality(pred_q, yq) + \
               (1-ALPHA) * criterion_motion(pred_m, ym)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss, q_preds, q_true, m_mse = 0, [], [], 0
    with torch.no_grad():
        for xb, yq, ym in val_loader:
            xb, yq, ym = xb.to(device), yq.to(device), ym.to(device)
            pred_q, pred_m = model(xb)
            loss = ALPHA * criterion_quality(pred_q, yq) + \
                   (1-ALPHA) * criterion_motion(pred_m, ym)
            val_loss  += loss.item()
            q_preds   += pred_q.argmax(1).cpu().tolist()
            q_true    += yq.cpu().tolist()
            m_mse     += criterion_motion(pred_m, ym).item()

    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    q_acc     = accuracy_score(q_true, q_preds)
    avg_mmse  = m_mse / len(val_loader)
    scheduler.step(avg_val)

    history["train_loss"].append(avg_train)
    history["val_loss"].append(avg_val)
    history["val_quality_acc"].append(q_acc)
    history["val_motion_mse"].append(avg_mmse)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(),
                   os.path.expanduser("~/S2S/experiments/level5_dualhead_best.pt"))

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | train={avg_train:.4f} "
              f"val={avg_val:.4f} | quality_acc={q_acc:.3f} "
              f"motion_mse={avg_mmse:.4f}")

elapsed = time.time() - t0

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\nTraining done in {elapsed:.0f}s")
print(f"Best val loss: {best_val_loss:.4f}")
print(f"Final quality accuracy: {history['val_quality_acc'][-1]:.3f}")
print(f"Final motion MSE: {history['val_motion_mse'][-1]:.4f}")

# Smoothness improvement metric
final_smoothness_pred = []
model.eval()
with torch.no_grad():
    for xb, yq, ym in val_loader:
        _, pred_m = model(xb.to(device))
        final_smoothness_pred += pred_m.cpu()[:, -1].tolist()

pred_smooth = np.array(final_smoothness_pred) * Y_std[-1] + Y_mean[-1]
true_smooth = Y_motion[val_idx, -1]
print(f"\nSmoothing prediction correlation: "
      f"{np.corrcoef(pred_smooth, true_smooth)[0,1]:.3f}")

results = {
    "experiment": "Level 5 Dual-Head Physical AI",
    "date": time.strftime("%Y-%m-%d"),
    "n_pairs": len(X),
    "n_features": int(X.shape[1]),
    "n_motion_targets": int(Y_motion.shape[1]),
    "motion_targets": ["pos_x","pos_y","pos_z",
                       "vel_x","vel_y","vel_z",
                       "jerk_rms","smoothness"],
    "sources": {"NinaPro": int(sum(sources==0)),
                "Amputee": int(sum(sources==1)),
                "RoboTurk": int(sum(sources==2))},
    "epochs": EPOCHS,
    "device": device,
    "best_val_loss": round(best_val_loss, 4),
    "final_quality_acc": round(history["val_quality_acc"][-1], 4),
    "final_motion_mse": round(history["val_motion_mse"][-1], 4),
    "smoothness_correlation": round(
        float(np.corrcoef(pred_smooth, true_smooth)[0,1]), 3),
    "level5_dualhead_proven": history["val_quality_acc"][-1] > 0.70
}

out = os.path.expanduser("~/S2S/experiments/results_level5_dualhead.json")
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults → {out}")
