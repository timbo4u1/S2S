#!/usr/bin/env python3
"""
S2S Level 5 — Cross-Dataset Transfer Test

Train CNN on PTT-PPG IMU (accel+gyro, 6 channels).
Test zero-shot on UCI HAR — no retraining, no UCI HAR data seen during training.

If the CNN agrees with the rule engine on UCI HAR without retraining:
  → physics knowledge transferred from PTT-PPG to UCI HAR
  → the CNN learned physics, not just dataset-specific patterns
  → rule engine can be replaced at inference time

Run from ~/S2S:
  python3 experiments/experiment_level5_transfer.py \
    --ptt  ~/physionet.org/files/pulse-transit-time-ppg/1.1.0/ \
    --uci  "data/uci_har/UCI HAR Dataset/" \
    --out  experiments/results_level5_transfer.json
"""

import os, sys, json, math, random, time, argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wfdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# PTT-PPG settings
PTT_HZ       = 500
PTT_WINDOW   = 256
PTT_STEP     = 1000
PTT_ACCEL    = [12, 13, 14]   # a_x, a_y, a_z
PTT_GYRO     = [15, 16, 17]   # g_x, g_y, g_z
PTT_IMU_CH   = PTT_ACCEL + PTT_GYRO   # 6 channels
PTT_DT_NS    = int(1e9 / PTT_HZ)

PTT_TRAIN    = [f"s{i}" for i in range(1, 17)]
PTT_TEST     = [f"s{i}" for i in range(17, 23)]
ACTIVITIES   = ["walk", "sit", "run"]

# UCI HAR settings
UCI_HZ       = 50.0
UCI_WINDOW   = 128
UCI_DT_NS    = int(1e9 / UCI_HZ)

N_CHANNELS   = 6   # accel xyz + gyro xyz — same for both datasets
TIER_MAP     = {"GOLD": 0, "SILVER": 0, "BRONZE": 0, "REJECTED": 1}
TIER_NAMES   = ["PASS", "REJECT"]

# ── PTT-PPG LOADER (IMU only) ─────────────────────────────────────────────────

def load_ptt_ppg_imu(data_dir):
    base    = Path(data_dir)
    windows = []
    for subj in PTT_TRAIN + PTT_TEST:
        for act in ACTIVITIES:
            rec_path = base / f"{subj}_{act}"
            if not (base / f"{subj}_{act}.hea").exists():
                continue
            try:
                record = wfdb.rdrecord(str(rec_path))
                sig    = record.p_signal
                if sig is None: continue
                n = sig.shape[0]
                for start in range(0, n - PTT_WINDOW, PTT_STEP):
                    chunk = sig[start:start+PTT_WINDOW, :]
                    raw   = chunk[:, PTT_IMU_CH].astype(np.float32)
                    if np.isnan(raw).any(): continue
                    accel = raw[:, 0:3].tolist()
                    gyro  = raw[:, 3:6].tolist()
                    ts_ns = [int(j*PTT_DT_NS) + int(random.gauss(0,300))
                             for j in range(PTT_WINDOW)]
                    windows.append({
                        "raw":        raw.tolist(),
                        "accel":      accel,
                        "gyro":       gyro,
                        "timestamps": ts_ns,
                        "subject":    subj,
                        "activity":   act,
                        "split":      "train" if subj in PTT_TRAIN else "test",
                        "dataset":    "ptt_ppg",
                    })
            except Exception:
                continue
    print(f"  PTT-PPG IMU: {len(windows)} windows from "
          f"{len(set(w['subject'] for w in windows))} subjects")
    return windows

# ── UCI HAR LOADER ────────────────────────────────────────────────────────────

def load_signal_file(path):
    windows = []
    with open(path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if vals: windows.append(vals)
    return windows

def load_uci_har(data_dir):
    base    = Path(data_dir)
    windows = []
    for split in ["train", "test"]:
        sig_dir  = base / split / "Inertial Signals"
        subjects = [int(x) for x in (base/split/f"subject_{split}.txt").read_text().split()]
        labels   = [int(x) for x in (base/split/f"y_{split}.txt").read_text().split()]
        bax = load_signal_file(sig_dir/f"body_acc_x_{split}.txt")
        bay = load_signal_file(sig_dir/f"body_acc_y_{split}.txt")
        baz = load_signal_file(sig_dir/f"body_acc_z_{split}.txt")
        bgx = load_signal_file(sig_dir/f"body_gyro_x_{split}.txt")
        bgy = load_signal_file(sig_dir/f"body_gyro_y_{split}.txt")
        bgz = load_signal_file(sig_dir/f"body_gyro_z_{split}.txt")
        for i in range(len(labels)):
            accel = [[bax[i][j], bay[i][j], baz[i][j]] for j in range(UCI_WINDOW)]
            gyro  = [[bgx[i][j], bgy[i][j], bgz[i][j]] for j in range(UCI_WINDOW)]
            raw   = [[accel[t][0], accel[t][1], accel[t][2],
                      gyro[t][0],  gyro[t][1],  gyro[t][2]]
                     for t in range(UCI_WINDOW)]
            ts_ns = [int(j*UCI_DT_NS)+int(random.gauss(0,300))
                     for j in range(UCI_WINDOW)]
            windows.append({
                "raw":        raw,
                "accel":      accel,
                "gyro":       gyro,
                "timestamps": ts_ns,
                "subject":    subjects[i],
                "label":      labels[i]-1,
                "split":      split,
                "dataset":    "uci_har",
            })
    print(f"  UCI HAR: {len(windows)} windows from "
          f"{len(set(w['subject'] for w in windows))} subjects")
    return windows

# ── S2S CERTIFICATION ─────────────────────────────────────────────────────────

def certify_window(w):
    try:
        from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
        result = PhysicsEngine().certify(
            imu_raw={"timestamps_ns": w["timestamps"],
                     "accel": w["accel"], "gyro": w["gyro"]},
            segment="forearm")
        score = result.get("physical_law_score", result.get("score", 0)) or 0
        tier  = result.get("tier", "BRONZE")
        return int(score), tier
    except Exception:
        return 0, "REJECTED"

# ── DATASET ───────────────────────────────────────────────────────────────────

def get_norm_stats(windows_with_tiers, window_size):
    all_raw = np.array([[r for r in w["raw"]]
                        for w, _ in windows_with_tiers], dtype=np.float32)
    # Pad or trim to window_size
    if all_raw.shape[1] != window_size:
        all_raw = all_raw[:, :window_size, :]
    means = all_raw.mean(axis=(0,1))
    stds  = all_raw.std(axis=(0,1)) + 1e-8
    return means.astype(np.float32), stds.astype(np.float32)

class IMUDataset(Dataset):
    def __init__(self, windows_with_tiers, means, stds, window_size):
        self.X = []
        self.y = []
        for w, tier in windows_with_tiers:
            raw = np.array(w["raw"], dtype=np.float32)
            # Resize to window_size
            if raw.shape[0] > window_size:
                raw = raw[:window_size]
            elif raw.shape[0] < window_size:
                pad = np.zeros((window_size - raw.shape[0], raw.shape[1]),
                               dtype=np.float32)
                raw = np.vstack([raw, pad])
            # Per-window normalization — device independent
            w_mean = raw.mean(axis=0)
            w_std  = raw.std(axis=0) + 1e-8
            raw = (raw - w_mean) / w_std
            # (window_size, 6) → (6, window_size)
            x = torch.from_numpy(raw.T.astype(np.float32))
            self.X.append(x)
            self.y.append(TIER_MAP[tier])

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ── 1D CNN ────────────────────────────────────────────────────────────────────

class IMUCertifierCNN(nn.Module):
    """6-channel IMU CNN — works on any IMU dataset at any Hz."""
    def __init__(self, n_channels=6, n_classes=2, window_size=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # works for any input length
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 32),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x): return self.classifier(self.encoder(x))

# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_cnn(model, train_dl, val_dl, epochs, device):
    weights   = torch.tensor([4.0, 1.0]).to(device)
    criterion = nn.NLLLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(device)
    best_f1 = 0; best_state = None

    for e in range(1, epochs+1):
        model.train()
        total_loss = 0
        for X, y in train_dl:
            X = X.to(device)
            y = torch.tensor(list(y), dtype=torch.long).to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if e % 10 == 0 or e == epochs:
            acc, f1, rr = evaluate(model, val_dl, device)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.clone() for k,v in model.state_dict().items()}
            print(f"  ep {e:3d}/{epochs}  loss={total_loss/len(train_dl):.4f}"
                  f"  acc={acc:.3f}  f1={f1:.4f}  reject_recall={rr:.3f}")
    if best_state: model.load_state_dict(best_state)
    return model

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            p = model(X).argmax(dim=1).cpu().numpy()
            preds.extend(p)
            labels.extend(list(y.numpy() if hasattr(y,'numpy') else y))
    tp=defaultdict(int); fp=defaultdict(int); fn=defaultdict(int)
    correct=0
    cm=[[0,0],[0,0]]
    for p,l in zip(preds,labels):
        p=int(p); l=int(l)
        cm[l][p]+=1
        if p==l: correct+=1; tp[l]+=1
        else: fp[p]+=1; fn[l]+=1
    acc=correct/len(labels)
    f1s=[]
    for c in range(2):
        pr=tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0
        rc=tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0
        f1s.append(2*pr*rc/(pr+rc) if pr+rc else 0)
    rr=tp[1]/(tp[1]+fn[1]) if tp[1]+fn[1] else 0
    return acc, sum(f1s)/2, rr

def zero_shot_agreement(model, windows_with_tiers, means, stds,
                        window_size, device):
    """Run CNN on new dataset — no retraining, no labels used."""
    model.eval()
    agree=0; total=0
    tier_agr=defaultdict(lambda:{"agree":0,"total":0})
    with torch.no_grad():
        for w, rule_tier in windows_with_tiers:
            raw = np.array(w["raw"], dtype=np.float32)
            if raw.shape[0] > window_size: raw=raw[:window_size]
            elif raw.shape[0] < window_size:
                pad=np.zeros((window_size-raw.shape[0],raw.shape[1]),dtype=np.float32)
                raw=np.vstack([raw,pad])
            # Per-window normalization — removes device amplitude differences
            w_mean = raw.mean(axis=0)
            w_std  = raw.std(axis=0) + 1e-8
            raw = (raw - w_mean) / w_std
            x   = torch.from_numpy(raw.T.astype(np.float32)).unsqueeze(0).to(device)
            pred= TIER_NAMES[model(x).argmax(dim=1).item()]
            rule_binary = "REJECT" if rule_tier=="REJECTED" else "PASS"
            if pred==rule_binary:
                agree+=1
                tier_agr[rule_tier]["agree"]+=1
            tier_agr[rule_tier]["total"]+=1
            total+=1
    overall = agree/total if total else 0
    per_tier= {t: round(v["agree"]/v["total"],4) if v["total"] else 0
               for t,v in tier_agr.items()}
    return overall, per_tier

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ptt",    required=True)
    p.add_argument("--uci",    required=True)
    p.add_argument("--out",    required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch",  type=int, default=32)
    args = p.parse_args()

    device = "mps"  if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    print("\nS2S Level 5 — Cross-Dataset Transfer Test")
    print("=" * 60)
    print(f"  Train: PTT-PPG IMU (subjects s1-s16, 500Hz)")
    print(f"  Test A: PTT-PPG IMU (subjects s17-s22, never seen)")
    print(f"  Test B: UCI HAR     (all subjects, zero-shot, 50Hz)")
    print(f"  Channels: 6 (accel xyz + gyro xyz)")
    print(f"  Device: {device}")
    print(f"\n  Key question: does physics knowledge transfer")
    print(f"  from PTT-PPG to UCI HAR with zero retraining?")

    # ── Load PTT-PPG
    print("\n── Step 1: Load PTT-PPG IMU ─────────────────────────────")
    ptt_wins = load_ptt_ppg_imu(args.ptt)
    ptt_train = [w for w in ptt_wins if w["split"]=="train"]
    ptt_test  = [w for w in ptt_wins if w["split"]=="test"]

    # ── Certify PTT-PPG
    print(f"\n── Step 2: Certify PTT-PPG with rule engine ─────────────")
    t0 = time.time()
    ptt_train_cert = []
    for k, w in enumerate(ptt_train, 1):
        s, tier = certify_window(w)
        ptt_train_cert.append((w, tier))
        if k % 1000 == 0:
            print(f"  [{k}/{len(ptt_train)}] elapsed={int(time.time()-t0)}s")

    ptt_test_cert = [(w, certify_window(w)[1]) for w in ptt_test]
    print(f"  Train tiers: {dict(Counter(t for _,t in ptt_train_cert))}")
    print(f"  Test  tiers: {dict(Counter(t for _,t in ptt_test_cert))}")

    # ── Inject corruptions
    augmented = []
    for w, tier in ptt_train_cert:
        if random.random() < 0.15:
            wc  = dict(w)
            raw = [row[:] for row in w["raw"]]
            ct  = random.randint(0, 2)
            if ct == 0:
                for t in range(len(raw)):
                    raw[t] = [max(-0.3, min(0.3, v)) for v in raw[t]]
            elif ct == 1:
                off = [random.uniform(3, 8) for _ in range(6)]
                for t in range(len(raw)):
                    raw[t] = [v+off[i] for i,v in enumerate(raw[t])]
            else:
                for ax in range(6):
                    val = raw[0][ax]
                    for t in range(len(raw)): raw[t][ax] = val
            wc["raw"] = raw
            augmented.append((wc, "REJECTED"))
        else:
            augmented.append((w, tier))
    ptt_train_cert = augmented
    print(f"  After augment: {dict(Counter(t for _,t in ptt_train_cert))}")

    # ── Normalize from PTT-PPG training data
    means, stds = get_norm_stats(ptt_train_cert, PTT_WINDOW)

    train_ds = IMUDataset(ptt_train_cert, means, stds, PTT_WINDOW)
    val_ds   = IMUDataset(ptt_test_cert,  means, stds, PTT_WINDOW)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    # ── Train CNN on PTT-PPG IMU
    print(f"\n── Step 3: Train CNN on PTT-PPG IMU ─────────────────────")
    n_params = sum(p.numel() for p in IMUCertifierCNN().parameters())
    print(f"  Parameters: {n_params:,}")
    t0    = time.time()
    model = IMUCertifierCNN(window_size=PTT_WINDOW)
    model = train_cnn(model, train_dl, val_dl, args.epochs, device)
    train_time = round(time.time()-t0)

    # ── Evaluate on PTT-PPG test
    print(f"\n── Step 4: Evaluate on PTT-PPG s17-s22 (never seen) ─────")
    acc_ptt, f1_ptt, rr_ptt = evaluate(model, val_dl, device)
    agr_ptt, per_ptt = zero_shot_agreement(
        model, ptt_test_cert, means, stds, PTT_WINDOW, device)
    print(f"  PTT-PPG test acc={acc_ptt:.3f}  f1={f1_ptt:.4f}  "
          f"reject_recall={rr_ptt:.3f}  agreement={agr_ptt*100:.1f}%")

    # ── Load and certify UCI HAR (zero-shot target)
    print(f"\n── Step 5: Zero-shot test on UCI HAR ────────────────────")
    print(f"  Loading UCI HAR...")
    uci_wins = load_uci_har(args.uci)
    print(f"  Certifying UCI HAR with rule engine...")
    t0 = time.time()
    uci_cert = []
    for k, w in enumerate(uci_wins, 1):
        s, tier = certify_window(w)
        uci_cert.append((w, tier))
        if k % 2000 == 0:
            print(f"  [{k}/{len(uci_wins)}] elapsed={int(time.time()-t0)}s")
    print(f"  UCI HAR tiers: {dict(Counter(t for _,t in uci_cert))}")

    # ── Resample UCI HAR 50Hz → 500Hz to match PTT-PPG training domain
    print(f"  Resampling UCI HAR 50Hz → 500Hz...")
    from scipy import signal as scipy_signal
    resampled_cert = []
    for w, tier in uci_cert:
        raw = np.array(w["raw"], dtype=np.float32)  # (128, 6)
        # Resample each channel from 128 samples to 1280 samples (×10)
        raw_up = scipy_signal.resample(raw, 1280, axis=0).astype(np.float32)
        # Take first 256 samples (0.512s — same as PTT-PPG window)
        raw_256 = raw_up[:256]
        wc = dict(w)
        wc["raw"] = raw_256.tolist()
        resampled_cert.append((wc, tier))
    uci_cert = resampled_cert
    print(f"  Resampled: {len(uci_cert)} windows, shape now 256×6")

    # ── Zero-shot agreement — CNN never saw UCI HAR
    print(f"\n  Running CNN zero-shot on UCI HAR (no retraining)...")
    agr_uci, per_uci = zero_shot_agreement(
        model, uci_cert, means, stds, UCI_WINDOW, device)

    # Also evaluate with PyTorch loader for F1
    uci_ds = IMUDataset(uci_cert, means, stds, UCI_WINDOW)
    uci_dl = DataLoader(uci_ds, batch_size=64, shuffle=False)
    acc_uci, f1_uci, rr_uci = evaluate(model, uci_dl, device)

    # ── Results
    print(f"\n{'='*60}")
    print(f"  S2S LEVEL 5 — CROSS-DATASET TRANSFER RESULTS")
    print(f"{'='*60}")
    proven = agr_uci >= 0.65 and rr_uci >= 0.40

    print(f"\n  ┌─ PTT-PPG → PTT-PPG (same domain, unseen subjects) ────")
    print(f"  │  Agreement:     {agr_ptt*100:.1f}%")
    print(f"  │  REJECT recall: {rr_ptt*100:.1f}%")
    print(f"  │  Macro F1:      {f1_ptt:.4f}")
    print(f"  ├─ PTT-PPG → UCI HAR (cross-domain, zero-shot) ─────────")
    print(f"  │  Agreement:     {agr_uci*100:.1f}%")
    print(f"  │  REJECT recall: {rr_uci*100:.1f}%")
    print(f"  │  Macro F1:      {f1_uci:.4f}")
    print(f"  │")
    print(f"  │  CNN trained on PTT-PPG 500Hz")
    print(f"  │  Tested on UCI HAR 50Hz — different device,")
    print(f"  │  different Hz, different subjects, zero retraining.")
    print(f"  │")
    if proven:
        print(f"  │  ✓ Physics knowledge transferred.")
        print(f"  │  The CNN learned physics, not dataset patterns.")
        print(f"  │  Rule engine can be replaced at inference time.")
    else:
        print(f"  │  ✗ Transfer not proven yet.")
        print(f"  │  Agreement={agr_uci*100:.1f}% (need ≥65%)")
        print(f"  │  REJECT recall={rr_uci*100:.1f}% (need ≥40%)")
    print(f"  └────────────────────────────────────────────────────────")
    print(f"\n  Per-tier agreement on UCI HAR:")
    for tier, ag in sorted(per_uci.items()):
        n = Counter(t for _,t in uci_cert)[tier]
        print(f"    {tier:<12} {ag*100:.1f}%  (n={n})")

    out = {
        "experiment":        "S2S Level 5 Cross-Dataset Transfer",
        "train_dataset":     "PTT-PPG 500Hz s1-s16 IMU only",
        "test_dataset_A":    "PTT-PPG 500Hz s17-s22 (never seen)",
        "test_dataset_B":    "UCI HAR 50Hz all subjects (zero-shot)",
        "n_channels":        N_CHANNELS,
        "ptt_train_windows": len(train_ds),
        "ptt_test_windows":  len(val_ds),
        "uci_windows":       len(uci_cert),
        "epochs":            args.epochs,
        "device":            device,
        "train_time_s":      train_time,
        "ptt_agreement":     round(agr_ptt, 4),
        "ptt_reject_recall": round(rr_ptt, 4),
        "ptt_f1":            round(f1_ptt, 4),
        "uci_agreement":     round(agr_uci, 4),
        "uci_reject_recall": round(rr_uci, 4),
        "uci_f1":            round(f1_uci, 4),
        "uci_per_tier":      per_uci,
        "transfer_proven":   proven,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out,"w"), indent=2)
    print(f"\n  Saved → {args.out}")

if __name__ == "__main__":
    main()
