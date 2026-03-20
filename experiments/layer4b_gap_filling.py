#!/usr/bin/env python3
"""
layer4b_gap_filling.py — S2S Layer 4b: Physics-Constrained Gap Filling v2

Trained on QUADRUPLETS (4 consecutive windows) at t=1/3, 1/2, 2/3.
Properly tests whether neural gap filling beats linear interpolation
at non-midpoint positions.

Architecture:
  Input:  [start(13) + end(13) + progress(1)] = 27-dim
  Output: intermediate_features(13)
  Physics prior: output = linear_interp + learned_correction

Usage:
    python3.9 experiments/layer4b_gap_filling.py --train
    python3.9 experiments/layer4b_gap_filling.py --eval
    python3.9 experiments/layer4b_gap_filling.py --demo
"""
import os, sys, json, argparse, time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/S2S"))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
except ImportError:
    print("ERROR: pip3.9 install torch"); sys.exit(1)

NPZ_PATH     = Path("experiments/sequences_real.npz")
MODEL_PATH   = Path("experiments/layer4b_model.pt")
RESULTS_PATH = Path("experiments/results_layer4b.json")

INPUT_DIM = 13
HIDDEN    = 256


class GapFiller(nn.Module):
    """
    Physics-residual gap filler.
    output = linear_interpolation(start, end, t) + correction(start, end, t)
    """
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN):
        super().__init__()
        combined = input_dim * 2 + 1

        self.encoder = nn.Sequential(
            nn.Linear(combined, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
        )
        self.correction_head = nn.Linear(hidden // 2, input_dim)

    def forward(self, start, end, t):
        start = start.float()
        end   = end.float()
        t     = t.float()
        x     = torch.cat([start, end, t], dim=-1)
        h     = self.encoder(x)
        correction = self.correction_head(h)
        linear = start + t * (end - start)
        return linear + correction


def compute_stats(X):
    return (X.mean(axis=0).astype(np.float32),
            (X.std(axis=0) + 1e-8).astype(np.float32))


def normalize(X, mean, std):
    return ((np.array(X, dtype=np.float32) - mean) / std)


def denormalize(X, mean, std):
    return (np.array(X, dtype=np.float32) * std + mean)


def build_quadruplets(npz_path=NPZ_PATH):
    """
    Extract 4-consecutive-window groups from sequences_real.npz.
    Each quadruplet (A,B,C,D) → 3 training samples at t=1/3, 2/3, 1/2.
    Also extract triplets at t=0.5.
    """
    data    = np.load(str(npz_path), allow_pickle=True)
    X       = data["X"].astype(np.float32)
    sources = data["sources"]

    starts_list, ends_list, targets_list, t_list = [], [], [], []
    n_quad = 0
    n_trip = 0

    for src_id in np.unique(sources):
        idx = np.where(sources == src_id)[0]

        for k in range(len(idx) - 3):
            i0, i1, i2, i3 = idx[k], idx[k+1], idx[k+2], idx[k+3]
            if not (i1-i0 == 1 and i2-i1 == 1 and i3-i2 == 1):
                continue
            A, B, C, D = X[i0], X[i1], X[i2], X[i3]

            for target, t_val in [(B, 1/3), (C, 2/3), ((B+C)/2, 0.5)]:
                starts_list.append(A)
                ends_list.append(D)
                targets_list.append(target)
                t_list.append(t_val)
            n_quad += 1

        for k in range(len(idx) - 2):
            i0, i1, i2 = idx[k], idx[k+1], idx[k+2]
            if not (i1-i0 == 1 and i2-i1 == 1):
                continue
            A, B, C = X[i0], X[i1], X[i2]
            starts_list.append(A)
            ends_list.append(C)
            targets_list.append(B)
            t_list.append(0.5)
            n_trip += 1

    print(f"Quadruplets: {n_quad} x3 = {n_quad*3} samples")
    print(f"Triplets:    {n_trip} at t=0.5")
    print(f"Total:       {len(starts_list)} training samples")

    return (np.array(starts_list,  dtype=np.float32),
            np.array(ends_list,    dtype=np.float32),
            np.array(targets_list, dtype=np.float32),
            np.array(t_list,       dtype=np.float32).reshape(-1, 1))


def train(epochs=100, batch_size=256, lr=1e-3):
    print(f"\nLayer 4b v2 — Quadruplet Gap Filling — {epochs} epochs")
    print("=" * 60)

    starts, ends, targets, ts = build_quadruplets()

    all_feats = np.concatenate([starts, ends, targets], axis=0)
    feat_mean, feat_std = compute_stats(all_feats)

    S_n = torch.tensor(normalize(starts,  feat_mean, feat_std))
    E_n = torch.tensor(normalize(ends,    feat_mean, feat_std))
    T_n = torch.tensor(normalize(targets, feat_mean, feat_std))
    P_t = torch.tensor(ts)

    dataset  = TensorDataset(S_n, E_n, T_n, P_t)
    n_val    = max(1, int(len(dataset) * 0.15))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model     = GapFiller(hidden=HIDDEN)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params  Train: {n_train}  Val: {n_val}\n")

    best_val = float("inf")
    history  = []
    t0       = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for s, e, tgt, p in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(s, e, p), tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(s)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for s, e, tgt, p in val_loader:
                val_loss += criterion(model(s, e, p), tgt).item() * len(s)
        val_loss /= n_val
        scheduler.step()

        history.append({"epoch": epoch,
                        "train": round(train_loss, 6),
                        "val":   round(val_loss, 6)})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "feat_mean":   feat_mean.tolist(),
                "feat_std":    feat_std.tolist(),
                "config":      {"hidden": HIDDEN, "input_dim": INPUT_DIM},
                "epoch":       epoch,
                "val_loss":    val_loss,
            }, MODEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val:.4f}  ({time.time()-t0:.0f}s)")

    # Per-t evaluation
    model.eval()
    print(f"\nBest val loss: {best_val:.4f}")
    print(f"\n{'t':>6}  {'Neural r':>10}  {'Linear r':>10}  {'Delta r':>8}  "
          f"{'Neural MAE':>12}  {'Linear MAE':>12}")
    print("─" * 66)

    results_by_t = {}
    for t_val in [1/3, 0.5, 2/3]:
        mask = np.abs(ts.flatten() - t_val) < 0.01
        if mask.sum() < 10:
            continue

        s_n   = torch.tensor(normalize(starts[mask],  feat_mean, feat_std))
        e_n   = torch.tensor(normalize(ends[mask],    feat_mean, feat_std))
        p_t   = torch.tensor(ts[mask])
        tgt   = targets[mask]
        lin   = starts[mask] + t_val * (ends[mask] - starts[mask])

        with torch.no_grad():
            pred_n = model(s_n, e_n, p_t).numpy()
        pred = denormalize(pred_n, feat_mean, feat_std)

        nr_vals, lr_vals = [], []
        for i in range(INPUT_DIM):
            if tgt[:, i].std() > 1e-8:
                nr_vals.append(float(np.corrcoef(pred[:, i], tgt[:, i])[0, 1]))
                lr_vals.append(float(np.corrcoef(lin[:, i],  tgt[:, i])[0, 1]))

        nr    = float(np.mean(nr_vals))
        lr_   = float(np.mean(lr_vals))
        n_mae = float(np.mean(np.abs(pred - tgt)))
        l_mae = float(np.mean(np.abs(lin  - tgt)))

        print(f"{t_val:>6.2f}  {nr:>10.3f}  {lr_:>10.3f}  {nr-lr_:>+8.3f}  "
              f"{n_mae:>12.4f}  {l_mae:>12.4f}")

        results_by_t[f"t_{t_val:.2f}"] = {
            "neural_r":    round(nr, 3),
            "linear_r":    round(lr_, 3),
            "delta_r":     round(nr - lr_, 3),
            "neural_mae":  round(n_mae, 4),
            "linear_mae":  round(l_mae, 4),
            "n_samples":   int(mask.sum()),
        }

    results = {
        "layer": "4b_v2",
        "task": "quadruplet_gap_filling",
        "n_train": n_train,
        "n_val": n_val,
        "best_val_loss": round(best_val, 6),
        "results_by_t": results_by_t,
        "n_params": n_params,
        "history": history[-10:],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults → {RESULTS_PATH}")
    return results


def load_model():
    if not MODEL_PATH.exists():
        print(f"No model at {MODEL_PATH}. Run --train first.")
        sys.exit(1)
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    model = GapFiller(hidden=ckpt["config"]["hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return (model,
            np.array(ckpt["feat_mean"], dtype=np.float32),
            np.array(ckpt["feat_std"],  dtype=np.float32))


def fill_gap(start_features, end_features, n_steps=3):
    """
    Fill n_steps intermediate windows between start and end.
    Returns list of n_steps feature vectors (13-dim each).
    """
    model, feat_mean, feat_std = load_model()
    s = torch.tensor(
        normalize(np.array(start_features, dtype=np.float32),
                  feat_mean, feat_std)).unsqueeze(0)
    e = torch.tensor(
        normalize(np.array(end_features, dtype=np.float32),
                  feat_mean, feat_std)).unsqueeze(0)
    intermediates = []
    for step in range(1, n_steps + 1):
        t = float(step) / (n_steps + 1)
        p = torch.tensor([[t]], dtype=torch.float32)
        with torch.no_grad():
            pred_n = model(s, e, p).numpy()[0]
        intermediates.append(denormalize(pred_n, feat_mean, feat_std))
    return intermediates


def evaluate():
    model, feat_mean, feat_std = load_model()
    starts, ends, targets, ts  = build_quadruplets()

    NAMES = ["accel_rms","accel_std","accel_max","gyro_rms","gyro_std",
             "jerk_rms","jerk_p95","freq_x","freq_y","freq_z",
             "accel_corr","accel_norm","entropy"]

    print(f"\nLayer 4b v2 — Full Evaluation")
    print(f"{'t':>6}  {'Feature':<14} {'Neural r':>10}  {'Linear r':>10}  {'Δ':>6}")
    print("─" * 52)

    for t_val in [1/3, 0.5, 2/3]:
        mask = np.abs(ts.flatten() - t_val) < 0.01
        if mask.sum() < 10:
            continue
        s_n  = torch.tensor(normalize(starts[mask], feat_mean, feat_std))
        e_n  = torch.tensor(normalize(ends[mask],   feat_mean, feat_std))
        p_t  = torch.tensor(ts[mask])
        tgt  = targets[mask]
        lin  = starts[mask] + t_val * (ends[mask] - starts[mask])

        with torch.no_grad():
            pred_n = model(s_n, e_n, p_t).numpy()
        pred = denormalize(pred_n, feat_mean, feat_std)

        for i, name in enumerate(NAMES):
            if tgt[:, i].std() > 1e-8:
                nr  = float(np.corrcoef(pred[:, i], tgt[:, i])[0, 1])
                lr_ = float(np.corrcoef(lin[:, i],  tgt[:, i])[0, 1])
                print(f"{t_val:>6.2f}  {name:<14} {nr:>10.3f}  {lr_:>10.3f}  {nr-lr_:>+6.3f}")
        print()


def demo():
    starts, ends, targets, ts = build_quadruplets()
    idx = np.where(np.abs(ts.flatten() - 1/3) < 0.01)[0]
    if len(idx) == 0:
        print("No t=1/3 samples"); return
    example = idx[min(42, len(idx)-1)]
    filled  = fill_gap(starts[example], ends[example], n_steps=2)
    NAMES   = ["accel_rms","accel_std","accel_max","gyro_rms","gyro_std",
               "jerk_rms","jerk_p95","freq_x","freq_y","freq_z",
               "accel_corr","accel_norm","entropy"]
    print("\nLayer 4b Demo — 2 gaps filled between start and end")
    print(f"{'Feature':<14} {'Start':>8}  {'t=0.33':>8}  {'t=0.67':>8}  {'End':>8}")
    print("─" * 54)
    for j, name in enumerate(NAMES):
        print(f"{name:<14} {starts[example][j]:>8.3f}  "
              f"{filled[0][j]:>8.3f}  {filled[1][j]:>8.3f}  "
              f"{ends[example][j]:>8.3f}")
    mae    = np.mean(np.abs(filled[0] - targets[example]))
    lin_m  = np.mean(np.abs((starts[example]+ends[example])/2 - targets[example]))
    print(f"\nGapFiller MAE: {mae:.4f}  Linear MAE: {lin_m:.4f}")


def main():
    parser = argparse.ArgumentParser(description="S2S Layer 4b v2 — Gap Filling")
    parser.add_argument("--train",  action="store_true")
    parser.add_argument("--eval",   action="store_true")
    parser.add_argument("--demo",   action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch",  type=int, default=256)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    elif args.eval:
        evaluate()
    elif args.demo:
        demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
