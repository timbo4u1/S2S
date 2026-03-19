#!/usr/bin/env python3
"""
layer4_sequence_model.py — S2S Layer 4a: Action Sequence Prediction

Trains a small transformer on consecutive certified motion windows.
Given the current window (13-dim physics features), predicts the next
window (8-dim motion targets: position x3, velocity x3, jerk_rms, smoothness).

This is the foundation for:
  4a. predict_next_action([reach]) → [grasp]
  4b. gap_fill([reach], [place]) → reconstruct middle
  4c. sequence → text label (Layer 5 bridge)

Usage:
    python3 layer4_sequence_model.py --train
    python3 layer4_sequence_model.py --train --epochs 100
    python3 layer4_sequence_model.py --predict "reach forward"
    python3 layer4_sequence_model.py --eval

Input:  experiments/sequences_real.npz
Output: experiments/layer4_model.pt + experiments/results_layer4.json
"""
import os, sys, json, argparse, math, time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/S2S"))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("ERROR: torch not installed. Run: pip3 install torch")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SEQUENCES_PATH = Path("experiments/sequences_real.npz")
MODEL_PATH     = Path("experiments/layer4_model.pt")
RESULTS_PATH   = Path("experiments/results_layer4.json")

# Input/output dims from sequences_real.npz
INPUT_DIM  = 13   # physics features per window
OUTPUT_DIM = 8    # next window targets: pos(3) + vel(3) + jerk_rms + smoothness

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MotionPredictor(nn.Module):
    """
    Small transformer encoder + MLP head.
    Input:  (batch, INPUT_DIM) — current window features
    Output: (batch, OUTPUT_DIM) — predicted next window
    """
    def __init__(self, input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
                 hidden=128, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.input_proj(x).unsqueeze(1)   # (batch, 1, hidden)
        x = self.transformer(x)               # (batch, 1, hidden)
        x = x.squeeze(1)                      # (batch, hidden)
        return self.head(x)                   # (batch, output_dim)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sequences(path=SEQUENCES_PATH, tier_filter=None):
    """
    Load sequences_real.npz.
    tier_filter: None=all, 'GOLD'=gold only, 'SILVER+'=silver and gold
    Returns X (N, 13), Y (N, 8) as float32 tensors.
    """
    if not path.exists():
        print(f"ERROR: {path} not found. Run extract_sequences.py first.")
        sys.exit(1)

    data   = np.load(path, allow_pickle=True)
    X      = data["X"].astype(np.float32)
    Y      = data["Y"].astype(np.float32)
    tiers  = data["tiers"]
    sources = data["sources"]

    print(f"Loaded {len(X)} sequence pairs")
    print(f"  Sources: {np.unique(sources, return_counts=True)}")
    print(f"  Tiers:   {np.unique(tiers, return_counts=True)}")

    # Optional tier filtering
    if tier_filter == "GOLD":
        mask = tiers == "GOLD"
        X, Y = X[mask], Y[mask]
        print(f"  After GOLD filter: {len(X)} pairs")
    elif tier_filter == "SILVER+":
        mask = np.isin(tiers, ["GOLD", "SILVER"])
        X, Y = X[mask], Y[mask]
        print(f"  After SILVER+ filter: {len(X)} pairs")

    # Normalize X and Y to zero mean, unit variance
    X_mean, X_std = X.mean(0), X.std(0) + 1e-8
    Y_mean, Y_std = Y.mean(0), Y.std(0) + 1e-8
    X_norm = (X - X_mean) / X_std
    Y_norm = (Y - Y_mean) / Y_std

    stats = {
        "X_mean": X_mean.tolist(), "X_std": X_std.tolist(),
        "Y_mean": Y_mean.tolist(), "Y_std": Y_std.tolist(),
    }

    return (torch.tensor(X_norm), torch.tensor(Y_norm),
            torch.tensor(X), torch.tensor(Y), stats)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs=50, batch_size=64, lr=1e-3, tier_filter="SILVER+",
          hidden=128, n_heads=4, n_layers=2):

    print(f"\nLayer 4a Training — {epochs} epochs, hidden={hidden}")
    print("=" * 55)

    X_norm, Y_norm, X_raw, Y_raw, stats = load_sequences(tier_filter=tier_filter)

    dataset = TensorDataset(X_norm, Y_norm)
    n_val   = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model     = MotionPredictor(hidden=hidden, n_heads=n_heads, n_layers=n_layers)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Train: {n_train}  Val: {n_val}")
    print()

    best_val_loss = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= n_val

        scheduler.step()

        history.append({"epoch": epoch, "train_loss": round(train_loss, 6),
                        "val_loss": round(val_loss, 6)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "stats": stats,
                "config": {"hidden": hidden, "n_heads": n_heads,
                           "n_layers": n_layers, "input_dim": INPUT_DIM,
                           "output_dim": OUTPUT_DIM},
                "epoch": epoch,
                "val_loss": val_loss,
            }, MODEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val_loss:.4f}  "
                  f"({elapsed:.0f}s)")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved → {MODEL_PATH}")

    # Smoothness correlation on validation set
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds.append(model(xb).numpy())
            targets.append(yb.numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Correlation per output dimension
    correlations = []
    for i in range(OUTPUT_DIM):
        if targets[:, i].std() > 1e-8:
            r = np.corrcoef(preds[:, i], targets[:, i])[0, 1]
            correlations.append(float(r))

    mean_r = float(np.mean(correlations)) if correlations else 0.0
    smoothness_r = correlations[-1] if correlations else 0.0

    print(f"Mean correlation r: {mean_r:.3f}")
    print(f"Smoothness r:       {smoothness_r:.3f}")

    results = {
        "layer": 4,
        "sub_layer": "4a_sequence_prediction",
        "n_train": n_train,
        "n_val": n_val,
        "epochs": epochs,
        "best_val_loss": round(best_val_loss, 6),
        "mean_correlation_r": round(mean_r, 3),
        "smoothness_correlation_r": round(smoothness_r, 3),
        "output_correlations": [round(r, 3) for r in correlations],
        "history": history[-10:],
        "model_path": str(MODEL_PATH),
        "tier_filter": tier_filter,
        "n_params": n_params,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {RESULTS_PATH}")
    return results


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_model():
    if not MODEL_PATH.exists():
        print(f"ERROR: No trained model at {MODEL_PATH}. Run --train first.")
        sys.exit(1)
    ckpt   = torch.load(MODEL_PATH, map_location="cpu")
    cfg    = ckpt["config"]
    model  = MotionPredictor(
        input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
        hidden=cfg["hidden"], n_heads=cfg["n_heads"], n_layers=cfg["n_layers"]
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["stats"], ckpt


def predict_next(features_raw):
    """
    Given a 13-dim physics feature vector, predict the next motion window.
    features_raw: list or numpy array of 13 floats
    Returns: 8-dim prediction (denormalized)
    """
    model, stats, _ = load_model()
    X_mean = np.array(stats["X_mean"], dtype=np.float32)
    X_std  = np.array(stats["X_std"],  dtype=np.float32)
    Y_mean = np.array(stats["Y_mean"], dtype=np.float32)
    Y_std  = np.array(stats["Y_std"],  dtype=np.float32)

    x = (np.array(features_raw, dtype=np.float32) - X_mean) / X_std
    with torch.no_grad():
        y_norm = model(torch.tensor(x).unsqueeze(0)).numpy()[0]
    y = y_norm * Y_std + Y_mean
    return y


def evaluate():
    """Evaluate model on full dataset, print per-output correlations."""
    model, stats, ckpt = load_model()
    X_norm, Y_norm, X_raw, Y_raw, _ = load_sequences()

    X_mean = np.array(stats["X_mean"], dtype=np.float32)
    X_std  = np.array(stats["X_std"],  dtype=np.float32)
    Y_mean = np.array(stats["Y_mean"], dtype=np.float32)
    Y_std  = np.array(stats["Y_std"],  dtype=np.float32)

    loader = DataLoader(TensorDataset(X_norm, Y_norm), batch_size=256)
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb).numpy())
            targets.append(yb.numpy())

    preds   = np.concatenate(preds) * np.array(stats["Y_std"]) + np.array(stats["Y_mean"])
    targets = np.concatenate(targets) * np.array(stats["Y_std"]) + np.array(stats["Y_mean"])

    labels = ["pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z",
              "jerk_rms", "smoothness"]

    print(f"\nLayer 4a Evaluation  (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    print(f"{'Output':<14} {'Corr r':>8}  {'MAE':>10}")
    print("─" * 38)
    for i, label in enumerate(labels[:OUTPUT_DIM]):
        r   = np.corrcoef(preds[:, i], targets[:, i])[0, 1] if targets[:, i].std() > 1e-8 else 0
        mae = np.mean(np.abs(preds[:, i] - targets[:, i]))
        print(f"{label:<14} {r:>8.3f}  {mae:>10.4f}")

    mean_r = np.mean([np.corrcoef(preds[:, i], targets[:, i])[0, 1]
                      for i in range(OUTPUT_DIM) if targets[:, i].std() > 1e-8])
    print(f"{'Mean':<14} {mean_r:>8.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="S2S Layer 4a — Motion Sequence Predictor")
    parser.add_argument("--train",   action="store_true", help="Train the model")
    parser.add_argument("--eval",    action="store_true", help="Evaluate trained model")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--hidden",  type=int, default=128)
    parser.add_argument("--batch",   type=int, default=64)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--tiers",   default="SILVER+",
                        choices=["all", "SILVER+", "GOLD"])
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, batch_size=args.batch,
              lr=args.lr, tier_filter=args.tiers, hidden=args.hidden)

    elif args.eval:
        evaluate()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
