#!/usr/bin/env python3
"""
layer4c_intent_recognition.py — S2S Layer 4c: Motion → Language v3

CrossEntropy classification instead of contrastive loss.
Trains motion encoder to predict intent label directly.
Simpler, stable, no NaN.

Architecture:
  Motion encoder: 13-dim → 128-dim → n_labels (softmax)
  At inference: encode motion → cosine sim with text embeddings

Usage:
    python3.9 experiments/layer4c_intent_recognition.py --train
    python3.9 experiments/layer4c_intent_recognition.py --eval
    python3.9 experiments/layer4c_intent_recognition.py --query "pick up cup"
    python3.9 experiments/layer4c_intent_recognition.py --classify
"""
import os, sys, json, argparse, time, random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.expanduser("~/S2S"))

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset, random_split
except ImportError:
    print("pip3.9 install torch"); sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("pip3.9 install sentence-transformers"); sys.exit(1)

NPZ_PATH     = Path("experiments/sequences_real.npz")
INDEX_PATH   = Path("experiments/retrieval_index_v3.json")
MODEL_PATH   = Path("experiments/layer4c_model.pt")
RESULTS_PATH = Path("experiments/results_layer4c.json")

INPUT_DIM = 13
HIDDEN    = 256
PROJ_DIM  = 384

SOURCE_LABELS = {
    0: ["forearm reach", "hand grasp", "wrist rotation",
        "finger extension", "forearm curl", "elbow flex",
        "wrist supination", "hand open", "pinch grip"],
    1: ["prosthetic grasp", "wrist flex", "hand open",
        "pinch grip", "lateral grip", "key grip",
        "hook grip", "power grasp", "precision grasp"],
    2: ["robot reach", "robot grasp", "pick object",
        "place object", "push forward", "pull back",
        "rotate wrist", "lift object", "lower object"],
    3: ["walking", "running", "cycling", "ascending stairs",
        "descending stairs", "standing", "sitting",
        "lying down", "jumping", "Nordic walking"],
    4: ["baseline motion", "stress response", "relaxed movement",
        "amusement gesture", "meditation stillness",
        "calm posture", "arousal movement"],
}


# ---------------------------------------------------------------------------
# Model — motion encoder + classification head
# ---------------------------------------------------------------------------

class MotionClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN,
                 n_labels=10, proj_dim=PROJ_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        # Classification head
        self.classifier = nn.Linear(hidden // 2, n_labels)
        # Projection head for embedding-space retrieval at inference
        self.projector = nn.Sequential(
            nn.Linear(hidden // 2, proj_dim),
            nn.LayerNorm(proj_dim),
        )

    def forward(self, x):
        x   = x.float()
        h   = self.encoder(x)
        cls = self.classifier(h)
        return cls

    def embed(self, x):
        x = x.float()
        h = self.encoder(x)
        e = self.projector(h)
        return nn.functional.normalize(e, dim=-1)


# ---------------------------------------------------------------------------
# Build data
# ---------------------------------------------------------------------------

def build_data():
    data    = np.load(str(NPZ_PATH), allow_pickle=True)
    X_seq   = data["X"].astype(np.float32)
    sources = data["sources"]

    rng = random.Random(42)
    features, labels = [], []

    for feat, src in zip(X_seq, sources):
        label = rng.choice(SOURCE_LABELS[int(src)])
        features.append(feat)
        labels.append(label)

    n_retrieval = 0
    if INDEX_PATH.exists():
        index   = json.loads(INDEX_PATH.read_text())
        windows = index.get("windows", [])
        for entry in windows:
            instr = entry.get("instruction", "")
            feats = entry.get("features", [])
            if instr and len(feats) == INPUT_DIM and not any(f != f for f in feats):
                features.append(np.array(feats, dtype=np.float32))
                labels.append(instr)
                n_retrieval += 1
        print(f"  Retrieval index: {n_retrieval} entries")

    print(f"  Total: {len(features)} samples")

    unique_labels = sorted(set(labels))
    label_to_id   = {l: i for i, l in enumerate(unique_labels)}
    label_ids     = [label_to_id[l] for l in labels]

    counts = defaultdict(int)
    for lid in label_ids:
        counts[lid] += 1
    print(f"  Unique labels: {len(unique_labels)}")
    print(f"  Samples/label: min={min(counts.values())} "
          f"max={max(counts.values())} "
          f"mean={int(sum(counts.values())/len(counts))}")

    X = np.array(features, dtype=np.float32)
    return X, label_ids, unique_labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs=80, batch_size=256, lr=1e-3):
    print(f"\nLayer 4c v3 — CrossEntropy Intent Classification — {epochs} epochs")
    print("=" * 60)

    print("Loading sentence-transformers...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    text_model.eval()

    print("Building training data...")
    X_raw, label_ids, unique_labels = build_data()
    n_labels = len(unique_labels)

    # Normalize
    feat_mean = X_raw.mean(axis=0).astype(np.float32)
    feat_std  = (X_raw.std(axis=0) + 1e-8).astype(np.float32)
    X_norm    = ((X_raw - feat_mean) / feat_std).astype(np.float32)

    X_t = torch.tensor(X_norm)
    Y_t = torch.tensor(label_ids, dtype=torch.long)

    dataset  = TensorDataset(X_t, Y_t)
    n_val    = max(1, int(len(dataset) * 0.15))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model     = MotionClassifier(n_labels=n_labels)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params  "
          f"Train: {n_train}  Val: {n_val}  Labels: {n_labels}\n")

    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss    += loss.item() * len(xb)
            train_correct += (logits.argmax(1) == yb).sum().item()

        train_loss /= n_train
        train_acc   = train_correct / n_train

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits   = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= n_val
        val_acc   = val_correct / n_val
        scheduler.step()

        history.append({"epoch": epoch,
                        "train_loss": round(train_loss, 4),
                        "train_acc":  round(train_acc, 4),
                        "val_loss":   round(val_loss, 4),
                        "val_acc":    round(val_acc, 4)})

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            torch.save({
                "model_state":   model.state_dict(),
                "feat_mean":     feat_mean.tolist(),
                "feat_std":      feat_std.tolist(),
                "unique_labels": unique_labels,
                "n_labels":      n_labels,
                "config": {"hidden": HIDDEN, "proj_dim": PROJ_DIM,
                           "input_dim": INPUT_DIM, "n_labels": n_labels},
                "epoch":    epoch,
                "val_acc":  val_acc,
                "val_loss": val_loss,
            }, MODEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"loss={train_loss:.4f}/{val_loss:.4f}  "
                  f"acc={train_acc:.3f}/{val_acc:.3f}  "
                  f"best_acc={best_val_acc:.3f}  "
                  f"({time.time()-t0:.0f}s)")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Best val loss:     {best_val_loss:.4f}")

    # Save label embeddings for text-based retrieval at inference
    print("Encoding label embeddings for retrieval...")
    label_embs = text_model.encode(unique_labels, batch_size=64,
                                    show_progress_bar=False)
    np.save("experiments/layer4c_label_embs.npy", label_embs)
    Path("experiments/layer4c_labels.json").write_text(
        json.dumps(unique_labels))
    print(f"Model saved → {MODEL_PATH}")

    results = {
        "layer": "4c_v3",
        "task": "crossentropy_intent_classification",
        "n_train": n_train,
        "n_val": n_val,
        "n_unique_labels": n_labels,
        "best_val_accuracy": round(best_val_acc, 4),
        "best_val_loss": round(best_val_loss, 4),
        "n_params": n_params,
        "history": history[-10:],
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    return results


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def load_inference():
    if not MODEL_PATH.exists():
        print("No model. Run --train first."); sys.exit(1)
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    cfg   = ckpt["config"]
    model = MotionClassifier(n_labels=cfg["n_labels"],
                             hidden=cfg.get("hidden", HIDDEN),
                             proj_dim=cfg.get("proj_dim", PROJ_DIM))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    feat_mean  = np.array(ckpt["feat_mean"], dtype=np.float32)
    feat_std   = np.array(ckpt["feat_std"],  dtype=np.float32)
    labels     = ckpt["unique_labels"]
    label_embs = np.load("experiments/layer4c_label_embs.npy").astype(np.float32)
    label_embs /= np.linalg.norm(label_embs, axis=1, keepdims=True) + 1e-8
    return model, feat_mean, feat_std, labels, label_embs


def encode_motion(model, feat_mean, feat_std, feats):
    x = ((np.array(feats, dtype=np.float32) - feat_mean) / feat_std)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    with torch.no_grad():
        emb = model.embed(torch.tensor(x)).numpy()
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)


def classify_motion(feats, top_k=3):
    model, feat_mean, feat_std, labels, _ = load_inference()
    x = ((np.array(feats, dtype=np.float32) - feat_mean) / feat_std)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    with torch.no_grad():
        logits = model(torch.tensor(x))[0]
        probs  = torch.softmax(logits, dim=-1).numpy()
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(labels[i], float(probs[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate():
    model, feat_mean, feat_std, labels, label_embs = load_inference()
    X_raw, label_ids, unique_labels = build_data()

    rng = random.Random(99)
    idx = rng.sample(range(len(X_raw)), min(2000, len(X_raw)))
    X_test = X_raw[idx]
    y_test = [label_ids[i] for i in idx]

    X_norm = (X_test - feat_mean) / feat_std
    with torch.no_grad():
        logits = model(torch.tensor(X_norm)).numpy()

    print("\nLayer 4c v3 — Evaluation")
    print("=" * 40)

    for k in [1, 3, 5]:
        correct = sum(
            1 for i, true_id in enumerate(y_test)
            if true_id in np.argsort(logits[i])[::-1][:k]
        )
        print(f"Top-{k:2d} accuracy: {correct/len(y_test):.3f}  "
              f"({correct}/{len(y_test)})")

    # Per-source accuracy
    data    = np.load(str(NPZ_PATH), allow_pickle=True)
    sources = data["sources"]
    SOURCE_NAMES = {0:"NinaPro",1:"Amputee",2:"RoboTurk",3:"PAMAP2",4:"WESAD"}
    print("\nPer-source top-1 (on sequences_real.npz subset):")
    for src_id in range(5):
        src_idx = [i for i, j in enumerate(idx)
                   if j < len(sources) and sources[j] == src_id]
        if not src_idx:
            continue
        correct = sum(1 for i in src_idx
                      if np.argmax(logits[i]) == y_test[i])
        print(f"  {SOURCE_NAMES[src_id]:<12} "
              f"{correct}/{len(src_idx)} "
              f"({100*correct//max(len(src_idx),1)}%)")


# ---------------------------------------------------------------------------
# Text query
# ---------------------------------------------------------------------------

def query_by_text(query_text, top_k=5):
    model, feat_mean, feat_std, labels, label_embs = load_inference()
    text_model = SentenceTransformer("all-MiniLM-L6-v2")

    q_emb = text_model.encode([query_text], show_progress_bar=False)[0]
    q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-8)

    label_sims = np.dot(label_embs, q_emb)
    top_labels = np.argsort(label_sims)[::-1][:top_k]

    print(f"\nQuery: '{query_text}'")
    print("Matching labels:")
    for i in top_labels:
        print(f"  {label_sims[i]:.4f}  {labels[i]}")

    # Find matching motion windows
    data    = np.load(str(NPZ_PATH), allow_pickle=True)
    X_raw   = data["X"].astype(np.float32)
    sources = data["sources"]

    motion_embs = []
    with torch.no_grad():
        for i in range(0, len(X_raw), 512):
            motion_embs.append(
                encode_motion(model, feat_mean, feat_std, X_raw[i:i+512]))
    motion_embs = np.concatenate(motion_embs)

    motion_sims = np.dot(motion_embs, q_emb)
    top_win     = np.argsort(motion_sims)[::-1][:top_k]

    SOURCE_NAMES = {0:"NinaPro",1:"Amputee",2:"RoboTurk",3:"PAMAP2",4:"WESAD"}
    print("\nMatching motion windows:")
    for i in top_win:
        print(f"  [{SOURCE_NAMES.get(int(sources[i]),'?'):<10}] "
              f"sim={motion_sims[i]:.4f}")


# ---------------------------------------------------------------------------
# Classify random windows
# ---------------------------------------------------------------------------

def classify():
    data = np.load(str(NPZ_PATH), allow_pickle=True)
    X    = data["X"].astype(np.float32)
    print("\nClassifying 5 random windows:")
    print("─" * 40)
    for _ in range(5):
        idx     = random.randint(0, len(X)-1)
        results = classify_motion(X[idx], top_k=3)
        print(f"\nWindow {idx}:")
        for label, prob in results:
            print(f"  {prob:.4f}  {label}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--eval",     action="store_true")
    parser.add_argument("--classify", action="store_true")
    parser.add_argument("--query",    default=None)
    parser.add_argument("--epochs",   type=int, default=80)
    parser.add_argument("--batch",    type=int, default=256)
    parser.add_argument("--lr",       type=float, default=1e-3)
    args = parser.parse_args()

    if args.train:
        train(epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    elif args.eval:
        evaluate()
    elif args.query:
        query_by_text(args.query)
    elif args.classify:
        classify()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
