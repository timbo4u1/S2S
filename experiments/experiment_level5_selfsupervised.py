#!/usr/bin/env python3
"""
S2S Level 5 — Self-Supervised Pre-training on Certified Data

The certifier score IS the supervision signal. No activity labels used
during pre-training. Fine-tune on 10% of UCI HAR labels.

Design:
  G_pretrained — encoder pre-trained to predict S2S score → fine-tuned on 10% labels
  H_random     — same encoder, random init             → fine-tuned on 10% labels

If G_pretrained F1 > H_random F1 → Level 5 proven.

Run from ~/S2S:
  python3 experiments/experiment_level5_selfsupervised.py \
    --uci  "data/uci_har/UCI HAR Dataset/" \
    --out  experiments/results_level5_selfsupervised.json
"""

import os, sys, json, math, random, time, argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)

HZ          = 50.0
WINDOW_SIZE = 128
DT_NS       = int(1e9 / HZ)

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_signal_file(path):
    windows = []
    with open(path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if vals:
                windows.append(vals)
    return windows

def load_uci_har(data_dir):
    base    = Path(data_dir)
    windows = []
    for split in ["train", "test"]:
        sig_dir  = base / split / "Inertial Signals"
        subjects = [int(x) for x in (base / split / f"subject_{split}.txt").read_text().split()]
        labels   = [int(x) for x in (base / split / f"y_{split}.txt").read_text().split()]
        bax = load_signal_file(sig_dir / f"body_acc_x_{split}.txt")
        bay = load_signal_file(sig_dir / f"body_acc_y_{split}.txt")
        baz = load_signal_file(sig_dir / f"body_acc_z_{split}.txt")
        bgx = load_signal_file(sig_dir / f"body_gyro_x_{split}.txt")
        bgy = load_signal_file(sig_dir / f"body_gyro_y_{split}.txt")
        bgz = load_signal_file(sig_dir / f"body_gyro_z_{split}.txt")
        for i in range(len(labels)):
            accel = [[bax[i][j], bay[i][j], baz[i][j]] for j in range(WINDOW_SIZE)]
            gyro  = [[bgx[i][j], bgy[i][j], bgz[i][j]] for j in range(WINDOW_SIZE)]
            ts_ns = [int(j * DT_NS) + int(random.gauss(0, 300)) for j in range(WINDOW_SIZE)]
            windows.append({
                "accel": accel, "gyro": gyro, "timestamps": ts_ns,
                "subject": subjects[i], "label": labels[i] - 1, "split": split,
            })
    print(f"  UCI HAR: {len(windows)} windows from "
          f"{len(set(w['subject'] for w in windows))} subjects")
    return windows

# ── S2S CERTIFICATION ─────────────────────────────────────────────────────────

def certify_window(w):
    try:
        from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
        imu_raw = {"timestamps_ns": w["timestamps"],
                   "accel": w["accel"], "gyro": w["gyro"]}
        result = PhysicsEngine().certify(imu_raw=imu_raw, segment="forearm")
        score  = result.get("physical_law_score", result.get("score", 0)) or 0
        tier   = result.get("tier", "BRONZE")
        return int(score), tier
    except Exception:
        return 0, "REJECTED"

# ── FEATURES ──────────────────────────────────────────────────────────────────

def extract_features(w):
    """18 features: per-axis mean+std+range for accel(9) + gyro(9)."""
    feats = []
    for sensor in [w["accel"], w["gyro"]]:
        for axis in range(3):
            vals = [sensor[i][axis] for i in range(len(sensor))]
            mu   = sum(vals) / len(vals)
            var  = sum((v - mu)**2 for v in vals) / len(vals)
            feats += [mu, math.sqrt(var), max(vals) - min(vals)]
    return feats

# ── ENCODER + HEADS ───────────────────────────────────────────────────────────

class Encoder:
    """18 → 128 → 64 shared backbone."""
    def __init__(self, input_dim=18):
        s1 = math.sqrt(2.0 / input_dim)
        s2 = math.sqrt(2.0 / 128)
        self.W1 = [[random.gauss(0, s1) for _ in range(input_dim)] for _ in range(128)]
        self.b1 = [0.0] * 128
        self.W2 = [[random.gauss(0, s2) for _ in range(128)] for _ in range(64)]
        self.b2 = [0.0] * 64

    def forward(self, x):
        h1 = [max(0, sum(self.W1[j][i]*x[i] for i in range(len(x))) + self.b1[j])
               for j in range(128)]
        h2 = [max(0, sum(self.W2[k][j]*h1[j] for j in range(128)) + self.b2[k])
               for k in range(64)]
        return h1, h2

    def backward_from_h2(self, x, h1, h2, dh2, lr):
        # Backprop through layer 2
        for k in range(64):
            if h2[k] > 0:
                for j in range(128):
                    self.W2[k][j] -= lr * dh2[k] * h1[j]
                self.b2[k] -= lr * dh2[k]
        # Backprop to h1
        dh1 = [sum(self.W2[k][j]*dh2[k] for k in range(64)) * (1 if h1[j]>0 else 0)
                for j in range(128)]
        for j in range(128):
            for i in range(len(x)):
                self.W1[j][i] -= lr * dh1[j] * x[i]
            self.b1[j] -= lr * dh1[j]

class RegressionHead:
    """64 → 1 for score prediction (pre-training)."""
    def __init__(self):
        s = math.sqrt(2.0 / 64)
        self.W = [random.gauss(0, s) for _ in range(64)]
        self.b = 0.0

    def forward(self, h2):
        return sum(self.W[j]*h2[j] for j in range(64)) + self.b

    def backward(self, h2, pred, target, lr):
        err  = pred - target           # MSE gradient
        dh2  = [err * self.W[j] for j in range(64)]
        for j in range(64):
            self.W[j] -= lr * err * h2[j]
        self.b -= lr * err
        return dh2

class ClassifierHead:
    """64 → 6 for activity classification (fine-tuning)."""
    def __init__(self):
        s = math.sqrt(2.0 / 64)
        self.W = [[random.gauss(0, s) for _ in range(64)] for _ in range(6)]
        self.b = [0.0] * 6

    def forward(self, h2):
        o = [sum(self.W[k][j]*h2[j] for j in range(64)) + self.b[k] for k in range(6)]
        m = max(o)
        e = [math.exp(v - m) for v in o]
        s = sum(e)
        return [v/s for v in e]

    def backward(self, h2, probs, label, lr):
        dL  = [p - (1 if i == label else 0) for i, p in enumerate(probs)]
        dh2 = [sum(self.W[k][j]*dL[k] for k in range(6)) for j in range(64)]
        for k in range(6):
            for j in range(64):
                self.W[k][j] -= lr * dL[k] * h2[j]
            self.b[k] -= lr * dL[k]
        return dh2

def normalize_features(train_X, test_X=None):
    n = len(train_X[0])
    means = [sum(x[i] for x in train_X)/len(train_X) for i in range(n)]
    stds  = [max(math.sqrt(sum((x[i]-means[i])**2 for x in train_X)/len(train_X)), 1e-8)
              for i in range(n)]
    def norm(X):
        return [[(x[i]-means[i])/stds[i] for i in range(n)] for x in X]
    if test_X is not None:
        return norm(train_X), norm(test_X), means, stds
    return norm(train_X), means, stds

def apply_norm(X, means, stds):
    n = len(means)
    return [[(x[i]-means[i])/stds[i] for i in range(n)] for x in X]

def evaluate_classifier(encoder, head, X, y):
    correct = 0
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for x, label in zip(X, y):
        _, h2   = encoder.forward(x)
        probs   = head.forward(h2)
        pred    = probs.index(max(probs))
        if pred == label: correct += 1
        tp[label] += (pred == label)
        fp[pred]  += (pred != label)
        fn[label] += (pred != label)
    acc = correct / len(y)
    f1s = []
    for c in range(6):
        p  = tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0
        r  = tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0
        f1s.append(2*p*r/(p+r) if p+r else 0)
    return acc, sum(f1s)/6

# ── PRE-TRAINING ──────────────────────────────────────────────────────────────

def pretrain_encoder(pretrain_X, pretrain_scores, epochs=30, lr=0.005):
    """
    Pre-train encoder to predict S2S physics score from IMU features.
    No activity labels used. The certifier IS the teacher.
    """
    encoder = Encoder()
    head    = RegressionHead()

    # Normalize scores to [0, 1]
    max_score = max(pretrain_scores) or 1
    norm_scores = [s / max_score for s in pretrain_scores]

    print(f"  Pre-training on {len(pretrain_X)} certified windows "
          f"(score range {min(pretrain_scores)}–{max(pretrain_scores)})")

    for e in range(1, epochs+1):
        idx = list(range(len(pretrain_X)))
        random.shuffle(idx)
        total_loss = 0.0
        for i in idx:
            _, h2   = encoder.forward(pretrain_X[i])
            pred    = head.forward(h2)
            loss    = 0.5 * (pred - norm_scores[i])**2
            total_loss += loss
            dh2     = head.backward(h2, pred, norm_scores[i], lr)
            encoder.backward_from_h2(pretrain_X[i],
                                     encoder.forward(pretrain_X[i])[0],
                                     h2, dh2, lr)
        if e % 10 == 0 or e == epochs:
            print(f"  ep {e:3d}/{epochs}  mse={total_loss/len(pretrain_X):.4f}")

    return encoder

# ── FINE-TUNING ───────────────────────────────────────────────────────────────

def finetune(encoder, finetune_X, finetune_y, test_X, test_y,
             freeze_encoder=True, epochs=40, lr=0.005, label=""):
    """Fine-tune classifier head. Optionally freeze encoder weights."""
    head = ClassifierHead()

    for e in range(1, epochs+1):
        idx = list(range(len(finetune_X)))
        random.shuffle(idx)
        for i in idx:
            _, h2   = encoder.forward(finetune_X[i])
            probs   = head.forward(h2)
            dh2     = head.backward(h2, probs, finetune_y[i], lr)
            if not freeze_encoder:
                encoder.backward_from_h2(finetune_X[i],
                                         encoder.forward(finetune_X[i])[0],
                                         h2, dh2, lr)
        if e % 10 == 0 or e == epochs:
            acc, f1 = evaluate_classifier(encoder, head, test_X, test_y)
            print(f"  ep {e:3d}/{epochs}  acc={acc:.3f}  f1={f1:.4f}  {label}")

    acc, f1 = evaluate_classifier(encoder, head, test_X, test_y)
    print(f"  Final: acc={acc:.4f}  F1={f1:.4f}")
    return acc, f1

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uci",    required=True, help="UCI HAR Dataset/ directory")
    p.add_argument("--out",    required=True)
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--finetune_epochs", type=int, default=40)
    p.add_argument("--label_pct",       type=float, default=0.10,
                   help="Fraction of training labels to use for fine-tuning (default 0.10)")
    args = p.parse_args()

    print("\nS2S Level 5 — Self-Supervised Pre-training")
    print("=" * 60)
    print(f"  Pre-train: predict S2S score (no activity labels)")
    print(f"  Fine-tune: {int(args.label_pct*100)}% of UCI HAR labels")
    print(f"  Test:      full UCI HAR test split")

    # ── Load data
    print("\nLoading UCI HAR...")
    all_windows = load_uci_har(args.uci)

    train_windows = [w for w in all_windows if w["split"] == "train"]
    test_windows  = [w for w in all_windows if w["split"] == "test"]

    # ── Certify ALL training windows (labels never looked at)
    print(f"\nCertifying {len(train_windows)} training windows with S2S...")
    t0 = time.time()
    certified = []
    for k, w in enumerate(train_windows, 1):
        score, tier = certify_window(w)
        certified.append((w, score, tier))
        if k % 1000 == 0:
            print(f"  [{k}/{len(train_windows)}] elapsed={int(time.time()-t0)}s")

    scores = [s for _, s, _ in certified]
    tiers  = [t for _, _, t in certified]
    from collections import Counter
    tier_counts = Counter(tiers)
    print(f"\n  Score distribution: min={min(scores)} max={max(scores)} "
          f"mean={sum(scores)/len(scores):.1f}")
    print(f"  Tier distribution: {dict(tier_counts)}")

    # Pre-train corpus: GOLD + SILVER windows (highest quality only)
    pretrain_pool = [(w, s) for w, s, t in certified if t in ("GOLD", "SILVER", "BRONZE")]
    print(f"\n  Pre-training corpus: {len(pretrain_pool)} windows "
          f"(GOLD+SILVER+BRONZE, scores ≥ floor)")

    # ── Extract features
    print("\nExtracting features...")
    all_train_X  = [extract_features(w) for w, _, _ in certified]
    all_train_y  = [w["label"]          for w, _, _ in certified]
    test_X_raw   = [extract_features(w) for w in test_windows]
    test_y       = [w["label"]          for w in test_windows]

    pretrain_X_raw    = [extract_features(w) for w, _ in pretrain_pool]
    pretrain_scores_v = [s for _, s in pretrain_pool]

    # Normalize using full training set statistics
    norm_all_train_X, norm_test_X, means, stds = normalize_features(all_train_X, test_X_raw)
    norm_pretrain_X = apply_norm(pretrain_X_raw, means, stds)

    # ── 10% labeled fine-tuning set (fixed seed, stratified by class)
    random.seed(42)
    label_pct = args.label_pct
    finetune_idx = []
    for cls in range(6):
        cls_idx = [i for i, y in enumerate(all_train_y) if y == cls]
        n_keep  = max(1, int(len(cls_idx) * label_pct))
        finetune_idx += random.sample(cls_idx, n_keep)
    finetune_X = [norm_all_train_X[i] for i in finetune_idx]
    finetune_y = [all_train_y[i]       for i in finetune_idx]
    print(f"\n  Fine-tuning labels: {len(finetune_X)} windows "
          f"({int(label_pct*100)}% of {len(all_train_X)} train)")
    print(f"  Test set: {len(test_y)} windows")

    results = {}
    t_start = time.time()

    # ──────────────────────────────────────────────────────────────────────────
    # Condition G — S2S pre-trained encoder → fine-tune on 10% labels
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Condition G_pretrained  |  S2S pre-trained → fine-tune {int(label_pct*100)}% labels")
    print(f"  Step 1: Pre-training (certifier as teacher, no activity labels)")
    t0 = time.time()
    random.seed(42)
    encoder_G = pretrain_encoder(norm_pretrain_X, pretrain_scores_v,
                                  epochs=args.pretrain_epochs, lr=0.005)

    print(f"  Step 2: Fine-tuning on {int(label_pct*100)}% labels (encoder frozen)")
    acc_G, f1_G = finetune(encoder_G, finetune_X, finetune_y,
                            norm_test_X, test_y,
                            freeze_encoder=True,
                            epochs=args.finetune_epochs, lr=0.005,
                            label="[G frozen]")
    results["G_pretrained"] = {
        "acc": round(acc_G, 4), "f1": round(f1_G, 4),
        "n_pretrain": len(pretrain_pool), "n_finetune": len(finetune_X),
        "desc": f"S2S pretrained → finetune {int(label_pct*100)}% labels (frozen encoder)",
        "time_s": round(time.time()-t0)
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Condition G2 — S2S pre-trained encoder → fine-tune ALL layers
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Condition G2_pretrained_full  |  S2S pre-trained → fine-tune ALL layers")
    t0 = time.time()
    random.seed(42)
    encoder_G2 = pretrain_encoder(norm_pretrain_X, pretrain_scores_v,
                                   epochs=args.pretrain_epochs, lr=0.005)
    acc_G2, f1_G2 = finetune(encoder_G2, finetune_X, finetune_y,
                              norm_test_X, test_y,
                              freeze_encoder=False,
                              epochs=args.finetune_epochs, lr=0.003,
                              label="[G2 full]")
    results["G2_pretrained_full"] = {
        "acc": round(acc_G2, 4), "f1": round(f1_G2, 4),
        "n_pretrain": len(pretrain_pool), "n_finetune": len(finetune_X),
        "desc": f"S2S pretrained → finetune {int(label_pct*100)}% labels (full fine-tune)",
        "time_s": round(time.time()-t0)
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Condition H — Random init encoder → fine-tune on 10% labels (baseline)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Condition H_random  |  Random init → fine-tune {int(label_pct*100)}% labels")
    t0 = time.time()
    random.seed(42)
    encoder_H = Encoder()   # fresh random weights, no pre-training
    acc_H, f1_H = finetune(encoder_H, finetune_X, finetune_y,
                            norm_test_X, test_y,
                            freeze_encoder=False,
                            epochs=args.finetune_epochs, lr=0.005,
                            label="[H random]")
    results["H_random"] = {
        "acc": round(acc_H, 4), "f1": round(f1_H, 4),
        "n_pretrain": 0, "n_finetune": len(finetune_X),
        "desc": f"Random init → finetune {int(label_pct*100)}% labels (no pretraining)",
        "time_s": round(time.time()-t0)
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Condition H_full — Random init, ALL training labels (upper bound)
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Condition H_full_labels  |  Random init → ALL labels (upper bound)")
    t0 = time.time()
    random.seed(42)
    encoder_Hf = Encoder()
    acc_Hf, f1_Hf = finetune(encoder_Hf, norm_all_train_X, all_train_y,
                              norm_test_X, test_y,
                              freeze_encoder=False,
                              epochs=args.finetune_epochs, lr=0.005,
                              label="[H_full]")
    results["H_full_labels"] = {
        "acc": round(acc_Hf, 4), "f1": round(f1_Hf, 4),
        "n_pretrain": 0, "n_finetune": len(norm_all_train_X),
        "desc": "Random init → ALL labels (100% supervised upper bound)",
        "time_s": round(time.time()-t0)
    }

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  S2S LEVEL 5 — SELF-SUPERVISED RESULTS")
    print(f"{'='*60}")
    print(f"\n  {'Condition':<28} {'F1':>7}  {'n_finetune':>10}  Description")
    print(f"  {'-'*70}")
    for k, v in results.items():
        print(f"  {k:<28} {v['f1']:>7.4f}  {v['n_finetune']:>10}  {v['desc']}")

    G  = results["G_pretrained"]["f1"]
    G2 = results["G2_pretrained_full"]["f1"]
    H  = results["H_random"]["f1"]
    Hf = results["H_full_labels"]["f1"]

    best_G = max(G, G2)

    print(f"\n  ┌─ LEVEL 5: Self-Supervised Pre-training ───────────────")
    print(f"  │  Random init  10% labels (H):        {H:.4f} F1")
    print(f"  │  S2S pretrain 10% labels (G frozen): {G:.4f} F1")
    print(f"  │  S2S pretrain 10% labels (G2 full):  {G2:.4f} F1")
    print(f"  │  Random init 100% labels (upper):    {Hf:.4f} F1")
    print(f"  │")
    print(f"  │  S2S pre-train gain vs random:       {(best_G-H)*100:+.2f}% F1")
    print(f"  │  Label efficiency: {int(label_pct*100)}% labels + S2S pretrain")
    print(f"  │  vs 100% labels baseline:            {(best_G-Hf)*100:+.2f}% F1")
    print(f"  │")
    proven = best_G > H
    print(f"  │  Verdict: {'✓ PROVEN' if proven else '✗ Not proven'}")
    print(f"  └────────────────────────────────────────────────────────")

    out = {
        "experiment":       "S2S Level 5 Self-Supervised Pre-training",
        "dataset":          "UCI HAR 50Hz 30 subjects 6 activities",
        "label_pct":        args.label_pct,
        "pretrain_windows": len(pretrain_pool),
        "finetune_windows": len(finetune_X),
        "test_windows":     len(test_y),
        "results":          results,
        "level5_gain_vs_random_f1": round(best_G - H, 4),
        "level5_proven":    proven,
        "total_time_s":     round(time.time() - t_start),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  Saved → {args.out}")

if __name__ == "__main__":
    main()
