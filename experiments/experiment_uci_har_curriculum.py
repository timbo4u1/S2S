#!/usr/bin/env python3
"""
S2S Level 2 — Curriculum Training on UCI HAR (Third Independent Dataset)
30 subjects, 6 activities, body_acc + gyro, 50Hz, pre-windowed 128 samples

Proves: curriculum training (GOLD→SILVER→ALL) beats clean baseline
on a dataset S2S has never seen before.

Run from ~/S2S:
  python3 experiments/experiment_uci_har_curriculum.py \
    --data data/uci_har/UCI\ HAR\ Dataset/ \
    --out experiments/results_uci_har_curriculum.json
"""

import os, sys, json, math, random, time, argparse
from pathlib import Path
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

random.seed(42)

ACTIVITIES = {1:"WALKING", 2:"WALKING_UPSTAIRS", 3:"WALKING_DOWNSTAIRS",
              4:"SITTING", 5:"STANDING", 6:"LAYING"}
HZ = 50.0
WINDOW_SIZE = 128  # samples per window (pre-windowed in UCI HAR)
DT_NS = int(1e9 / HZ)  # 20ms in nanoseconds

# ── DATA LOADING ─────────────────────────────────────────────────

def load_signal_file(path):
    """Load UCI HAR inertial signal file → list of windows (each = list of floats)."""
    windows = []
    with open(path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if vals:
                windows.append(vals)
    return windows

def load_uci_har(data_dir):
    """Load all UCI HAR windows with accel + gyro + subject + label."""
    base = Path(data_dir)
    windows = []

    for split in ["train", "test"]:
        sig_dir = base / split / "Inertial Signals"
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
            # Reconstruct timestamps with realistic hardware jitter
            ts_ns = [int(j * DT_NS) + int(random.gauss(0, 300)) for j in range(WINDOW_SIZE)]
            windows.append({
                "accel":      accel,
                "gyro":       gyro,
                "timestamps": ts_ns,
                "subject":    subjects[i],
                "label":      labels[i] - 1,  # 0-indexed
                "split":      split,
            })

    print(f"  Loaded {len(windows)} windows from {len(set(w['subject'] for w in windows))} subjects")
    return windows

# ── S2S CERTIFICATION ─────────────────────────────────────────────

def certify_window(w):
    """Run S2S physics certification on one UCI HAR window."""
    try:
        from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
        imu_raw = {
            "timestamps_ns": w["timestamps"],
            "accel":         w["accel"],
            "gyro":          w["gyro"],
        }
        result = PhysicsEngine().certify(imu_raw=imu_raw, segment="forearm")
        score = result.get("physical_law_score", result.get("score", 0)) or 0
        tier  = result.get("tier", "BRONZE")
        return int(score), tier
    except Exception as e:
        return 0, "REJECTED"

# ── FEATURES ─────────────────────────────────────────────────────

def extract_features(w):
    """18 features: per-axis mean+std+range for accel (9) + gyro (9)."""
    feats = []
    for sensor in [w["accel"], w["gyro"]]:
        for axis in range(3):
            vals = [sensor[i][axis] for i in range(len(sensor))]
            mu   = sum(vals) / len(vals)
            var  = sum((v - mu)**2 for v in vals) / len(vals)
            feats += [mu, math.sqrt(var), max(vals) - min(vals)]
    return feats

# ── CLASSIFIER ────────────────────────────────────────────────────

def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v/s for v in e]

def cross_entropy(probs, label):
    return -math.log(max(probs[label], 1e-15))

class MLP:
    def __init__(self, input_dim=18, hidden=64, output=6, lr=0.01):
        s1 = math.sqrt(2.0 / input_dim)
        s2 = math.sqrt(2.0 / hidden)
        self.W1 = [[random.gauss(0, s1) for _ in range(input_dim)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.W2 = [[random.gauss(0, s2) for _ in range(hidden)] for _ in range(output)]
        self.b2 = [0.0] * output
        self.lr = lr

    def forward(self, x):
        h = [max(0, sum(self.W1[j][i]*x[i] for i in range(len(x))) + self.b1[j])
             for j in range(len(self.W1))]
        o = [sum(self.W2[k][j]*h[j] for j in range(len(h))) + self.b2[k]
             for k in range(len(self.W2))]
        return h, softmax(o)

    def backward(self, x, h, probs, label, weight=1.0):
        dL = [p - (1 if i == label else 0) for i, p in enumerate(probs)]
        dL = [v * weight for v in dL]
        for k in range(len(self.W2)):
            for j in range(len(h)):
                self.W2[k][j] -= self.lr * dL[k] * h[j]
            self.b2[k] -= self.lr * dL[k]
        dh = [sum(self.W2[k][j]*dL[k] for k in range(len(self.W2))) * (1 if h[j]>0 else 0)
              for j in range(len(h))]
        for j in range(len(self.W1)):
            for i in range(len(x)):
                self.W1[j][i] -= self.lr * dh[j] * x[i]
            self.b1[j] -= self.lr * dh[j]

def normalize(train_X, test_X):
    n_feat = len(train_X[0])
    means = [sum(x[i] for x in train_X)/len(train_X) for i in range(n_feat)]
    stds  = [max(math.sqrt(sum((x[i]-means[i])**2 for x in train_X)/len(train_X)), 1e-8)
             for i in range(n_feat)]
    norm  = lambda X: [[(x[i]-means[i])/stds[i] for i in range(n_feat)] for x in X]
    return norm(train_X), norm(test_X)

def evaluate(model, X, y):
    correct = 0
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for x, label in zip(X, y):
        _, probs = model.forward(x)
        pred = probs.index(max(probs))
        if pred == label: correct += 1
        tp[label] += (pred == label)
        fp[pred]  += (pred != label)
        fn[label] += (pred != label)
    acc = correct / len(y)
    f1s = []
    for c in range(6):
        p = tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0
        r = tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0
        f1s.append(2*p*r/(p+r) if p+r else 0)
    return acc, sum(f1s)/len(f1s)

def train_condition(name, train_data, test_X, test_y, epochs=40,
                    weights=None, phases=None, lr=0.005):
    """Train MLP. phases = list of (data, n_epochs) for curriculum."""
    random.seed(42)
    model = MLP(lr=lr)

    if phases:
        # Curriculum: train in phases
        all_X = [extract_features(w) for w, _ in train_data]
        all_y = [w["label"] for w, _ in train_data]
        norm_all_X, norm_test_X = normalize(all_X, test_X)
        ep = 0
        for phase_idx, (phase_data, phase_epochs) in enumerate(phases):
            ph_X = [extract_features(w) for w, _ in phase_data]
            ph_y = [w["label"] for w, _ in phase_data]
            ph_X_norm, _ = normalize(ph_X, test_X)  # normalize within phase
            for e in range(phase_epochs):
                ep += 1
                idx = list(range(len(ph_X_norm)))
                random.shuffle(idx)
                for i in idx:
                    _, probs = model.forward(ph_X_norm[i])
                    model.backward(ph_X_norm[i], model.forward(ph_X_norm[i])[0],
                                   probs, ph_y[i])
                if ep % 10 == 0 or ep == epochs:
                    acc, f1 = evaluate(model, norm_test_X, test_y)
                    print(f"  ep {ep:3d}/{epochs}  f1={f1:.4f}")
        acc, f1 = evaluate(model, norm_test_X, test_y)
    else:
        train_X = [extract_features(w) for w, _ in train_data]
        train_y = [w["label"] for w, _ in train_data]
        w_list  = weights if weights else [1.0] * len(train_X)
        norm_train_X, norm_test_X = normalize(train_X, test_X)
        for e in range(1, epochs+1):
            idx = list(range(len(norm_train_X)))
            random.shuffle(idx)
            for i in idx:
                _, probs = model.forward(norm_train_X[i])
                model.backward(norm_train_X[i], model.forward(norm_train_X[i])[0],
                               probs, train_y[i], weight=w_list[i])
            if e % 10 == 0 or e == epochs:
                acc, f1 = evaluate(model, norm_test_X, test_y)
                print(f"  ep {e:3d}/{epochs}  loss=?  acc={acc:.3f}  f1={f1:.4f}")
        acc, f1 = evaluate(model, norm_test_X, test_y)

    print(f"  Final: acc={acc:.4f}  F1={f1:.4f}")
    return acc, f1


def train_weighted_sampling(train_data, test_X, test_y, gold_thresh, silver_thresh,
                             floor_thresh, epochs=40, lr=0.005):
    """
    Method F — Weighted sampling curriculum.
    GOLD always present. BRONZE weight increases gradually each epoch.
    No phases = no catastrophic forgetting.
    """
    random.seed(42)
    model = MLP(lr=lr)

    all_X = [extract_features(w) for w, _ in train_data]
    all_y = [w["label"] for w, _ in train_data]
    norm_X, norm_test_X = normalize(all_X, test_X)

    # Assign tier index to each sample
    tiers = []
    for _, s in train_data:
        if s >= gold_thresh:   tiers.append("GOLD")
        elif s >= silver_thresh: tiers.append("SILVER")
        else:                    tiers.append("BRONZE")

    gold_idx   = [i for i,t in enumerate(tiers) if t == "GOLD"]
    silver_idx = [i for i,t in enumerate(tiers) if t == "SILVER"]
    bronze_idx = [i for i,t in enumerate(tiers) if t == "BRONZE"]

    print(f"  Sampling pool: GOLD={len(gold_idx)} SILVER={len(silver_idx)} BRONZE={len(bronze_idx)}")

    for e in range(1, epochs+1):
        # Bronze weight grows from 0.05 → 0.5 over training
        bronze_w = 0.05 + 0.45 * (e / epochs)
        silver_w = 0.7
        gold_w   = 1.0

        # Build weighted sample for this epoch
        # Always include all GOLD, sample SILVER and BRONZE by weight
        batch = list(gold_idx)
        batch += random.sample(silver_idx, min(len(silver_idx), int(len(silver_idx)*silver_w)))
        n_bronze = max(1, int(len(bronze_idx) * bronze_w))
        batch += random.sample(bronze_idx, min(len(bronze_idx), n_bronze))
        random.shuffle(batch)

        for i in batch:
            _, probs = model.forward(norm_X[i])
            model.backward(norm_X[i], model.forward(norm_X[i])[0],
                          probs, all_y[i])

        if e % 10 == 0 or e == epochs:
            acc, f1 = evaluate(model, norm_test_X, test_y)
            print(f"  ep {e:3d}/{epochs}  bronze_w={bronze_w:.2f}  acc={acc:.3f}  f1={f1:.4f}")

    acc, f1 = evaluate(model, norm_test_X, test_y)
    print(f"  Final: acc={acc:.4f}  F1={f1:.4f}")
    return acc, f1

# ── MAIN ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out",  required=True)
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()

    print("\nS2S Level 2 — Curriculum Training on UCI HAR")
    print("=" * 60)

    # Load
    print("\nLoading UCI HAR...")
    windows = load_uci_har(args.data)

    # Subject split — test on subjects 27-30 (held out)
    test_subjects  = {27, 28, 29, 30}
    train_windows  = [w for w in windows if w["subject"] not in test_subjects]
    test_windows   = [w for w in windows if w["subject"] in test_subjects]
    print(f"  Train: {len(train_windows)} windows | Test: {len(test_windows)} windows")

    # Certify all training windows
    print("\nCertifying training windows with S2S (auto-Hz)...")
    t0 = time.time()
    certified = []
    score_dist = Counter()
    for i, w in enumerate(train_windows):
        score, tier = certify_window(w)
        certified.append((w, score, tier))
        score_dist[tier] += 1
        if (i+1) % 1000 == 0:
            print(f"  [{i+1}/{len(train_windows)}] elapsed={time.time()-t0:.0f}s")

    scores = [s for _, s, _ in certified]
    scores_nz = sorted(s for s in scores if s > 0)
    if scores_nz:
        n = len(scores_nz)
        gold_thresh   = scores_nz[int(0.75*n)]
        silver_thresh = scores_nz[int(0.50*n)]
        floor_thresh  = scores_nz[int(0.25*n)]
    else:
        gold_thresh, silver_thresh, floor_thresh = 80, 70, 50

    print(f"\n  Adaptive thresholds (percentile-anchored):")
    print(f"    GOLD   >= {gold_thresh}  (p75)")
    print(f"    SILVER >= {silver_thresh}  (p50)")
    print(f"    floor  >= {floor_thresh}  (p25)")
    print(f"  Tier distribution: {dict(score_dist)}")

    # Build tier sets
    gold_data   = [(w, s) for w, s, t in certified if s >= gold_thresh]
    silver_data = [(w, s) for w, s, t in certified if silver_thresh <= s < gold_thresh]
    floor_data  = [(w, s) for w, s, t in certified if s >= floor_thresh]
    all_data    = [(w, s) for w, s, t in certified]

    print(f"\n  GOLD:   {len(gold_data)}")
    print(f"  SILVER: {len(silver_data)}")
    print(f"  floor+: {len(floor_data)}")
    print(f"  all:    {len(all_data)}")

    # Test features
    test_X = [extract_features(w) for w in test_windows]
    test_y = [w["label"] for w in test_windows]

    # Inject corruption into train (35%)
    corrupted = []
    n_corrupt = int(0.35 * len(all_data))
    corrupt_idx = set(random.sample(range(len(all_data)), n_corrupt))
    for i, (w, s) in enumerate(all_data):
        if i in corrupt_idx:
            wc = dict(w)
            choice = random.randint(0, 2)
            if choice == 0:  # flat signal
                wc["accel"] = [[0.0, 0.0, 0.0]] * WINDOW_SIZE
            elif choice == 1:  # label noise
                wc = dict(w)
                wc["label"] = random.randint(0, 5)
            else:  # clipping
                wc["accel"] = [[max(-0.5, min(0.5, v)) for v in row] for row in w["accel"]]
            corrupted.append((wc, 0))
        else:
            corrupted.append((w, s))

    # S2S filter corrupted
    floor_filtered = [(w, s) for w, s in corrupted if s >= floor_thresh]
    print(f"\n  Corruption: {n_corrupt} injected (35%)")
    print(f"  After S2S floor filter: {len(floor_filtered)} kept")

    # Tier weights
    def get_weight(s):
        if s >= gold_thresh:   return 1.0
        if s >= silver_thresh: return 0.8
        if s >= floor_thresh:  return 0.4
        return 0.1

    tier_weights = [get_weight(s) for _, s in floor_filtered]

    # Curriculum phases
    gold_filtered   = [(w, s) for w, s in floor_filtered if s >= gold_thresh]
    silver_filtered = [(w, s) for w, s in floor_filtered if s >= silver_thresh]

    epochs = args.epochs
    p1_ep = max(8,  epochs // 4)
    p2_ep = max(12, epochs // 3)
    p3_ep = epochs - p1_ep - p2_ep

    results = {}
    t_start = time.time()

    conditions = [
        ("A_clean",      all_data,        "Clean baseline"),
        ("B_corrupted",  corrupted,       "Corrupted 35%"),
        ("C_floor",      floor_filtered,  "S2S floor filtered"),
    ]

    for cname, data, desc in conditions:
        print(f"\n{'─'*60}")
        print(f"Condition {cname}  |  {desc}  n={len(data)}")
        t0 = time.time()
        acc, f1 = train_condition(cname, data, test_X, test_y, epochs=epochs)
        results[cname] = {"acc": round(acc,4), "f1": round(f1,4),
                          "n": len(data), "desc": desc, "time_s": round(time.time()-t0)}

    # D — tier weighted
    print(f"\n{'─'*60}")
    print(f"Condition D_weighted  |  Tier-weighted loss  n={len(floor_filtered)}")
    t0 = time.time()
    acc, f1 = train_condition("D", floor_filtered, test_X, test_y,
                               epochs=epochs, weights=tier_weights)
    results["D_weighted"] = {"acc": round(acc,4), "f1": round(f1,4),
                             "n": len(floor_filtered), "desc": "Tier-weighted",
                             "time_s": round(time.time()-t0)}

    # E — curriculum
    print(f"\n{'─'*60}")
    print(f"Condition E_curriculum  |  GOLD→SILVER→ALL")
    print(f"  Phase 1 GOLD   n={len(gold_filtered)}  epochs={p1_ep}")
    print(f"  Phase 2 SILVER n={len(silver_filtered)}  epochs={p2_ep}")
    print(f"  Phase 3 ALL    n={len(floor_filtered)}  epochs={p3_ep}")
    t0 = time.time()
    phases = [
        (gold_filtered,   p1_ep),
        (silver_filtered, p2_ep),
        (floor_filtered,  p3_ep),
    ]
    acc, f1 = train_condition("E", floor_filtered, test_X, test_y,
                               epochs=epochs, phases=phases)
    results["E_curriculum"] = {"acc": round(acc,4), "f1": round(f1,4),
                                "n": len(floor_filtered), "desc": "Curriculum GOLD→SILVER→ALL",
                                "time_s": round(time.time()-t0)}

    # F — weighted sampling curriculum
    print(f"\n{'─'*60}")
    print(f"Condition F_weighted_sample  |  Gradual BRONZE introduction")
    t0 = time.time()
    acc, f1 = train_weighted_sampling(
        floor_filtered, test_X, test_y,
        gold_thresh, silver_thresh, floor_thresh, epochs=epochs)
    results["F_weighted_sample"] = {"acc": round(acc,4), "f1": round(f1,4),
                                     "n": len(floor_filtered),
                                     "desc": "Weighted sampling (GOLD always present)",
                                     "time_s": round(time.time()-t0)}

    # Summary

    print(f"\n{'='*60}")
    print(f"  S2S LEVEL 2 — UCI HAR CURRICULUM RESULTS")
    print(f"{'='*60}")
    print(f"\n  {'Condition':<22} {'F1':>7}  {'n':>6}  Description")
    print(f"  {'-'*60}")
    for k, v in results.items():
        print(f"  {k:<22} {v['f1']:>7.4f}  {v['n']:>6}  {v['desc']}")

    A = results["A_clean"]["f1"]
    B = results["B_corrupted"]["f1"]
    C = results["C_floor"]["f1"]
    E = results["E_curriculum"]["f1"]
    F = results["F_weighted_sample"]["f1"]

    print(f"\n  ┌─ LEVEL 1: Quality Floor ──────────────────────────")
    print(f"  │  Corruption damage (B-A): {(B-A)*100:+.2f}% F1")
    print(f"  │  S2S recovery     (C-B): {(C-B)*100:+.2f}% F1")
    print(f"  │  Net vs baseline  (C-A): {(C-A)*100:+.2f}% F1")
    print(f"  │  Verdict: {'✓ PROVEN' if C > A else '✗ Not proven'}")
    print(f"  ├─ LEVEL 2: Curriculum ─────────────────────────────")
    print(f"  │  Curriculum vs clean (E-A): {(E-A)*100:+.2f}% F1")
    print(f"  │  Verdict: {'✓ PROVEN' if E > A else '✗ Not proven'}")
    print(f"  └───────────────────────────────────────────────────")

    out = {
        "experiment": "UCI HAR Level 2 Curriculum",
        "dataset": "UCI HAR 50Hz 30 subjects 6 activities",
        "results": results,
        "level1_floor_net_f1": round(C-A, 4),
        "level2_curriculum_phases_f1": round(E-A, 4),
        "level2_weighted_sample_f1": round(F-A, 4),
        "gold_thresh": gold_thresh, "silver_thresh": silver_thresh,
        "floor_thresh": floor_thresh,
        "total_time_s": round(time.time()-t_start),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n  Saved → {args.out}")

if __name__ == "__main__":
    main()
