#!/usr/bin/env python3
"""
S2S Benchmark v4 — WISDM raw sensor windows
Honest experiment: raw accel features, subject-based split, no physics circularity.

Conditions:
  A = all WISDM records (baseline)
  B = certified only (SILVER + GOLD, physics_score >= 60)
  C = random subsample same size as B (controls for quantity)

Train/test split: by subject_id — no subject leakage.

Run: python3 experiments/wisdm_benchmark.py --dataset s2s_dataset/ --out experiments/results_wisdm.json
"""

import os, sys, json, math, random, time, argparse, glob
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOMAINS   = ['DAILY_LIVING', 'LOCOMOTION', 'PRECISION', 'SOCIAL', 'SPORT']
DOM2IDX   = {d: i for i, d in enumerate(DOMAINS)}
random.seed(42)

# ── RAW FEATURE EXTRACTION ────────────────────────────────────────
# Pure signal features from accel windows — no physics metadata used

def extract_raw_features(imu_raw):
    """
    14 features from raw accel only:
      per-axis mean, std, range (3×3=9)
      accel magnitude mean, std, max (3)
      dominant FFT frequency (1)
      zero-crossing rate (1)
    """
    accel = imu_raw.get('accel', [])
    if len(accel) < 10:
        return None

    n = len(accel)
    axes = [[accel[i][j] for i in range(n)] for j in range(3)]

    def mean(v): return sum(v) / len(v)
    def std(v):
        m = mean(v)
        return math.sqrt(sum((x - m)**2 for x in v) / len(v))
    def rng(v): return max(v) - min(v)

    feats = []
    for ax in axes:
        feats += [mean(ax), std(ax), rng(ax)]

    # Magnitude
    mag = [math.sqrt(sum(accel[i][j]**2 for j in range(3))) for i in range(n)]
    feats += [mean(mag), std(mag), max(mag)]

    # Dominant FFT frequency (simplified — find peak in first half)
    if n >= 16:
        # Manual DFT for top frequency (no numpy)
        half = n // 2
        magnitudes = []
        for k in range(1, min(half, 20)):
            re = sum(mag[t] * math.cos(2 * math.pi * k * t / n) for t in range(n))
            im = sum(mag[t] * math.sin(2 * math.pi * k * t / n) for t in range(n))
            magnitudes.append((re**2 + im**2, k))
        dom_freq = max(magnitudes)[1] / n if magnitudes else 0.0
    else:
        dom_freq = 0.0
    feats.append(dom_freq)

    # Zero crossing rate of x-axis
    zcr = sum(1 for i in range(1, n) if axes[0][i-1] * axes[0][i] < 0) / n
    feats.append(zcr)

    return feats  # 14 features


def normalize(X_train, X_test):
    """Z-score normalize using train statistics."""
    n_feat = len(X_train[0])
    means, stds = [], []
    for j in range(n_feat):
        col = [x[j] for x in X_train]
        m = sum(col) / len(col)
        s = math.sqrt(sum((v - m)**2 for v in col) / len(col)) or 1.0
        means.append(m); stds.append(s)

    def norm(X):
        return [[(x[j] - means[j]) / stds[j] for j in range(n_feat)] for x in X]

    return norm(X_train), norm(X_test)


# ── MLP (same as v3 but with normalized input) ────────────────────
def softmax(x):
    m = max(x)
    e = [math.exp(max(-60.0, min(60.0, v - m))) for v in x]
    s = sum(e)
    if s == 0 or s != s:
        return [1.0 / len(x)] * len(x)
    return [v / s for v in e]

def clip(x, lo, hi): return max(lo, min(hi, x))


class MLP:
    def __init__(self, input_dim, hidden=64, output=5, lr=0.01):
        self.lr = lr
        s1 = math.sqrt(2.0 / input_dim)  # He init
        s2 = math.sqrt(2.0 / hidden)
        self.W1 = [[random.gauss(0, s1) for _ in range(input_dim)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.W2 = [[random.gauss(0, s2) for _ in range(hidden)] for _ in range(output)]
        self.b2 = [0.0] * output
        self.gW1 = [[1e-8] * input_dim for _ in range(hidden)]
        self.gb1 = [1e-8] * hidden
        self.gW2 = [[1e-8] * hidden for _ in range(output)]
        self.gb2 = [1e-8] * output

    def forward(self, x):
        h = [max(0.0, min(50.0, sum(self.W1[j][i] * x[i] for i in range(len(x))) + self.b1[j]))
             for j in range(len(self.b1))]
        logits = [sum(self.W2[k][j] * h[j] for j in range(len(h))) + self.b2[k]
                  for k in range(len(self.b2))]
        return h, softmax(logits)

    def backward(self, x, h, probs, label):
        no, nh = len(self.b2), len(self.b1)
        dl = [clip(p - (1.0 if i == label else 0.0), -2, 2) for i, p in enumerate(probs)]
        for k in range(no):
            for j in range(nh):
                g = dl[k] * h[j]
                self.gW2[k][j] += g * g
                self.W2[k][j] -= self.lr / math.sqrt(self.gW2[k][j]) * g
            self.gb2[k] += dl[k] * dl[k]
            self.b2[k] -= self.lr / math.sqrt(self.gb2[k]) * dl[k]
        dh = [clip(sum(self.W2[k][j] * dl[k] for k in range(no)), -2, 2) * (1 if h[j] > 0 else 0)
              for j in range(nh)]
        for j in range(nh):
            for i in range(len(x)):
                g = dh[j] * x[i]
                self.gW1[j][i] += g * g
                self.W1[j][i] -= self.lr / math.sqrt(self.gW1[j][i]) * g
            self.gb1[j] += dh[j] * dh[j]
            self.b1[j] -= self.lr / math.sqrt(self.gb1[j]) * dh[j]

    def _clip_weights(self, max_norm=5.0):
        for row in self.W1:
            n = math.sqrt(sum(v * v for v in row))
            if n > max_norm:
                s = max_norm / n
                for i in range(len(row)): row[i] *= s
        for row in self.W2:
            n = math.sqrt(sum(v * v for v in row))
            if n > max_norm:
                s = max_norm / n
                for i in range(len(row)): row[i] *= s

    def train_epoch(self, samples):
        random.shuffle(samples)
        loss, nan_count = 0.0, 0
        for feat, label in samples:
            h, probs = self.forward(feat)
            step_loss = -math.log(max(probs[label], 1e-10))
            if step_loss != step_loss:
                nan_count += 1
                continue
            loss += step_loss
            self.backward(feat, h, probs, label)
        self._clip_weights()
        return loss / max(len(samples) - nan_count, 1)

    def evaluate(self, samples):
        correct = 0
        per_class = defaultdict(lambda: [0, 0])  # [correct, total]
        for feat, label in samples:
            _, probs = self.forward(feat)
            pred = probs.index(max(probs))
            per_class[label][1] += 1
            if pred == label:
                correct += 1
                per_class[label][0] += 1
        acc = correct / max(len(samples), 1)
        f1_scores = []
        for label_idx in range(len(DOMAINS)):
            c, t = per_class[label_idx]
            f1_scores.append(c / max(t, 1))
        macro_f1 = sum(f1_scores) / len(f1_scores)
        per_domain_f1 = {DOMAINS[i]: round(f1_scores[i], 4) for i in range(len(DOMAINS))}
        return acc, macro_f1, per_domain_f1


# ── DATA LOADING ──────────────────────────────────────────────────

def load_wisdm(dataset_dir):
    """Load WISDM records with raw sensor data. Returns list of (features, label, subject, score)."""
    samples = []
    skipped = 0
    print("Loading WISDM records with raw sensor data...")

    for fpath in glob.glob(os.path.join(dataset_dir, '**', '*.json'), recursive=True):
        try:
            with open(fpath) as f:
                rec = json.load(f)
        except Exception:
            continue

        # Only WISDM records with raw data
        if rec.get('dataset', rec.get('dataset_source', '')) != 'WISDM_2019':
            continue

        imu_raw = rec.get('imu_raw', {})
        if not imu_raw or not imu_raw.get('accel'):
            continue

        domain = rec.get('domain', '')
        if domain not in DOM2IDX:
            continue

        feats = extract_raw_features(imu_raw)
        if feats is None:
            skipped += 1
            continue

        cert = rec.get('certification', {})
        score = cert.get('physical_law_score', rec.get('physics_score', 0))
        subject = rec.get('subject_id', 'unknown')

        samples.append((feats, DOM2IDX[domain], subject, score))

    print(f"  Loaded {len(samples)} samples ({skipped} skipped, no raw data)")
    return samples


def subject_split(samples, test_fraction=0.2):
    """Split by subject ID — no subject appears in both train and test."""
    subjects = list(set(s[2] for s in samples))
    random.shuffle(subjects)
    n_test = max(1, int(len(subjects) * test_fraction))
    test_subjects = set(subjects[:n_test])
    train = [(f, l) for f, l, s, _ in samples if s not in test_subjects]
    test  = [(f, l) for f, l, s, _ in samples if s in test_subjects]
    print(f"  Train subjects: {len(subjects)-n_test}  Test subjects: {n_test}")
    print(f"  Train samples:  {len(train)}  Test samples: {len(test)}")
    return train, test


# ── MAIN ──────────────────────────────────────────────────────────

def run(dataset_dir, output_path, epochs=30):
    print("S2S Benchmark v4 — WISDM raw sensor windows")
    print("Features: 14 raw accel features (no physics metadata)")
    print("Split:    by subject ID (no leakage)")
    print("=" * 52)

    all_samples = load_wisdm(dataset_dir)
    if not all_samples:
        print("ERROR: No WISDM records with raw data found.")
        sys.exit(1)

    # Domain distribution
    domain_counts = Counter(DOMAINS[s[1]] for s in all_samples)
    cert_counts   = Counter(DOMAINS[s[1]] for s in all_samples if s[3] >= 60)
    print(f"\nDomain distribution (all / certified):")
    for d in DOMAINS:
        print(f"  {d:<15} {domain_counts.get(d,0):>6} / {cert_counts.get(d,0):>6}")

    # Subject-based split
    print("\nSubject-based train/test split:")
    train_all, test = subject_split(all_samples)

    # Subject-based split (deterministic)
    subjects = sorted(set(s[2] for s in all_samples))
    random.seed(42)
    random.shuffle(subjects)
    n_test = max(1, int(len(subjects) * 0.2))
    test_subject_ids = set(subjects[:n_test])

    train_all  = [(f, l) for f, l, s, _  in all_samples if s not in test_subject_ids]
    train_cert = [(f, l) for f, l, s, sc in all_samples if s not in test_subject_ids and sc >= 60]
    test       = [(f, l) for f, l, s, _  in all_samples if s in test_subject_ids]

    # Random subsample same size as certified
    random.seed(42)
    train_rand = random.sample(train_all, min(len(train_cert), len(train_all)))

    print(f"\n  A (all):      {len(train_all):>6} train samples")
    print(f"  B (certified):{len(train_cert):>6} train samples  (score >= 60)")
    print(f"  C (random):   {len(train_rand):>6} train samples  (same n as B)")
    print(f"  Test:         {len(test):>6} samples")

    # Normalize using train_all statistics
    X_all,  _ = normalize([f for f, _ in train_all],  [f for f, _ in test])
    _, X_test  = normalize([f for f, _ in train_all],  [f for f, _ in test])
    X_cert, _  = normalize([f for f, _ in train_cert], [f for f, _ in test])
    _, X_test_cert = normalize([f for f, _ in train_cert], [f for f, _ in test])
    X_rand, _  = normalize([f for f, _ in train_rand], [f for f, _ in test])
    _, X_test_rand = normalize([f for f, _ in train_rand], [f for f, _ in test])

    # Use train_all normalization for all test sets (consistent)
    X_all_norm,  X_test_norm = normalize([f for f, _ in train_all], [f for f, _ in test])
    X_cert_norm, _           = normalize([f for f, _ in train_cert], [f for f, _ in test])
    X_rand_norm, _           = normalize([f for f, _ in train_rand], [f for f, _ in test])

    train_sets = [
        ("A_all_data",         list(zip(X_all_norm,  [l for _, l in train_all]))),
        ("B_certified",        list(zip(X_cert_norm, [l for _, l in train_cert]))),
        ("C_random_subsample", list(zip(X_rand_norm, [l for _, l in train_rand]))),
    ]
    test_set = list(zip(X_test_norm, [l for _, l in test]))

    results = {}
    for cond, train_data in train_sets:
        print(f"\n{'─'*52}")
        print(f"Condition {cond}  n={len(train_data)}")
        model = MLP(input_dim=14, hidden=64, output=5, lr=0.01)
        t0 = time.time()
        for ep in range(epochs):
            loss = model.train_epoch(train_data)
            if (ep + 1) % 10 == 0:
                acc, f1, _ = model.evaluate(test_set)
                print(f"  Epoch {ep+1:3d}/{epochs}  loss={loss:.4f}  acc={acc:.3f}  f1={f1:.3f}")
        acc, f1, pf1 = model.evaluate(test_set)
        elapsed = time.time() - t0
        print(f"  Final: {acc:.4f} ({acc*100:.1f}%)  F1={f1:.4f}  [{elapsed:.0f}s]")
        results[cond] = {
            "accuracy":      round(acc, 4),
            "macro_f1":      round(f1, 4),
            "per_domain_f1": pf1,
            "train_n":       len(train_data),
            "test_n":        len(test_set),
        }

    print(f"\n{'='*52}  SUMMARY")
    for c, r in results.items():
        print(f"  {c}: {r['accuracy']:.4f} acc  {r['macro_f1']:.4f} f1  n={r['train_n']}")

    if 'B_certified' in results and 'C_random_subsample' in results:
        da = results['B_certified']['accuracy'] - results['C_random_subsample']['accuracy']
        df = results['B_certified']['macro_f1']  - results['C_random_subsample']['macro_f1']
        print(f"\n  ┌─ CERTIFICATION QUALITY EFFECT (B vs C, same n) ─────────")
        print(f"  │  Accuracy: {'+' if da>=0 else ''}{da*100:.2f}%")
        print(f"  │  Macro F1: {'+' if df>=0 else ''}{df*100:.2f}%")
        verdict = "✓ Certified data is higher quality" if df > 0.005 else \
                  "~ No significant quality effect" if abs(df) <= 0.005 else \
                  "✗ Random data outperforms certified"
        print(f"  │  Verdict:  {verdict}")
        print(f"  └───────────────────────────────────────────────────────────")

    out = {
        "experiment":  "s2s_v4_wisdm_raw",
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "features":    "14 raw accel features (mean/std/range per axis, magnitude stats, FFT, ZCR)",
        "split":       "subject-based (no leakage)",
        "conditions":  results,
        "note":        "A=all WISDM, B=certified(score>=60), C=random same size as B. "
                       "B vs C isolates data quality from quantity.",
    }
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='s2s_dataset/')
    p.add_argument('--out',     default='experiments/results_wisdm.json')
    p.add_argument('--epochs',  type=int, default=30)
    args = p.parse_args()
    run(args.dataset, args.out, args.epochs)
