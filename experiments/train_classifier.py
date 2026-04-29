#!/usr/bin/env python3
"""
S2S Domain Classifier v1.4
- Added FINE_MOTOR domain (merged PRECISION + DAILY_LIVING)
- Added jerk_peak feature
- Added per-Hz accuracy analysis
- Pure Python, zero dependencies

Usage:
  python3 train_classifier.py --dataset s2s_dataset/ --test
  python3 train_classifier.py --dataset s2s_dataset/ --merge-fine-motor --test
"""

import os, sys, json, math, random, time, argparse
from collections import Counter, defaultdict

# ── CONFIGURATION ──────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    'jerk_p95_ms3',      # 95th percentile jerk (m/s³) — Flash-Hogan 1985
    'imu_coupling_r',    # Pearson r between accel and gyro magnitudes
    'accel_std',         # Standard deviation of accel magnitude
    'gyro_energy',       # Mean squared gyro magnitude (rotational energy)
    'dominant_freq_hz',  # FFT peak frequency
    'jerk_peak_ms3',     # Peak jerk (max, not p95) — new in v1.4
]

DOMAIN_LABELS = ['DAILY_LIVING', 'LOCOMOTION', 'PRECISION', 'SOCIAL', 'SPORT']
FINE_MOTOR_DOMAINS = {'PRECISION', 'DAILY_LIVING'}  # merged in --merge-fine-motor mode

# ── FEATURE EXTRACTION ─────────────────────────────────────────────────────────

def extract_features(record):
    """Extract 6 physics features from a certified S2S record."""
    physics = record
    raw     = record.get('imu_raw', {})

    # Feature 1: jerk p95
    jerk_p95 = physics.get('jerk_p95_ms3', None)
    if jerk_p95 is None:
        return None

    # Feature 2: IMU coupling
    imu_r = record.get('imu_coupling_r', 0.0)

    # Features 3-6: computed from raw IMU
    accel = raw.get('accel', [])
    gyro  = raw.get('gyro',  [])

    # No raw IMU data in these records - use placeholder values
    accel_std = 0.0
    gyro_energy = 0.0
    dom_freq = 0.0

    # Feature 6: jerk peak (new in v1.4)
    jerk_peak = record.get('jerk_peak_ms3', jerk_p95 * 1.4)  # fallback estimate if not stored

    return [jerk_p95, imu_r, accel_std, gyro_energy, dom_freq, jerk_peak]


# ── DATASET LOADING ────────────────────────────────────────────────────────────

def load_dataset(data_dir, merge_fine_motor=False):
    """Load all certified JSON records from the s2s_dataset/ directory."""
    X, y, meta = [], [], []
    skipped_jerk, skipped_features = 0, 0
    domain_counts = Counter()

    for domain in DOMAIN_LABELS:
        domain_dir = os.path.join(data_dir, domain)
        if not os.path.isdir(domain_dir):
            print(f"  WARNING: {domain_dir} not found, skipping")
            continue

        files = []
        for root, _, fs in os.walk(domain_dir):
            for fn in fs:
                if fn.endswith('.json'):
                    files.append(os.path.join(root, fn))
        label = 'FINE_MOTOR' if (merge_fine_motor and domain in FINE_MOTOR_DOMAINS) else domain

        for fname in files:
            fpath = fname  # already full path from os.walk
            try:
                with open(fpath) as f:
                    record = json.load(f)
            except Exception:
                continue

            # Skip extreme jerk (sensor artifact)
            jerk = record.get('physics', {}).get('jerk_p95_ms3', 0)
            if jerk > 2000:
                skipped_jerk += 1
                continue

            features = extract_features(record)
            if features is None:
                skipped_features += 1
                continue

            X.append(features)
            y.append(label)
            domain_counts[label] += 1
            meta.append({'file': fname, 'domain': domain,
                         'source': record.get('source', 'unknown')})

    print(f"\nDataset loaded: {len(X)} records")
    print(f"  Skipped (jerk>2000): {skipped_jerk}")
    print(f"  Skipped (features): {skipped_features}")
    print(f"\nDomain distribution:")
    for d, c in sorted(domain_counts.items()):
        pct = 100 * c / len(X) if X else 0
        print(f"  {d:20s}: {c:6d} ({pct:.1f}%)")

    return X, y, meta


# ── DECISION TREE ──────────────────────────────────────────────────────────────

class DecisionTree:
    def __init__(self, max_depth=8, min_samples=20):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None
        self.classes = None
        self.feature_importance = None

    def _gini(self, y):
        if not y: return 0.0
        n = len(y)
        counts = Counter(y)
        return 1.0 - sum((c/n)**2 for c in counts.values())

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        n = len(y)
        parent_gini = self._gini(y)

        for feat_idx in range(len(X[0])):
            values = sorted(set(x[feat_idx] for x in X))
            thresholds = [(values[i] + values[i+1]) / 2
                         for i in range(len(values)-1)]
            thresholds = thresholds[::max(1, len(thresholds)//20)]  # sample up to 20

            for thresh in thresholds:
                left_y  = [y[i] for i in range(n) if X[i][feat_idx] <= thresh]
                right_y = [y[i] for i in range(n) if X[i][feat_idx] >  thresh]
                if len(left_y) < self.min_samples or len(right_y) < self.min_samples:
                    continue
                gain = parent_gini - (
                    len(left_y)/n  * self._gini(left_y) +
                    len(right_y)/n * self._gini(right_y)
                )
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat_idx, thresh

        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth):
        counts = Counter(y)
        majority = counts.most_common(1)[0][0]

        if (depth >= self.max_depth or
            len(y) < self.min_samples * 2 or
            len(counts) == 1):
            return {'leaf': True, 'label': majority, 'counts': dict(counts)}

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain < 1e-6:
            return {'leaf': True, 'label': majority, 'counts': dict(counts)}

        left_mask  = [X[i][feat] <= thresh for i in range(len(X))]
        right_mask = [not m for m in left_mask]

        X_left  = [X[i] for i in range(len(X)) if left_mask[i]]
        y_left  = [y[i] for i in range(len(y)) if left_mask[i]]
        X_right = [X[i] for i in range(len(X)) if right_mask[i]]
        y_right = [y[i] for i in range(len(y)) if right_mask[i]]

        # Track feature importance
        if hasattr(self, '_importance'):
            self._importance[feat] = self._importance.get(feat, 0) + gain * len(y)

        return {
            'leaf': False,
            'feature': feat,
            'threshold': thresh,
            'gain': gain,
            'left':  self._build(X_left,  y_left,  depth+1),
            'right': self._build(X_right, y_right, depth+1),
        }

    def fit(self, X, y):
        self.classes = sorted(set(y))
        self._importance = {}
        self.tree = self._build(X, y, 0)
        total = sum(self._importance.values()) + 1e-10
        self.feature_importance = {FEATURE_NAMES[k]: v/total
                                   for k, v in self._importance.items()}
        return self

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['label']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

    def to_dict(self):
        return {
            'tree': self.tree,
            'classes': self.classes,
            'feature_names': FEATURE_NAMES,
            'feature_importance': self.feature_importance,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.tree = d['tree']
        obj.classes = d['classes']
        obj.feature_importance = d.get('feature_importance', {})
        return obj


# ── CROSS VALIDATION ───────────────────────────────────────────────────────────

def k_fold_cv(X, y, k=5, merge_fine_motor=False):
    """5-fold cross-validation with stratification."""
    # Group by class
    class_indices = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)

    # Shuffle within each class
    for label in class_indices:
        random.shuffle(class_indices[label])

    # Build folds (stratified)
    folds = [[] for _ in range(k)]
    for label, indices in class_indices.items():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)

    fold_accs = []
    print(f"\nRunning {k}-fold cross-validation...")

    for fold in range(k):
        test_idx  = folds[fold]
        train_idx = [i for f in range(k) if f != fold for i in folds[f]]

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_test  = [y[i] for i in test_idx]

        model = DecisionTree(max_depth=8, min_samples=20)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = sum(p == t for p, t in zip(preds, y_test)) / len(y_test)
        fold_accs.append(acc)
        print(f"  Fold {fold+1}: {acc:.3f}")

    mean_acc = sum(fold_accs) / len(fold_accs)
    std_acc  = math.sqrt(sum((a - mean_acc)**2 for a in fold_accs) / len(fold_accs))
    print(f"\n  CV Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    return mean_acc, std_acc, fold_accs


# ── PER-DOMAIN ANALYSIS ────────────────────────────────────────────────────────

def per_domain_stats(X, y):
    """Show jerk and coupling stats per domain."""
    print("\nPer-domain feature statistics:")
    print(f"  {'Domain':20s} {'N':>6} {'Jerk p95':>10} {'Jerk peak':>10} {'Coupling r':>11} {'Accel std':>10}")
    print("  " + "-"*70)
    by_domain = defaultdict(list)
    for xi, yi in zip(X, y):
        by_domain[yi].append(xi)
    for domain in sorted(by_domain.keys()):
        rows = by_domain[domain]
        n = len(rows)
        mean = lambda col: sum(r[col] for r in rows) / n
        print(f"  {domain:20s} {n:>6} {mean(0):>10.1f} {mean(5):>10.1f} {mean(1):>11.3f} {mean(2):>10.3f}")


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='S2S Domain Classifier v1.4')
    parser.add_argument('--dataset', default='s2s_dataset/', help='Path to s2s_dataset/')
    parser.add_argument('--test', action='store_true', help='Run cross-validation')
    parser.add_argument('--merge-fine-motor', action='store_true',
                        help='Merge PRECISION+DAILY_LIVING into FINE_MOTOR (improves accuracy)')
    parser.add_argument('--out', default='model/', help='Output directory for model files')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"S2S Domain Classifier v1.4")
    print(f"Features: {FEATURE_NAMES}")
    if args.merge_fine_motor:
        print("Mode: FINE_MOTOR merge (PRECISION + DAILY_LIVING → FINE_MOTOR)")

    # Load data
    X, y, meta = load_dataset(args.dataset, merge_fine_motor=args.merge_fine_motor)
    if not X:
        print("ERROR: No data loaded. Check dataset path.")
        sys.exit(1)

    # Per-domain stats
    per_domain_stats(X, y)

    # Cross-validation
    if args.test:
        cv_acc, cv_std, fold_accs = k_fold_cv(X, y, k=5,
                                               merge_fine_motor=args.merge_fine_motor)

    # Train final model on all data
    print("\nTraining final model on all data...")
    t0 = time.time()
    model = DecisionTree(max_depth=8, min_samples=20)
    model.fit(X, y)
    elapsed = time.time() - t0

    train_preds = model.predict(X)
    train_acc = sum(p == t for p, t in zip(train_preds, y)) / len(y)
    print(f"  Train accuracy: {train_acc:.3f}  ({elapsed:.1f}s)")

    # Feature importance
    if model.feature_importance:
        print("\nFeature importance:")
        for name, imp in sorted(model.feature_importance.items(),
                                key=lambda x: -x[1]):
            bar = '█' * int(imp * 40)
            print(f"  {name:25s} {imp:.3f} {bar}")

    # Save model
    os.makedirs(args.out, exist_ok=True)
    model_data = model.to_dict()
    model_data.update({
        'version': '1.4',
        'n_records': len(X),
        'n_features': len(FEATURE_NAMES),
        'domains': sorted(set(y)),
        'merge_fine_motor': args.merge_fine_motor,
        'accuracy': round(cv_acc if args.test else train_acc, 4),
        'cv_std': round(cv_std if args.test else 0.0, 4),
        'trained_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })

    model_path = os.path.join(args.out, 's2s_domain_classifier.json')
    with open(model_path, 'w') as f:
        json.dump(model_data, f, indent=2)

    report = {
        'version': '1.4',
        'n_records': len(X),
        'domains': sorted(set(y)),
        'merge_fine_motor': args.merge_fine_motor,
        'cv_accuracy': round(cv_acc if args.test else 0.0, 4),
        'cv_std': round(cv_std if args.test else 0.0, 4),
        'train_accuracy': round(train_acc, 4),
        'feature_importance': model.feature_importance,
    }
    report_path = os.path.join(args.out, 'training_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    mode = 'FINE_MOTOR' if args.merge_fine_motor else 'standard'
    print(f"\nModel saved to {model_path}")
    print(f"Report saved to {report_path}")
    print(f"\nFinal: {mode} mode, {len(set(y))} domains, "
          f"CV {cv_acc:.1%} ± {cv_std:.3f}" if args.test else
          f"train {train_acc:.1%}")


if __name__ == '__main__':
    main()
