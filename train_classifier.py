#!/usr/bin/env python3
"""
train_classifier.py — S2S v1.3 | Domain Classifier
First Physical AI model trained on physics-certified motion data.

WHAT THIS DOES:
  Reads certified JSON records from your dataset
  Trains a classifier: physics features → motion domain
  No GPU needed. Runs on MacBook in seconds.
  Uses only Python stdlib — zero extra dependencies.

USAGE:
  python3 train_classifier.py --dataset s2s_dataset/ --test
  python3 train_classifier.py --dataset s2s_dataset/ --train --out model/
  python3 train_classifier.py --model model/ --jerk 31 --coupling 0.35 --score 69

FEATURES (input):
  jerk_p95_ms3     — 95th percentile jerk (m/s³)
  imu_coupling_r   — rigid body coupling correlation
  physics_score    — S2S certification score (0-100)
  duration_s       — recording duration

LABELS (output):
  LOCOMOTION       walk, run, stairs, jump
  DAILY_LIVING     iron, vacuum, cooking, computer
  PRECISION        point, write, surgery
  POWER            grasp, lift, push, throw
  SOCIAL           wave, clap, gesture
  SPORT            tennis, football, jump_rope
"""

import json, glob, os, math, argparse, random, csv
from collections import defaultdict

# ── Feature extraction ─────────────────────────────────────────────
def extract_features(record):
    """Extract 4 physics features from a certified record."""
    jerk     = record.get('jerk_p95_ms3') or 100.0
    coupling = record.get('imu_coupling_r') or 0.0
    score    = record.get('physics_score') or 50.0
    duration = record.get('duration_s') or 2.56

    # Log-scale jerk (range is huge: 10-5000 m/s³)
    log_jerk = math.log10(max(jerk, 1.0))

    # Normalize features to roughly 0-1
    return [
        log_jerk / 4.0,         # log10(5000)≈3.7, so /4 gives 0-1
        (coupling + 1) / 2.0,   # coupling is -1 to 1, shift to 0-1
        score / 100.0,           # already 0-100
        min(duration, 30) / 30.0 # cap at 30s
    ]

FEATURE_NAMES = ['log_jerk/4', 'coupling_norm', 'score/100', 'duration/30']

# ── Load dataset ───────────────────────────────────────────────────
def load_dataset(dataset_dir):
    """Load all certified JSON records and extract features + labels."""
    X, y, meta = [], [], []

    patterns = [
        f"{dataset_dir}/**/*.json",
        f"{dataset_dir}/*.json"
    ]

    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))

    # Remove duplicates
    paths = list(set(paths))

    skipped = 0
    for path in paths:
        try:
            with open(path) as f:
                rec = json.load(f)

            # Skip .DS_Store or non-records
            domain = rec.get('domain')
            action = rec.get('action')
            if not domain or not action:
                skipped += 1
                continue

            # Skip if jerk is wildly unrealistic (artifact of long recordings)
            jerk = rec.get('jerk_p95_ms3') or 0
            if jerk > 2000:
                skipped += 1
                continue

            feat = extract_features(rec)
            X.append(feat)
            y.append(domain)
            meta.append({
                'action': action,
                'domain': domain,
                'jerk':   jerk,
                'coupling': rec.get('imu_coupling_r'),
                'score':    rec.get('physics_score'),
                'tier':     rec.get('physics_tier'),
                'source':   rec.get('dataset_source', 'unknown'),
                'path':     path,
            })
        except Exception as e:
            skipped += 1

    print(f"  Loaded: {len(X)} records  Skipped: {skipped}")
    return X, y, meta


# ── Gaussian Naive Bayes classifier ───────────────────────────────
# Pure Python, zero dependencies, works great for 4 features
class GaussianNB:
    """
    Gaussian Naive Bayes — simple, interpretable, physics-appropriate.
    For each domain, models each feature as a Gaussian distribution.
    Classifies by finding which domain's distribution best explains input.
    """
    def __init__(self):
        self.classes  = []
        self.priors   = {}
        self.means    = {}
        self.stds     = {}

    def fit(self, X, y):
        from collections import defaultdict
        groups = defaultdict(list)
        for xi, yi in zip(X, y):
            groups[yi].append(xi)

        n_total = len(X)
        self.classes = sorted(groups.keys())

        for cls, samples in groups.items():
            self.priors[cls] = len(samples) / n_total
            n_feat = len(samples[0])
            self.means[cls] = [
                sum(s[i] for s in samples) / len(samples)
                for i in range(n_feat)
            ]
            self.stds[cls] = []
            for i in range(n_feat):
                vals = [s[i] for s in samples]
                mean = self.means[cls][i]
                var  = sum((v-mean)**2 for v in vals) / max(len(vals)-1, 1)
                self.stds[cls].append(max(math.sqrt(var), 1e-6))

        return self

    def _log_likelihood(self, cls, x):
        ll = math.log(self.priors[cls])
        for i, (xi, mean, std) in enumerate(
                zip(x, self.means[cls], self.stds[cls])):
            # log of Gaussian PDF
            ll -= math.log(std * math.sqrt(2*math.pi))
            ll -= 0.5 * ((xi - mean) / std) ** 2
        return ll

    def predict_one(self, x):
        scores = {cls: self._log_likelihood(cls, x) for cls in self.classes}
        return max(scores, key=scores.get), scores

    def predict(self, X):
        return [self.predict_one(x)[0] for x in X]

    def score(self, X, y):
        preds = self.predict(X)
        return sum(p==t for p,t in zip(preds,y)) / len(y)

    def save(self, path):
        model_data = {
            'type': 'GaussianNB_S2S_v1',
            'classes': self.classes,
            'priors':  self.priors,
            'means':   self.means,
            'stds':    self.stds,
            'features': FEATURE_NAMES,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(model_data, f, indent=2)
        print(f"  Model saved: {path}")

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        m = cls()
        m.classes = d['classes']
        m.priors  = d['priors']
        m.means   = d['means']
        m.stds    = d['stds']
        return m


# ── Cross-validation ───────────────────────────────────────────────
def cross_validate(X, y, k=5, seed=42):
    """k-fold cross-validation."""
    rng = random.Random(seed)
    indices = list(range(len(X)))
    rng.shuffle(indices)
    fold_size = len(indices) // k
    scores = []

    for fold in range(k):
        test_idx  = indices[fold*fold_size:(fold+1)*fold_size]
        train_idx = [i for i in indices if i not in set(test_idx)]
        if not test_idx or not train_idx: continue

        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test  = [X[i] for i in test_idx]
        y_test  = [y[i] for i in test_idx]

        model = GaussianNB().fit(X_train, y_train)
        acc   = model.score(X_test, y_test)
        scores.append(acc)

    return scores


# ── Dataset analysis ───────────────────────────────────────────────
def analyze_dataset(X, y, meta):
    print(f"\n{'='*55}")
    print(f"  DATASET ANALYSIS")
    print(f"{'='*55}")
    print(f"  Total records: {len(X)}")

    # By domain
    domain_counts = defaultdict(int)
    domain_jerk   = defaultdict(list)
    domain_coup   = defaultdict(list)
    domain_score  = defaultdict(list)
    action_counts = defaultdict(int)

    for m in meta:
        domain_counts[m['domain']] += 1
        action_counts[m['action']] += 1
        if m['jerk']:   domain_jerk[m['domain']].append(m['jerk'])
        if m['coupling']: domain_coup[m['domain']].append(m['coupling'])
        if m['score']:  domain_score[m['domain']].append(m['score'])

    print(f"\n  By Domain:")
    print(f"  {'Domain':<15} {'N':>5}  {'Jerk P95':>10}  {'Coupling r':>11}  {'Phys Score':>11}")
    print(f"  {'─'*15} {'─'*5}  {'─'*10}  {'─'*11}  {'─'*11}")
    for dom in sorted(domain_counts.keys()):
        n   = domain_counts[dom]
        j   = sum(domain_jerk[dom])/len(domain_jerk[dom]) if domain_jerk[dom] else 0
        r   = sum(domain_coup[dom])/len(domain_coup[dom]) if domain_coup[dom] else 0
        s   = sum(domain_score[dom])/len(domain_score[dom]) if domain_score[dom] else 0
        print(f"  {dom:<15} {n:>5}  {j:>10.1f}  {r:>11.4f}  {s:>11.1f}")

    print(f"\n  Top Actions:")
    for act, cnt in sorted(action_counts.items(), key=lambda x:-x[1])[:10]:
        dom = next((m['domain'] for m in meta if m['action']==act), '?')
        print(f"  {act:<25} {cnt:>5}  [{dom}]")

    # Source breakdown
    sources = defaultdict(int)
    for m in meta: sources[m['source']] += 1
    print(f"\n  By Source: {dict(sources)}")


# ── Main ───────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='S2S Domain Classifier')
    p.add_argument('--dataset', help='Path to s2s_dataset/ directory')
    p.add_argument('--train',   action='store_true', help='Train the model')
    p.add_argument('--test',    action='store_true', help='Analyze + cross-validate')
    p.add_argument('--out',     default='model/', help='Output directory for model')
    p.add_argument('--model',   help='Path to saved model for inference')
    p.add_argument('--jerk',    type=float, help='Jerk P95 (m/s³) for single prediction')
    p.add_argument('--coupling',type=float, help='IMU coupling r for single prediction')
    p.add_argument('--score',   type=float, default=69, help='Physics score')
    p.add_argument('--duration',type=float, default=2.56)
    args = p.parse_args()

    # ── Single prediction mode ─────────────────────────────────────
    if args.model and args.jerk is not None:
        model = GaussianNB.load(args.model)
        feat  = extract_features({
            'jerk_p95_ms3': args.jerk,
            'imu_coupling_r': args.coupling or 0.3,
            'physics_score': args.score,
            'duration_s': args.duration,
        })
        domain, scores = model.predict_one(feat)
        print(f"\nPrediction:")
        print(f"  jerk={args.jerk} m/s³  coupling_r={args.coupling}  score={args.score}")
        print(f"  → Domain: {domain}")
        print(f"\n  All domain scores (higher = more likely):")
        for d, s in sorted(scores.items(), key=lambda x: -x[1]):
            bar = '█' * max(0, int((s + 30) * 2))
            print(f"  {d:<15} {s:>8.2f}  {bar}")
        return

    if not args.dataset:
        p.print_help(); return

    print(f"\nS2S Domain Classifier")
    print(f"Loading dataset from: {args.dataset}")
    X, y, meta = load_dataset(args.dataset)

    if len(X) < 5:
        print(f"\n  Only {len(X)} records found.")
        print(f"  Need at least 5 records to train.")
        print(f"  Run s2s_dataset_adapter.py first to generate certified records.")
        return

    analyze_dataset(X, y, meta)

    if args.test or args.train:
        # Check class distribution
        class_counts = defaultdict(int)
        for yi in y: class_counts[yi] += 1
        valid_classes = [c for c,n in class_counts.items() if n >= 3]
        X_filt = [x for x,yi in zip(X,y) if yi in valid_classes]
        y_filt = [yi for yi in y if yi in valid_classes]

        if len(set(y_filt)) < 2:
            print(f"\n  Need at least 2 domains with 3+ records each.")
            print(f"  Current: {dict(class_counts)}")
            print(f"  Download more datasets with s2s_dataset_adapter.py")
            return

        print(f"\n  Training on {len(X_filt)} records, {len(set(y_filt))} domains")
        print(f"  Domains: {sorted(set(y_filt))}")

        # Cross-validation
        if len(X_filt) >= 10:
            cv_scores = cross_validate(X_filt, y_filt, k=min(5, len(X_filt)//3))
            if cv_scores:
                mean_acc = sum(cv_scores)/len(cv_scores)
                print(f"\n  {len(cv_scores)}-fold Cross-validation:")
                print(f"  Accuracy: {mean_acc*100:.1f}%  "
                      f"(folds: {[f'{s*100:.0f}%' for s in cv_scores]})")

        # Train final model on all data
        model = GaussianNB().fit(X_filt, y_filt)
        train_acc = model.score(X_filt, y_filt)
        print(f"  Train accuracy: {train_acc*100:.1f}%")

        # Show what physics features matter per domain
        print(f"\n  Domain Physics Profiles (what the model learned):")
        print(f"  {'Domain':<15} {'Jerk center':>12}  {'Coupling center':>16}  {'Score center':>13}")
        for cls in model.classes:
            # Denormalize means back to real units
            log_jerk_norm = model.means[cls][0]
            coup_norm     = model.means[cls][1]
            score_norm    = model.means[cls][2]
            jerk_real  = 10 ** (log_jerk_norm * 4)
            coup_real  = coup_norm * 2 - 1
            score_real = score_norm * 100
            print(f"  {cls:<15} {jerk_real:>10.1f} m/s³  "
                  f"{coup_real:>14.4f} r  {score_real:>11.1f}/100")

        if args.train:
            os.makedirs(args.out, exist_ok=True)
            model_path = os.path.join(args.out, 's2s_domain_classifier.json')
            model.save(model_path)

            # Save training report
            report = {
                "model_type": "GaussianNB",
                "n_features": 4,
                "feature_names": FEATURE_NAMES,
                "n_classes": len(model.classes),
                "classes": model.classes,
                "n_train": len(X_filt),
                "train_accuracy": round(train_acc, 4),
                "domain_profiles": {
                    cls: {
                        "jerk_center_ms3": round(10**(model.means[cls][0]*4), 1),
                        "coupling_r_center": round(model.means[cls][1]*2-1, 4),
                        "score_center": round(model.means[cls][2]*100, 1),
                    } for cls in model.classes
                },
                "science_basis": {
                    "jerk": "Flash-Hogan 1985: minimum jerk theorem",
                    "coupling": "Bernstein 1967: motor synergies",
                    "score": "S2S biomechanical physics certification"
                }
            }
            report_path = os.path.join(args.out, 'training_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"  Training report: {report_path}")

            print(f"\n  Test a prediction:")
            print(f"  python3 train_classifier.py --model {model_path} "
                  f"--jerk 31 --coupling 0.35 --score 69")

    print()

if __name__ == '__main__':
    main()
