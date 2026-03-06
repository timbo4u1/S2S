#!/usr/bin/env python3
"""
S2S Level 3 — Kalman Reconstruction Experiment (UCI_HAR only)
Run from ~/S2S:
  python3 experiments/level3_uci_kalman.py --dataset s2s_dataset/ --epochs 40

Tests whether Kalman-smoothed BRONZE records, after passing S2S re-check,
improve model performance vs discarding them entirely.

Conditions:
  A = all UCI_HAR with raw data (clean baseline)
  B = UCI_HAR with 35% corruption injected
  C = S2S floor filtered (threshold=60)
  D = C + Kalman-reconstructed BRONZE added back (weight 0.5)

If D > C → Level 3 proven: reconstruction recovers data above the floor
"""

import os, sys, json, glob, math, random, time, argparse
from collections import defaultdict, Counter

# ── DOMAINS ──────────────────────────────────────────────────────
# UCI_HAR has 6 activities mapped to 2 domains
ACTIVITY_MAP = {
    'walking':            'LOCOMOTION',
    'walking_upstairs':   'LOCOMOTION',
    'walking_downstairs': 'LOCOMOTION',
    'sitting':            'DAILY_LIVING',
    'standing':           'DAILY_LIVING',
    'laying':             'DAILY_LIVING',
}
DOMAINS  = ['LOCOMOTION', 'DAILY_LIVING']
DOM2IDX  = {d: i for i, d in enumerate(DOMAINS)}

# Activity-level classification (6 classes)
ACTIVITIES = ['walk','climb_stairs','descend_stairs',
              'sit_down','stand_up','laying']
ACT2IDX = {a: i for i, a in enumerate(ACTIVITIES)}

CORRUPTION_RATE = 0.35
CLIP_THRESHOLD  = 0.5   # g units
FLAT_THRESHOLD  = 0.005


# ══════════════════════════════════════════════════════════════════
# PHYSICS SCORER (calibrated for UCI_HAR at 50Hz)
# ══════════════════════════════════════════════════════════════════

def score_record(rec):
    imu = rec.get('imu_raw', {})
    accel = imu.get('accel', [])
    if not accel or len(accel) < 20:
        return 0
    n = len(accel)

    ax = [accel[i][0] for i in range(n)]
    ay = [accel[i][1] for i in range(n)]
    az = [accel[i][2] for i in range(n)]
    mean_x = sum(ax)/n
    mean_y = sum(ay)/n
    mean_z = sum(az)/n
    var_x = sum((v-mean_x)**2 for v in ax)/n
    var_y = sum((v-mean_y)**2 for v in ay)/n
    var_z = sum((v-mean_z)**2 for v in az)/n
    total_var = var_x + var_y + var_z

    # HARD FAIL: flat signal
    if total_var < FLAT_THRESHOLD:
        return 0

    # HARD FAIL: clipping at exactly ±CLIP_THRESHOLD*9.81
    # Clip detection: clipping creates flat top — consecutive identical values
    # NOT just high values (clean UCI_HAR p95 = 0.505 m/s² — threshold overlap)
    for j in range(3):
        col = [accel[i][j] for i in range(n)]
        max_val = max(abs(v) for v in col)
        if max_val < 0.1:
            continue  # flat signal caught separately
        # Count samples stuck at maximum — physical clipping signature
        at_max = sum(1 for v in col if abs(abs(v) - max_val) < 0.001)
        if at_max / n > 0.25:  # >25% at exact maximum = clipped
            return 0

    # SOFT SCORES
    scores = []

    # Variance quality (UCI_HAR at 50Hz has higher variance than WISDM at 20Hz)
    if total_var > 20.0:   scores.append(90)
    elif total_var > 5.0:  scores.append(80)
    elif total_var > 1.0:  scores.append(70)
    else:                  scores.append(55)

    # Jerk check — valid at 50Hz (UCI_HAR sample rate)
    dt = 1.0 / 50.0
    jerks = []
    for axis in range(3):
        sig = [accel[i][axis] for i in range(n)]
        vel = [(sig[k+1]-sig[k-1])/(2*dt) for k in range(1, n-1)]
        jerk = [(vel[k+1]-vel[k-1])/(2*dt) for k in range(1, len(vel)-1)]
        jerks.extend([abs(j) for j in jerk])

    if jerks:
        jerks.sort()
        p95 = jerks[int(len(jerks)*0.95)]
        if p95 > 500:
            return 0   # HARD FAIL — supra-human jerk (valid at 50Hz)
        scores.append(90 if p95 < 150 else 75)

    # Domain plausibility
    action = rec.get('action','').lower()
    domain = ACTIVITY_MAP.get(action, '')
    if domain:
        scores.append(80)
    else:
        scores.append(50)

    return int(sum(scores)/len(scores)) if scores else 0


def get_tier(score):
    if score >= 87: return 'GOLD'
    if score >= 75: return 'SILVER'
    if score >= 60: return 'BRONZE'
    return 'REJECTED'


# ══════════════════════════════════════════════════════════════════
# KALMAN SMOOTHER
# ══════════════════════════════════════════════════════════════════

def kalman_smooth(accel_window, hz=50):
    """
    Rauch-Tung-Striebel Kalman smoother.
    Physics model: constant velocity (F = [[1,dt],[0,1]]).
    At 50Hz (UCI_HAR), this is the correct motion model.
    """
    n = len(accel_window)
    if n < 4:
        return accel_window
    dt = 1.0 / hz
    q = 0.005
    R = 0.3

    smoothed_axes = []
    for axis in range(3):
        signal = [accel_window[i][axis] for i in range(n)]
        x = [signal[0], 0.0]
        P = [[1.0, 0.0], [0.0, 1.0]]
        xs, Ps = [list(x)], [[list(P[0]), list(P[1])]]

        for k in range(1, n):
            xp = [x[0]+dt*x[1], x[1]]
            Pp = [[P[0][0]+dt*P[1][0]+dt*(P[0][1]+dt*P[1][1])+q*dt**3/3,
                   P[0][1]+dt*P[1][1]+q*dt**2/2],
                  [P[1][0]+dt*P[1][1]+q*dt**2/2, P[1][1]+q*dt]]
            S = Pp[0][0] + R
            K = [Pp[0][0]/S, Pp[1][0]/S]
            inn = signal[k] - xp[0]
            x = [xp[0]+K[0]*inn, xp[1]+K[1]*inn]
            P = [[(1-K[0])*Pp[0][0], (1-K[0])*Pp[0][1]],
                 [Pp[1][0]-K[1]*Pp[0][0], Pp[1][1]-K[1]*Pp[0][1]]]
            xs.append(list(x))
            Ps.append([list(P[0]), list(P[1])])

        # RTS backward pass
        result = [xs[-1][0]]
        for k in range(n-2, -1, -1):
            xf, Pf = xs[k], Ps[k]
            xp = [xf[0]+dt*xf[1], xf[1]]
            Pp = [[Pf[0][0]+dt*Pf[1][0]+dt*(Pf[0][1]+dt*Pf[1][1])+q*dt**3/3,
                   Pf[0][1]+dt*Pf[1][1]+q*dt**2/2],
                  [Pf[1][0]+dt*Pf[1][1]+q*dt**2/2, Pf[1][1]+q*dt]]
            det = Pp[0][0]*Pp[1][1] - Pp[0][1]*Pp[1][0]
            if abs(det) < 1e-10:
                result.insert(0, xf[0])
                continue
            G00 = (Pf[0][0]*Pp[1][1]-Pf[0][1]*Pp[1][0]) / det
            result.insert(0, xf[0] + G00*(result[0]-xp[0]))

        smoothed_axes.append(result)

    return [[smoothed_axes[0][i], smoothed_axes[1][i], smoothed_axes[2][i]]
            for i in range(n)]


# ══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_features(rec):
    imu = rec.get('imu_raw', {})
    accel = imu.get('accel', [])
    if not accel or len(accel) < 20:
        return None
    n = len(accel)

    feats = []
    for axis in range(3):
        vals = [accel[i][axis] for i in range(n)]
        mean = sum(vals)/n
        std = math.sqrt(sum((v-mean)**2 for v in vals)/n)
        rng = max(vals)-min(vals)
        feats += [mean, std, rng]

    mags = [math.sqrt(sum(accel[i][j]**2 for j in range(3))) for i in range(n)]
    mag_mean = sum(mags)/n
    mag_std = math.sqrt(sum((v-mag_mean)**2 for v in mags)/n)
    mag_max = max(mags)
    feats += [mag_mean, mag_std, mag_max]

    # Dominant frequency (FFT approximation)
    N = n
    fft_mag = []
    for k in range(1, min(N//2, 10)):
        re = sum(mags[i]*math.cos(2*math.pi*k*i/N) for i in range(N))
        im = sum(mags[i]*math.sin(2*math.pi*k*i/N) for i in range(N))
        fft_mag.append(math.sqrt(re**2+im**2))
    dom_freq = fft_mag.index(max(fft_mag))+1 if fft_mag else 0
    feats.append(dom_freq)

    # Zero crossing rate
    zcr = sum(1 for i in range(1,n) if mags[i]*mags[i-1] < 0) / n
    feats.append(zcr)

    return feats


# ══════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════

class MLP:
    def __init__(self, input_dim, hidden, output, lr=0.01):
        self.lr = lr
        random.seed(42)
        scale1 = math.sqrt(2.0/input_dim)
        scale2 = math.sqrt(2.0/hidden)
        self.W1 = [[random.gauss(0,scale1) for _ in range(input_dim)] for _ in range(hidden)]
        self.b1 = [0.0]*hidden
        self.W2 = [[random.gauss(0,scale2) for _ in range(hidden)] for _ in range(output)]
        self.b2 = [0.0]*output

    def forward(self, x):
        h = [max(0, sum(self.W1[j][i]*x[i] for i in range(len(x)))+self.b1[j])
             for j in range(len(self.W1))]
        logits = [sum(self.W2[k][j]*h[j] for j in range(len(h)))+self.b2[k]
                  for k in range(len(self.W2))]
        mx = max(logits)
        exp_l = [math.exp(min(l-mx, 60)) for l in logits]
        s = sum(exp_l)
        probs = [e/s for e in exp_l]
        return h, probs

    def backward(self, x, h, probs, label):
        d2 = list(probs)
        d2[label] -= 1.0
        for k in range(len(self.W2)):
            for j in range(len(h)):
                self.W2[k][j] -= self.lr * d2[k] * h[j]
            self.b2[k] -= self.lr * d2[k]
        d1 = [sum(self.W2[k][j]*d2[k] for k in range(len(self.W2))) * (1 if h[j]>0 else 0)
              for j in range(len(h))]
        for j in range(len(self.W1)):
            for i in range(len(x)):
                self.W1[j][i] -= self.lr * d1[j] * x[i]
            self.b1[j] -= self.lr * d1[j]

    def _clip_weights(self, max_norm=5.0):
        for row in self.W1+self.W2:
            n = math.sqrt(sum(v**2 for v in row))
            if n > max_norm:
                for i in range(len(row)):
                    row[i] *= max_norm/n

    def train_epoch(self, samples, weight=1.0):
        random.shuffle(samples)
        loss, skips = 0.0, 0
        for item in samples:
            if len(item) == 3:
                feat, label, w = item
            else:
                feat, label = item
                w = weight
            h, probs = self.forward(feat)
            step_loss = -math.log(max(probs[label], 1e-10))
            if step_loss != step_loss:
                skips += 1
                continue
            loss += step_loss * w
            orig_lr = self.lr
            self.lr *= w
            self.backward(feat, h, probs, label)
            self.lr = orig_lr
        self._clip_weights()
        return loss / max(len(samples)-skips, 1)

    def evaluate(self, samples):
        correct = 0
        per_class = defaultdict(lambda: [0,0])
        for feat, label in samples:
            _, probs = self.forward(feat)
            pred = probs.index(max(probs))
            per_class[label][1] += 1
            if pred == label:
                correct += 1
                per_class[label][0] += 1
        acc = correct / max(len(samples), 1)
        n_cls = len(ACTIVITIES)
        f1s = [per_class[i][0]/max(per_class[i][1],1) for i in range(n_cls)]
        return acc, sum(f1s)/len(f1s)


# ══════════════════════════════════════════════════════════════════
# DATA LOADING + CORRUPTION
# ══════════════════════════════════════════════════════════════════

def load_uci(dataset_dir):
    print("Loading UCI_HAR records with raw data...")
    records = []
    for fpath in glob.glob(os.path.join(dataset_dir,'**','*.json'), recursive=True):
        try:
            with open(fpath) as f:
                rec = json.load(f)
        except Exception:
            continue
        if rec.get('dataset_source','') != 'UCI_HAR':
            continue
        if 'imu_raw' not in rec:
            continue
        action = rec.get('action','').lower()
        if action not in ACT2IDX:
            continue
        records.append(rec)
    print(f"  Loaded {len(records)} UCI_HAR records with raw data")
    return records


def subject_split(records, test_frac=0.2):
    subjects = sorted(set(r.get('person_id','?') for r in records))
    random.seed(42)
    random.shuffle(subjects)
    n_test = max(1, int(len(subjects)*test_frac))
    test_ids = set(subjects[:n_test])
    train = [r for r in records if r.get('person_id') not in test_ids]
    test  = [r for r in records if r.get('person_id') in test_ids]
    print(f"  Train subjects: {len(subjects)-n_test}  Test subjects: {n_test}")
    print(f"  Train: {len(train)}  Test: {len(test)}")
    return train, test


def inject_corruption(records, rate=CORRUPTION_RATE):
    random.seed(123)
    corrupted = []
    n_corrupt = int(len(records)*rate)
    indices = random.sample(range(len(records)), n_corrupt)
    corrupt_set = set(indices)
    counts = {'flat':0,'clip':0,'label':0}

    for i, rec in enumerate(records):
        r = json.loads(json.dumps(rec))  # deep copy
        if i in corrupt_set:
            ctype = i % 3
            accel = r['imu_raw']['accel']
            if ctype == 0:  # flat signal
                r['imu_raw']['accel'] = [[random.gauss(0,0.001)]*3
                                          for _ in range(len(accel))]
                counts['flat'] += 1
            elif ctype == 1:  # sensor clipping — UCI_HAR body accel max ~3 m/s²
                clip = 0.5  # m/s² — actually clips UCI_HAR signals
                r['imu_raw']['accel'] = [[max(-clip,min(clip,v))
                                           for v in row] for row in accel]
                counts['clip'] += 1
            else:  # label noise
                acts = [a for a in ACTIVITIES if a != r.get('action','')]
                if acts:
                    r['action'] = random.choice(acts)
                counts['label'] += 1
        corrupted.append(r)

    print(f"  Injected {sum(counts.values())} corruptions ({rate*100:.0f}%):")
    for k,v in counts.items():
        print(f"    {k:<15} {v}")
    return corrupted


# ══════════════════════════════════════════════════════════════════
# NORMALIZATION
# ══════════════════════════════════════════════════════════════════

def normalize(X, stats=None):
    if not X: return [], stats
    n_feat = len(X[0])
    if stats is None:
        means = [sum(x[i] for x in X)/len(X) for i in range(n_feat)]
        stds  = [math.sqrt(sum((x[i]-means[i])**2 for x in X)/len(X))+1e-8
                 for i in range(n_feat)]
        stats = (means, stds)
    means, stds = stats
    return [[( x[i]-means[i])/stds[i] for i in range(n_feat)] for x in X], stats


# ══════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run(dataset_dir, output_path, epochs=40):
    print()
    print("S2S Level 3 — Kalman Reconstruction Experiment")
    print("UCI_HAR dataset  |  50Hz  |  6 activities  |  Kalman RTS smoother")
    print("=" * 60)

    # Load
    all_records = load_uci(dataset_dir)
    if not all_records:
        print("ERROR: No UCI_HAR records with raw data found.")
        print("Run inject_uci_raw.py first.")
        sys.exit(1)

    # Split by subject
    print()
    train_clean, test_records = subject_split(all_records)

    # Extract test features
    test_samples = []
    for rec in test_records:
        feats = extract_features(rec)
        action = rec.get('action','').lower()
        if feats and action in ACT2IDX:
            test_samples.append((feats, ACT2IDX[action]))
    print(f"  Test samples: {len(test_samples)}")

    # Compute normalization stats from clean train features
    xA_raw = [extract_features(r) for r in train_clean]
    yA = [ACT2IDX[r.get('action','').lower()] for r in train_clean]
    valid = [(f,l) for f,l in zip(xA_raw,yA) if f is not None]
    xA_raw = [f for f,_ in valid]
    yA = [l for _,l in valid]
    xA_n, norm_stats = normalize(xA_raw)

    # Verify normalization worked
    if xA_n:
        means_check = [sum(x[i] for x in xA_n)/len(xA_n) for i in range(len(xA_n[0]))]
        stds_check  = [sum((x[i]-means_check[i])**2 for x in xA_n)/len(xA_n)**0.5
                       for i in range(len(xA_n[0]))]
        print(f"  Normalization check — feature mean range: "
              f"{min(means_check):.3f} to {max(means_check):.3f} (should be ~0)")

    xt_raw = [f for f,_ in test_samples]
    xt_n, _ = normalize(xt_raw, norm_stats)
    test_eval = list(zip(xt_n, [l for _,l in test_samples]))

    # Corrupt train
    print()
    print(f"Injecting corruption ({CORRUPTION_RATE*100:.0f}% of train)...")
    train_corrupted = inject_corruption(train_clean)

    # S2S filter
    print()
    print("Applying S2S quality floor (adaptive threshold)...")
    # Calibrate floor to THIS dataset — bottom 25% of clean scores
    # Physics laws are universal; thresholds are dataset-relative
    # CRITICAL: score LIVE after corruption injection
    # Stored physics_score is pre-corruption — useless for filtering
    print('  Live-scoring clean records for floor calibration...')
    scores_clean = sorted([score_record(r) for r in train_clean])
    print('  Live-scoring corrupted records...')
    scores_corr  = [score_record(r) for r in train_corrupted[:500]]
    # Floor = p25 of LIVE clean scores
    p25_idx = int(len(scores_clean) * 0.25)
    floor = scores_clean[p25_idx]
    print(f"  Adaptive floor = p25 of live clean scores = {floor}")
    print(f"  Score avg — clean: {sum(scores_clean)/len(scores_clean):.1f}"
          f"  corrupted: {sum(scores_corr)/len(scores_corr):.1f}")
    train_filtered = [r for r in train_corrupted
                      if score_record(r) >= floor]
    removed = len(train_corrupted) - len(train_filtered)
    print(f"  Kept {len(train_filtered)} / {len(train_corrupted)}  (removed {removed})")

    # Kalman reconstruction of BRONZE records
    print()
    print("Kalman reconstruction of BRONZE records...")
    reconstructed = []
    bronze_total = 0
    bronze_passed = 0
    bronze_failed = 0

    for rec in train_filtered:
        s = score_record(rec)
        if 60 <= s < 75:  # BRONZE
            bronze_total += 1
            accel = rec['imu_raw']['accel']
            smoothed = kalman_smooth(accel, hz=50)
            test_rec = json.loads(json.dumps(rec))
            test_rec['imu_raw']['accel'] = smoothed
            new_score = score_record(test_rec)
            if new_score >= 75:  # passed S2S re-check
                bronze_passed += 1
                reconstructed.append(test_rec)
            else:
                bronze_failed += 1

    print(f"  BRONZE records found:      {bronze_total}")
    print(f"  Passed S2S re-check (>=75): {bronze_passed}")
    print(f"  Failed re-check:            {bronze_failed}")
    print(f"  Reconstruction acceptance:  "
          f"{bronze_passed/max(bronze_total,1)*100:.1f}%")

    # Build condition datasets
    def make_samples(records, weight=1.0):
        out = []
        for rec in records:
            feats = extract_features(rec)
            action = rec.get('action','').lower()
            if feats and action in ACT2IDX:
                out.append((feats, ACT2IDX[action], weight))
        return out

    sA = make_samples(train_clean)
    sB = make_samples(train_corrupted)
    sC = make_samples(train_filtered)
    sD = sC + make_samples(reconstructed, weight=0.5)  # Kalman reconstructed

    # Normalize all conditions using SAME stats from clean train
    # Critical: all conditions use identical normalization
    def norm_s(samples):
        feats = [f for f,_,_ in samples]
        labels = [l for _,l,_ in samples]
        weights = [w for _,_,w in samples]
        fn, _ = normalize(feats, norm_stats)
        return list(zip(fn, labels, weights))

    sA_n = norm_s(sA)
    sB_n = norm_s(sB)
    sC_n = norm_s(sC)
    sD_n = norm_s(sD)

    # Sanity check — loss should start near log(n_classes) = log(6) ≈ 1.79
    print(f"  Training set sizes: A={len(sA_n)} B={len(sB_n)} "
          f"C={len(sC_n)} D={len(sD_n)}")

    print()
    print(f"Condition sizes:")
    print(f"  A (clean):         {len(sA_n)}")
    print(f"  B (corrupted):     {len(sB_n)}")
    print(f"  C (S2S filtered):  {len(sC_n)}")
    print(f"  D (C + Kalman):    {len(sD_n)}  (+{len(sD_n)-len(sC_n)} reconstructed)")

    # Train and evaluate
    results = {}
    conditions = [
        ("A_clean",       sA_n,  "Clean UCI_HAR baseline"),
        ("B_corrupted",   sB_n,  f"Corrupted {CORRUPTION_RATE*100:.0f}%, no filter"),
        ("C_s2s_floor",   sC_n,  "S2S floor filtered (score>=60)"),
        ("D_kalman_recon",sD_n,  "S2S floor + Kalman reconstructed BRONZE (w=0.5)"),
    ]

    for cond, train_data, desc in conditions:
        print()
        print(f"{'─'*60}")
        print(f"Condition {cond}  n={len(train_data)}")
        print(f"  {desc}")
        model = MLP(input_dim=14, hidden=64, output=len(ACTIVITIES), lr=0.001)
        t0 = time.time()
        for ep in range(epochs):
            loss = model.train_epoch(train_data)
            if (ep+1) % 10 == 0:
                acc, f1 = model.evaluate(test_eval)
                print(f"  Epoch {ep+1:3d}/{epochs}  loss={loss:.4f}"
                      f"  acc={acc:.3f}  f1={f1:.3f}")
        acc, f1 = model.evaluate(test_eval)
        elapsed = time.time()-t0
        print(f"  Final: {acc:.4f} ({acc*100:.1f}%)  F1={f1:.4f}  [{elapsed:.0f}s]")
        results[cond] = {
            "description": desc,
            "accuracy":    round(acc,4),
            "macro_f1":    round(f1,4),
            "train_n":     len(train_data),
            "test_n":      len(test_eval),
        }

    # Summary
    print()
    print("=" * 60)
    print("  LEVEL 3 RESULTS — KALMAN RECONSTRUCTION")
    print("=" * 60)
    print()
    print(f"  {'Condition':<20} {'Description':<35} {'acc':>6}  {'F1':>6}  {'n':>6}")
    print("  " + "─"*72)
    for cond, r in results.items():
        print(f"  {cond:<20} {r['description']:<35} "
              f"{r['accuracy']:.4f}  {r['macro_f1']:.4f}  {r['train_n']:>6}")

    rA = results.get('A_clean',{})
    rB = results.get('B_corrupted',{})
    rC = results.get('C_s2s_floor',{})
    rD = results.get('D_kalman_recon',{})

    print()
    if rA and rB and rC and rD:
        damage = rB['macro_f1'] - rA['macro_f1']
        recovery = rC['macro_f1'] - rB['macro_f1']
        kalman_gain = rD['macro_f1'] - rC['macro_f1']
        net = rD['macro_f1'] - rA['macro_f1']

        print(f"  Corruption damage  (B-A): {damage*100:+.2f}% F1")
        print(f"  S2S floor recovery (C-B): {recovery*100:+.2f}% F1")
        print(f"  Kalman gain        (D-C): {kalman_gain*100:+.2f}% F1  "
              f"← Level 3 claim")
        print(f"  Net vs baseline    (D-A): {net*100:+.2f}% F1")
        print(f"  Reconstructed records:    {bronze_passed}")
        print()

        if kalman_gain > 0.001:
            verdict = "✓ LEVEL 3 PROVEN — Kalman reconstruction improves over floor"
        elif kalman_gain > -0.001:
            verdict = "~ Neutral — reconstruction neither helps nor hurts"
        else:
            verdict = "✗ Reconstruction did not improve over floor filter alone"
        print(f"  Verdict: {verdict}")

    import json as json_mod
    results['meta'] = {
        "experiment": "s2s_level3_kalman_uci",
        "dataset": "UCI_HAR",
        "hz": 50,
        "bronze_total": bronze_total,
        "bronze_reconstructed": bronze_passed,
        "acceptance_rate": round(bronze_passed/max(bronze_total,1),3),
        "corruption_rate": CORRUPTION_RATE,
    }
    with open(output_path, 'w') as f:
        json_mod.dump(results, f, indent=2)
    print(f"\n  Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='s2s_dataset/')
    parser.add_argument('--epochs',  type=int, default=40)
    parser.add_argument('--out',     default='experiments/results_level3.json')
    args = parser.parse_args()
    run(args.dataset, args.out, args.epochs)
