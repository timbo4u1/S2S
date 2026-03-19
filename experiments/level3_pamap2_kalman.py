#!/usr/bin/env python3
"""
S2S Level 3 — Kalman Reconstruction on PAMAP2
Real 100Hz IMU data, 9 subjects, natural sensor noise.

PAMAP2 activity labels:
  1=lying, 2=sitting, 3=standing, 4=walking, 5=running,
  6=cycling, 7=nordic_walking, 12=ascending_stairs,
  13=descending_stairs, 16=vacuum_cleaning,
  17=ironing, 24=rope_jumping

We use chest IMU accelerometer (cols 21-23) — most stable placement.
Window: 256 samples = 2.56 seconds at 100Hz.

Run from ~/S2S:
  python3 experiments/level3_pamap2_kalman.py \
    --data data/pamap2/ --epochs 40 \
    --out experiments/results_level3_pamap2.json
"""

import os, sys, json, math, random, time, argparse
from collections import defaultdict

# ── Activity mapping ──────────────────────────────────────────────
# Map PAMAP2 labels to 3 domains matching S2S tier system
ACTIVITY_LABELS = {
    1:  'REST',        # lying
    2:  'REST',        # sitting
    3:  'REST',        # standing
    4:  'LOCOMOTION',  # walking
    5:  'LOCOMOTION',  # running
    6:  'LOCOMOTION',  # cycling
    7:  'LOCOMOTION',  # nordic walking
    12: 'LOCOMOTION',  # ascending stairs
    13: 'LOCOMOTION',  # descending stairs
    16: 'ACTIVITY',    # vacuum cleaning
    17: 'ACTIVITY',    # ironing
    24: 'ACTIVITY',    # rope jumping
}

DOMAINS  = ['REST', 'LOCOMOTION', 'ACTIVITY']
DOM2IDX  = {d: i for i, d in enumerate(DOMAINS)}

# PAMAP2 column indices (chest IMU)
COL_TIMESTAMP = 0
COL_ACTIVITY  = 1
COL_CHEST_AX  = 21  # chest accel x (m/s²)
COL_CHEST_AY  = 22
COL_CHEST_AZ  = 23

WINDOW_SIZE = 256   # samples at 100Hz = 2.56s
STEP_SIZE   = 128   # 50% overlap
HZ          = 100

CORRUPTION_RATE = 0.35


# ══════════════════════════════════════════════════════════════════
# PHYSICS SCORER — calibrated for 100Hz chest IMU
# ══════════════════════════════════════════════════════════════════

def score_window(accel):
    """
    Physics scorer for PAMAP2 chest IMU at 100Hz.
    Returns 0-100.
    """
    n = len(accel)
    if n < 20:
        return 0

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

    # HARD FAIL: flat signal (dead sensor)
    if total_var < 0.01:
        return 0

    # HARD FAIL: clipping — detect flat top signature
    for axis_vals in [ax, ay, az]:
        max_val = max(abs(v) for v in axis_vals)
        if max_val < 0.5:
            continue
        at_max = sum(1 for v in axis_vals if abs(abs(v)-max_val) < 0.01)
        if at_max / n > 0.20:
            return 0

    # HARD FAIL: jerk bound — 100Hz makes this fully valid
    dt = 1.0 / HZ
    jerks = []
    for axis_vals in [ax, ay, az]:
        vel = [(axis_vals[k+1]-axis_vals[k-1])/(2*dt) for k in range(1, n-1)]
        jerk = [(vel[k+1]-vel[k-1])/(2*dt) for k in range(1, len(vel)-1)]
        jerks.extend([abs(j) for j in jerk])
    if jerks:
        jerks.sort()
        p95 = jerks[int(len(jerks)*0.95)]
        if p95 > 5000:  # supra-human jerk at 100Hz (clean p99=1546)
            return 0

    scores = []

    # Variance quality (chest IMU at 100Hz)
    if total_var > 10.0:   scores.append(90)
    elif total_var > 2.0:  scores.append(80)
    elif total_var > 0.5:  scores.append(70)
    else:                  scores.append(58)

    # Jerk smoothness — human motion is smooth
    if jerks:
        p95 = jerks[int(len(jerks)*0.95)]
        if p95 < 100:   scores.append(90)
        elif p95 < 250: scores.append(80)
        else:           scores.append(65)

    # Signal continuity — no sudden jumps
    diffs = [abs(ax[i]-ax[i-1]) for i in range(1,n)]
    max_diff = max(diffs) if diffs else 0
    if max_diff < 5.0:   scores.append(85)
    elif max_diff < 15.0: scores.append(70)
    else:                 scores.append(55)

    return int(sum(scores)/len(scores)) if scores else 0


def get_tier(score):
    if score >= 87: return 'GOLD'
    if score >= 75: return 'SILVER'
    if score >= 60: return 'BRONZE'
    return 'REJECTED'


# ══════════════════════════════════════════════════════════════════
# KALMAN RTS SMOOTHER
# ══════════════════════════════════════════════════════════════════

def kalman_smooth(accel_window, hz=100):
    """RTS Kalman smoother, constant velocity model, 100Hz."""
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

        result = [xs[-1][0]]
        for k in range(n-2, -1, -1):
            xf, Pf = xs[k], Ps[k]
            xp = [xf[0]+dt*xf[1], xf[1]]
            Pp = [[Pf[0][0]+dt*Pf[1][0]+dt*(Pf[0][1]+dt*Pf[1][1])+q*dt**3/3,
                   Pf[0][1]+dt*Pf[1][1]+q*dt**2/2],
                  [Pf[1][0]+dt*Pf[1][1]+q*dt**2/2, Pf[1][1]+q*dt]]
            det = Pp[0][0]*Pp[1][1]-Pp[0][1]*Pp[1][0]
            if abs(det) < 1e-10:
                result.insert(0, xf[0]); continue
            G00 = (Pf[0][0]*Pp[1][1]-Pf[0][1]*Pp[1][0])/det
            result.insert(0, xf[0]+G00*(result[0]-xp[0]))

        smoothed_axes.append(result)

    return [[smoothed_axes[0][i], smoothed_axes[1][i], smoothed_axes[2][i]]
            for i in range(n)]


# ══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_features(accel):
    n = len(accel)
    if n < 20:
        return None

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

    # Dominant frequency
    N = n
    fft_mag = []
    for k in range(1, min(N//2, 15)):
        re = sum(mags[i]*math.cos(2*math.pi*k*i/N) for i in range(N))
        im = sum(mags[i]*math.sin(2*math.pi*k*i/N) for i in range(N))
        fft_mag.append(math.sqrt(re**2+im**2))
    dom_freq = fft_mag.index(max(fft_mag))+1 if fft_mag else 0
    feats.append(dom_freq)

    # ZCR on demeaned x-axis
    raw_x = [accel[i][0] for i in range(n)]
    mean_x = sum(raw_x)/n
    dx = [v-mean_x for v in raw_x]
    zcr = sum(1 for i in range(1,n) if dx[i]*dx[i-1] < 0) / n
    feats.append(zcr)

    return feats


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_pamap2(data_dir, subjects=None):
    """Load PAMAP2 .dat files, window into segments."""
    print(f"Loading PAMAP2 from {data_dir}...")
    all_windows = []

    dat_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.dat')])
    if subjects:
        dat_files = [f for f in dat_files if any(f'subject{s:03d}' in f or f'subject{s}' in f for s in subjects)]

    for fname in dat_files:
        subj_id = int(''.join(filter(str.isdigit, fname.split('.')[0])))
        fpath = os.path.join(data_dir, fname)
        fsize = os.path.getsize(fpath)

        # Skip corrupt subject101 (same size as zip)
        if fsize > 400 * 1024 * 1024:
            print(f"  Skipping {fname} (likely corrupt, {fsize//1024//1024}MB)")
            continue

        rows = []
        try:
            with open(fpath) as f:
                for line in f:
                    cols = line.strip().split()
                    if len(cols) < 24:
                        continue
                    try:
                        act = int(float(cols[COL_ACTIVITY]))
                        if act not in ACTIVITY_LABELS:
                            continue
                        ax = float(cols[COL_CHEST_AX])
                        ay = float(cols[COL_CHEST_AY])
                        az = float(cols[COL_CHEST_AZ])
                        if math.isnan(ax) or math.isnan(ay) or math.isnan(az):
                            continue
                        rows.append((act, [ax, ay, az]))
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print(f"  Error reading {fname}: {e}")
            continue

        # Slide window
        wins = 0
        i = 0
        while i + WINDOW_SIZE <= len(rows):
            # Check activity is consistent in window
            acts = [rows[i+j][0] for j in range(WINDOW_SIZE)]
            dominant = max(set(acts), key=acts.count)
            if acts.count(dominant) / WINDOW_SIZE < 0.8:
                i += STEP_SIZE
                continue
            accel = [rows[i+j][1] for j in range(WINDOW_SIZE)]
            domain = ACTIVITY_LABELS[dominant]
            all_windows.append({
                'subject': subj_id,
                'activity': dominant,
                'domain': domain,
                'accel': accel,
            })
            wins += 1
            i += STEP_SIZE

        print(f"  {fname}: {len(rows)} rows → {wins} windows")

    print(f"  Total windows: {len(all_windows)}")
    return all_windows


# ══════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════

class MLP:
    def __init__(self, input_dim, hidden, output, lr=0.001):
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
        exp_l = [math.exp(min(l-mx,60)) for l in logits]
        s = sum(exp_l)
        probs = [e/s for e in exp_l]
        return h, probs

    def backward(self, x, h, probs, label):
        d2 = list(probs); d2[label] -= 1.0
        for k in range(len(self.W2)):
            for j in range(len(h)):
                self.W2[k][j] -= self.lr*d2[k]*h[j]
            self.b2[k] -= self.lr*d2[k]
        d1 = [sum(self.W2[k][j]*d2[k] for k in range(len(self.W2)))*(1 if h[j]>0 else 0)
              for j in range(len(self.W1))]
        for j in range(len(self.W1)):
            for i in range(len(x)):
                self.W1[j][i] -= self.lr*d1[j]*x[i]
            self.b1[j] -= self.lr*d1[j]

    def _clip(self, max_norm=5.0):
        for row in self.W1+self.W2:
            n = math.sqrt(sum(v**2 for v in row))
            if n > max_norm:
                for i in range(len(row)): row[i] *= max_norm/n

    def train_epoch(self, samples):
        random.shuffle(samples)
        loss, skips = 0.0, 0
        for item in samples:
            feat, label, w = item if len(item)==3 else (*item, 1.0)
            h, probs = self.forward(feat)
            sl = -math.log(max(probs[label],1e-10))
            if sl != sl: skips+=1; continue
            loss += sl*w
            orig = self.lr; self.lr *= w
            self.backward(feat, h, probs, label)
            self.lr = orig
        self._clip()
        return loss/max(len(samples)-skips,1)

    def evaluate(self, samples):
        correct = 0
        pc = defaultdict(lambda:[0,0])
        for feat, label in samples:
            _, probs = self.forward(feat)
            pred = probs.index(max(probs))
            pc[label][1] += 1
            if pred == label: correct+=1; pc[label][0]+=1
        acc = correct/max(len(samples),1)
        f1s = [pc[i][0]/max(pc[i][1],1) for i in range(len(DOMAINS))]
        return acc, sum(f1s)/len(f1s)


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
    return [[(x[i]-means[i])/stds[i] for i in range(n_feat)] for x in X], stats


# ══════════════════════════════════════════════════════════════════
# CORRUPTION INJECTION
# ══════════════════════════════════════════════════════════════════

def inject_corruption(windows, rate=CORRUPTION_RATE):
    random.seed(123)
    import copy
    n = int(len(windows)*rate)
    corrupt_idx = set(random.sample(range(len(windows)), n))
    counts = {'flat':0,'clip':0,'label':0}
    result = []
    for i, w in enumerate(windows):
        if i not in corrupt_idx:
            result.append(w); continue
        r = copy.deepcopy(w)
        ctype = i % 3
        accel = r['accel']
        ns = len(accel)
        if ctype == 0:  # flat signal
            r['accel'] = [[random.gauss(0,0.001)]*3 for _ in range(ns)]
            counts['flat'] += 1
        elif ctype == 1:  # clipping — use p95 of actual signal
            clip = 16.0  # m/s² — p99 of real signal, actual saturation
            r['accel'] = [[max(-clip,min(clip,v)) for v in row] for row in accel]
            counts['clip'] += 1
        else:  # label noise
            acts = [a for a in ACTIVITY_LABELS if a != r['activity']]
            if acts:
                new_act = random.choice(acts)
                r['activity'] = new_act
                r['domain'] = ACTIVITY_LABELS[new_act]
            counts['label'] += 1
        result.append(r)
    print(f"  Injected {sum(counts.values())} corruptions ({rate*100:.0f}%):")
    for k,v in counts.items(): print(f"    {k:<10} {v}")
    return result


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run(data_dir, output_path, epochs=40):
    print()
    print("S2S Level 3 — Kalman Reconstruction (PAMAP2)")
    print("Real 100Hz chest IMU  |  9 subjects  |  12 activities → 3 domains")
    print("=" * 62)

    # Load
    all_windows = load_pamap2(data_dir)
    if len(all_windows) < 500:
        print("ERROR: Not enough windows loaded")
        sys.exit(1)

    # Subject split
    subjects = sorted(set(w['subject'] for w in all_windows))
    random.seed(42); random.shuffle(subjects)
    n_test = max(1, int(len(subjects)*0.2))
    test_ids = set(subjects[:n_test])
    train_w = [w for w in all_windows if w['subject'] not in test_ids]
    test_w  = [w for w in all_windows if w['subject'] in test_ids]
    print(f"\nSubject split: train={len(subjects)-n_test} test={n_test}")
    print(f"Windows: train={len(train_w)} test={len(test_w)}")

    # Score clean train — live scorer
    print("\nLive-scoring clean train windows...")
    scores_clean = []
    for w in train_w:
        s = score_window(w['accel'])
        scores_clean.append(s)
    scores_sorted = sorted(scores_clean)
    # Floor from non-zero scores only — zeros are hard REJECTED
    nonzero = [s for s in scores_sorted if s > 0]
    floor = nonzero[int(len(nonzero)*0.25)] if nonzero else 60
    print(f"  Score avg (non-zero): {sum(nonzero)/max(len(nonzero),1):.1f}")
    print(f"  Non-zero: {len(nonzero)}/{len(scores_sorted)}")
    print(f"  Adaptive floor (p25 of non-zero): {floor}")
    from collections import Counter
    tier_counts = Counter(get_tier(s) for s in scores_clean)
    for t in ['GOLD','SILVER','BRONZE','REJECTED']:
        print(f"    {t:<10} {tier_counts.get(t,0)}")

    # Build test samples
    test_feats = [extract_features(w['accel']) for w in test_w]
    test_labels = [DOM2IDX[w['domain']] for w in test_w]
    test_valid = [(f,l) for f,l in zip(test_feats,test_labels) if f is not None]

    # Normalization stats from clean train
    train_feats_all = [extract_features(w['accel']) for w in train_w]
    valid_feats = [f for f in train_feats_all if f is not None]
    _, norm_stats = normalize(valid_feats)

    def norm(feats_list):
        fn, _ = normalize(feats_list, norm_stats)
        return fn

    test_feats_n = norm([f for f,_ in test_valid])
    test_eval = list(zip(test_feats_n, [l for _,l in test_valid]))
    print(f"Test samples: {len(test_eval)}")

    # Inject corruption
    print(f"\nInjecting corruption ({CORRUPTION_RATE*100:.0f}%)...")
    train_corrupted = inject_corruption(train_w)

    # S2S floor — live score AFTER corruption
    print("\nApplying S2S adaptive floor...")
    scores_corr = [score_window(w['accel']) for w in train_corrupted[:500]]
    train_filtered = [w for w in train_corrupted if score_window(w['accel']) >= floor]
    removed = len(train_corrupted)-len(train_filtered)
    print(f"  Score avg corrupted: {sum(scores_corr)/len(scores_corr):.1f}")
    print(f"  Kept {len(train_filtered)} / {len(train_corrupted)} (removed {removed})")

    # Kalman reconstruction of BRONZE
    print("\nKalman reconstruction of BRONZE windows...")
    reconstructed = []
    bronze_total = passed = failed = 0
    for w in train_filtered:
        s = score_window(w['accel'])
        if 60 <= s < 75:
            bronze_total += 1
            import copy
            smoothed = kalman_smooth(w['accel'], hz=HZ)
            test_w2 = copy.deepcopy(w)
            test_w2['accel'] = smoothed
            new_score = score_window(test_w2['accel'])
            if new_score >= 75:
                reconstructed.append(test_w2)
                passed += 1
            else:
                failed += 1
    acceptance = passed/max(bronze_total,1)*100
    print(f"  BRONZE windows:         {bronze_total}")
    print(f"  Passed re-check (>=75): {passed}")
    print(f"  Failed re-check:        {failed}")
    print(f"  Acceptance rate:        {acceptance:.1f}%")

    # Build conditions
    def make_s(windows, weight=1.0):
        out = []
        for w in windows:
            f = extract_features(w['accel'])
            if f and w['domain'] in DOM2IDX:
                out.append((f, DOM2IDX[w['domain']], weight))
        return out

    sA = make_s(train_w)
    sB = make_s(train_corrupted)
    sC = make_s(train_filtered)
    sD = make_s(train_filtered) + make_s(reconstructed, weight=0.5)

    def norm_s(samples):
        feats=[f for f,_,_ in samples]
        labels=[l for _,l,_ in samples]
        weights=[w for _,_,w in samples]
        fn=norm(feats)
        return list(zip(fn,labels,weights))

    sA_n=norm_s(sA); sB_n=norm_s(sB); sC_n=norm_s(sC); sD_n=norm_s(sD)

    print(f"\nCondition sizes:")
    print(f"  A (clean):      {len(sA_n)}")
    print(f"  B (corrupted):  {len(sB_n)}")
    print(f"  C (floor):      {len(sC_n)}")
    print(f"  D (floor+Kalman): {len(sD_n)} (+{passed} reconstructed)")

    # Train
    results = {}
    conditions = [
        ("A_clean",       sA_n, "Clean PAMAP2 baseline"),
        ("B_corrupted",   sB_n, f"Corrupted {CORRUPTION_RATE*100:.0f}%, no filter"),
        ("C_s2s_floor",   sC_n, "Adaptive S2S floor"),
        ("D_kalman_recon",sD_n, f"Floor + Kalman recon (w=0.5, n={passed})"),
    ]

    for cond, train_data, desc in conditions:
        print(f"\n{'─'*62}")
        print(f"Condition {cond}  n={len(train_data)}")
        print(f"  {desc}")
        model = MLP(input_dim=14, hidden=64, output=len(DOMAINS), lr=0.001)
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
        results[cond] = {"description":desc,"accuracy":round(acc,4),
                         "macro_f1":round(f1,4),"train_n":len(train_data)}

    # Summary
    print(f"\n{'='*62}")
    print("  LEVEL 3 RESULTS — PAMAP2 (Real 100Hz IMU)")
    print(f"{'='*62}\n")
    print(f"  {'Condition':<22} {'F1':>6}  {'n':>6}")
    print("  "+"-"*40)
    for c,r in results.items():
        print(f"  {c:<22} {r['macro_f1']:.4f}  {r['train_n']:>6}")

    rA=results.get('A_clean',{})
    rB=results.get('B_corrupted',{})
    rC=results.get('C_s2s_floor',{})
    rD=results.get('D_kalman_recon',{})

    print()
    if all([rA,rB,rC,rD]):
        damage  = rB['macro_f1']-rA['macro_f1']
        recover = rC['macro_f1']-rB['macro_f1']
        kalman  = rD['macro_f1']-rC['macro_f1']
        net     = rD['macro_f1']-rA['macro_f1']
        print(f"  Corruption damage  (B-A): {damage*100:+.2f}% F1")
        print(f"  S2S floor recovery (C-B): {recover*100:+.2f}% F1")
        print(f"  Kalman gain        (D-C): {kalman*100:+.2f}% F1  ← Level 3 claim")
        print(f"  Net vs baseline    (D-A): {net*100:+.2f}% F1")
        print(f"  Reconstructed:            {passed} ({acceptance:.1f}% acceptance)")
        print()
        if damage < -0.005 and recover > 0.005 and kalman > 0.002:
            v = "ALL THREE LEVELS PROVEN ON REAL 100Hz IMU DATA"
            icon = "✓"
        elif kalman > 0.002:
            v = "LEVEL 3 PROVEN — Kalman improves over floor"
            icon = "✓"
        elif kalman > -0.002:
            v = "Neutral — reconstruction neither helps nor hurts"
            icon = "~"
        else:
            v = "Reconstruction did not improve over floor"
            icon = "✗"
        print(f"  {icon} {v}")

    results['meta'] = {
        "experiment":"s2s_level3_pamap2","dataset":"PAMAP2",
        "hz":HZ,"window_size":WINDOW_SIZE,
        "subjects_used":len(subjects),
        "bronze_found":bronze_total,"bronze_reconstructed":passed,
        "acceptance_rate":round(acceptance/100,3),
        "adaptive_floor":floor,"corruption_rate":CORRUPTION_RATE,
    }
    with open(output_path,'w') as f:
        json.dump(results,f,indent=2)
    print(f"\n  Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/pamap2/')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--out',    default='experiments/results_level3_pamap2.json')
    args = parser.parse_args()
    run(args.data, args.out, args.epochs)
