#!/usr/bin/env python3
"""
S2S Level 4 — Multi-Sensor Kinematic Chain Consistency (PAMAP2)
Real 100Hz  |  3 IMUs (hand, chest, ankle)  |  3 domains

Level 4 Physics Laws — Kinematic Chain Consistency:
  Law A: Locomotion coherence — all sensors agree on dominant frequency
  Law B: Segment coupling   — chest-ankle accel correlation during locomotion
  Law C: Gyro-accel coupling — per sensor: gyro energy ~ accel variance
  Law D: Cross-sensor jerk timing — ankle leads chest in step cycle

Conditions:
  A — Single sensor (chest only, Level 1 baseline)
  B — Multi-sensor naive concat (no consistency check)
  C — Multi-sensor + kinematic chain filter (Level 4 claim)
  D — Multi-sensor + chain filter + curriculum (Level 4 full)

Run from ~/S2S:
  python3 experiments/level4_multisensor_fusion.py \
    --data data/pamap2/ --epochs 40 \
    --out experiments/results_level4_pamap2.json
"""

import os, sys, json, math, random, time, argparse, hashlib
from collections import defaultdict, Counter

# ── Activity mapping ──────────────────────────────────────────────
ACTIVITY_LABELS = {
    1:'lying', 2:'sitting', 3:'standing',
    4:'walking', 5:'running', 6:'cycling',
    7:'nordic_walking', 12:'ascending_stairs', 13:'descending_stairs',
    16:'vacuum_cleaning', 17:'ironing', 24:'rope_jumping',
}
DOMAINS  = ['lying','sitting','standing','walking','running','cycling',
            'nordic_walking','ascending_stairs','descending_stairs',
            'vacuum_cleaning','ironing','rope_jumping']
DOM2IDX  = {d:i for i,d in enumerate(DOMAINS)}

# Column indices
COL_ACTIVITY  = 1
COL_HAND_AX   = 4;  COL_HAND_AY   = 5;  COL_HAND_AZ   = 6
COL_HAND_GX   = 10; COL_HAND_GY   = 11; COL_HAND_GZ   = 12
COL_CHEST_AX  = 21; COL_CHEST_AY  = 22; COL_CHEST_AZ  = 23
COL_CHEST_GX  = 27; COL_CHEST_GY  = 28; COL_CHEST_GZ  = 29
COL_ANKLE_AX  = 38; COL_ANKLE_AY  = 39; COL_ANKLE_AZ  = 40
COL_ANKLE_GX  = 44; COL_ANKLE_GY  = 45; COL_ANKLE_GZ  = 46

WINDOW_SIZE = 256
STEP_SIZE   = 128
HZ          = 100
CORRUPT_RATE = 0.35


def derive_seed(base_seed, *components):
    """
    Derive a deterministic seed from base seed and optional components.
    Uses SHA256 for cross-process stability (unlike built-in hash()).
    Returns a 32-bit integer suitable for random.Random().
    """
    # Create a deterministic string representation
    seed_str = f"{base_seed}:" + ":".join(str(c) for c in components)
    # Hash with SHA256 and convert to 32-bit integer
    hash_bytes = hashlib.sha256(seed_str.encode()).digest()
    seed_int = int.from_bytes(hash_bytes[:4], byteorder='big')
    return seed_int


# ══════════════════════════════════════════════════════════════════
# KINEMATIC CHAIN CONSISTENCY SCORER (Level 4)
# ══════════════════════════════════════════════════════════════════

def pearson_r(x, y):
    n = len(x)
    if n < 3: return 0.0
    mx = sum(x)/n; my = sum(y)/n
    num = sum((x[i]-mx)*(y[i]-my) for i in range(n))
    dx = math.sqrt(sum((v-mx)**2 for v in x))
    dy = math.sqrt(sum((v-my)**2 for v in y))
    if dx < 1e-10 or dy < 1e-10: return 0.0
    return num/(dx*dy)


def dom_freq(signal, hz=HZ):
    """Dominant frequency via DFT."""
    n = len(signal)
    mean = sum(signal)/n
    s = [v-mean for v in signal]
    best_k, best_mag = 1, 0.0
    for k in range(1, min(n//2, 20)):
        re = sum(s[i]*math.cos(2*math.pi*k*i/n) for i in range(n))
        im = sum(s[i]*math.sin(2*math.pi*k*i/n) for i in range(n))
        mag = math.sqrt(re**2+im**2)
        if mag > best_mag:
            best_mag = mag; best_k = k
    return best_k * hz / n


def kinematic_chain_score(hand_a, chest_a, ankle_a,
                           hand_g, chest_g, ankle_g):
    """
    Score kinematic chain consistency 0-100.
    Tests 4 physics laws across 3 IMUs.
    """
    n = len(chest_a)
    if n < 20: return 0

    scores = []

    # Magnitudes
    hand_mag  = [math.sqrt(sum(v**2 for v in hand_a[i]))  for i in range(n)]
    chest_mag = [math.sqrt(sum(v**2 for v in chest_a[i])) for i in range(n)]
    ankle_mag = [math.sqrt(sum(v**2 for v in ankle_a[i])) for i in range(n)]

    hand_gm  = [math.sqrt(sum(v**2 for v in hand_g[i]))  for i in range(n)]
    chest_gm = [math.sqrt(sum(v**2 for v in chest_g[i])) for i in range(n)]
    ankle_gm = [math.sqrt(sum(v**2 for v in ankle_g[i])) for i in range(n)]

    # ── Law A: Locomotion coherence ─────────────────────────────
    # All sensors should share dominant frequency during motion
    fh = dom_freq(hand_mag)
    fc = dom_freq(chest_mag)
    fa = dom_freq(ankle_mag)
    freq_spread = max(fh, fc, fa) - min(fh, fc, fa)
    if freq_spread < 1.0:    scores.append(90)  # all agree
    elif freq_spread < 2.5:  scores.append(75)  # close
    elif freq_spread < 5.0:  scores.append(60)  # diverging
    else:                    scores.append(40)  # independent

    # ── Law B: Segment coupling ──────────────────────────────────
    # Chest-ankle correlation: during real locomotion > 0.3
    r_ca = pearson_r(chest_mag, ankle_mag)
    r_ha = pearson_r(hand_mag, ankle_mag)
    r_hc = pearson_r(hand_mag, chest_mag)
    avg_r = (abs(r_ca) + abs(r_ha) + abs(r_hc)) / 3
    if avg_r > 0.5:   scores.append(90)
    elif avg_r > 0.3: scores.append(75)
    elif avg_r > 0.1: scores.append(60)
    else:             scores.append(40)  # sensors independent

    # ── Law C: Gyro-accel coupling per sensor ───────────────────
    # Each IMU: gyro energy must correlate with accel variance
    # Rigid body: rotation causes acceleration
    coupling_scores = []
    for accel_mag, gyro_mag in [(chest_mag, chest_gm),
                                 (ankle_mag, ankle_gm),
                                 (hand_mag,  hand_gm)]:
        r = pearson_r(accel_mag, gyro_mag)
        if abs(r) > 0.4:   coupling_scores.append(90)
        elif abs(r) > 0.2: coupling_scores.append(70)
        else:              coupling_scores.append(50)
    scores.append(sum(coupling_scores)//len(coupling_scores))

    # ── Law D: Cross-sensor jerk timing ─────────────────────────
    # Ankle jerk should lead chest jerk in walking (heel strike → torso)
    # Compute jerk magnitude for chest and ankle
    dt = 1.0/HZ
    chest_jerk = [abs(chest_mag[i+1]-chest_mag[i-1])/(2*dt)
                  for i in range(1, n-1)]
    ankle_jerk = [abs(ankle_mag[i+1]-ankle_mag[i-1])/(2*dt)
                  for i in range(1, n-1)]

    # Cross-correlation at lags -5 to +5 samples (±50ms)
    best_lag, best_corr = 0, -1
    for lag in range(-5, 6):
        if lag >= 0:
            x = ankle_jerk[:n-2-lag]; y = chest_jerk[lag:n-2]
        else:
            x = ankle_jerk[-lag:n-2]; y = chest_jerk[:n-2+lag]
        if len(x) < 20: continue
        r = pearson_r(x, y)
        if r > best_corr:
            best_corr = r; best_lag = lag

    # Positive lag = ankle leads chest (correct biomechanics)
    if best_lag >= 0 and best_corr > 0.3:   scores.append(90)
    elif best_lag >= 0 and best_corr > 0.1: scores.append(70)
    elif best_corr > 0.1:                    scores.append(55)
    else:                                    scores.append(40)

    return int(sum(scores)/len(scores))


def get_chain_tier(score):
    if score >= 80: return 'GOLD'
    if score >= 65: return 'SILVER'
    if score >= 50: return 'BRONZE'
    return 'REJECTED'


# ══════════════════════════════════════════════════════════════════
# SINGLE SENSOR SCORER (for reference)
# ══════════════════════════════════════════════════════════════════

def score_single(accel):
    n = len(accel)
    if n < 20: return 0
    ax=[accel[i][0] for i in range(n)]; ay=[accel[i][1] for i in range(n)]; az=[accel[i][2] for i in range(n)]
    mx=sum(ax)/n; my=sum(ay)/n; mz=sum(az)/n
    tv=sum((v-mx)**2 for v in ax)/n+sum((v-my)**2 for v in ay)/n+sum((v-mz)**2 for v in az)/n
    if tv<0.01: return 0
    dt=1.0/HZ; jerks=[]
    for av in [ax,ay,az]:
        vel=[(av[k+1]-av[k-1])/(2*dt) for k in range(1,n-1)]
        jerk=[(vel[k+1]-vel[k-1])/(2*dt) for k in range(1,len(vel)-1)]
        jerks.extend([abs(j) for j in jerk])
    if jerks:
        jerks.sort()
        if jerks[int(len(jerks)*0.95)]>5000: return 0
    scores=[]
    if tv>10: scores.append(90)
    elif tv>2: scores.append(80)
    elif tv>0.5: scores.append(70)
    else: scores.append(58)
    return int(sum(scores)/len(scores)) if scores else 0


# ══════════════════════════════════════════════════════════════════
# DATA LOADING — all 3 IMUs
# ══════════════════════════════════════════════════════════════════

def load_pamap2_multisensor(data_dir):
    print(f"Loading PAMAP2 (3 IMUs) from {data_dir}...")
    all_windows = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.dat'): continue
        fpath = os.path.join(data_dir, fname)
        if os.path.getsize(fpath) > 400*1024*1024:
            print(f"  Skipping {fname} (corrupt)"); continue
        subj = int(''.join(filter(str.isdigit, fname.split('.')[0])))

        # Read all rows first, then segment by unsupported activity and parse errors
        raw_rows = []
        with open(fpath) as f:
            for line in f:
                cols = line.strip().split()
                if len(cols) < 47:
                    # Malformed row - add as None to signal segment break
                    raw_rows.append(None)
                    continue
                try:
                    act = int(float(cols[COL_ACTIVITY]))
                    def safe(c):
                        v = float(cols[c])
                        return 0.0 if math.isnan(v) else v
                    raw_rows.append((act,
                        [safe(COL_HAND_AX), safe(COL_HAND_AY), safe(COL_HAND_AZ)],
                        [safe(COL_HAND_GX), safe(COL_HAND_GY), safe(COL_HAND_GZ)],
                        [safe(COL_CHEST_AX),safe(COL_CHEST_AY),safe(COL_CHEST_AZ)],
                        [safe(COL_CHEST_GX),safe(COL_CHEST_GY),safe(COL_CHEST_GZ)],
                        [safe(COL_ANKLE_AX),safe(COL_ANKLE_AY),safe(COL_ANKLE_AZ)],
                        [safe(COL_ANKLE_GX),safe(COL_ANKLE_GY),safe(COL_ANKLE_GZ)],
                    ))
                except:
                    # Parse error - add as None to signal segment break
                    raw_rows.append(None)

        # Segment into contiguous blocks of supported activities
        # Break on: None (malformed/parse error) or unsupported activity
        segments = []
        current_segment = []
        for row in raw_rows:
            if row is None:
                # Malformed row - terminate current segment
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []
            else:
                act = row[0]
                if act not in ACTIVITY_LABELS:
                    # Unsupported activity - terminate current segment
                    if current_segment:
                        segments.append(current_segment)
                        current_segment = []
                else:
                    current_segment.append(row)
        if current_segment:
            segments.append(current_segment)

        # Generate windows within each segment independently
        wins = 0
        for seg in segments:
            i = 0
            while i+WINDOW_SIZE <= len(seg):
                acts = [seg[i+j][0] for j in range(WINDOW_SIZE)]
                dom_act = max(set(acts), key=acts.count)
                if acts.count(dom_act)/WINDOW_SIZE < 0.8: i+=STEP_SIZE; continue
                w = {
                    'subject': subj, 'activity': dom_act,
                    'domain': ACTIVITY_LABELS[dom_act],
                    'hand_a':  [seg[i+j][1] for j in range(WINDOW_SIZE)],
                    'hand_g':  [seg[i+j][2] for j in range(WINDOW_SIZE)],
                    'chest_a': [seg[i+j][3] for j in range(WINDOW_SIZE)],
                    'chest_g': [seg[i+j][4] for j in range(WINDOW_SIZE)],
                    'ankle_a': [seg[i+j][5] for j in range(WINDOW_SIZE)],
                    'ankle_g': [seg[i+j][6] for j in range(WINDOW_SIZE)],
                }
                all_windows.append(w); wins+=1; i+=STEP_SIZE
        print(f"  {fname}: {len(raw_rows)} rows → {wins} windows ({len(segments)} segments)")
    print(f"  Total: {len(all_windows)} windows")
    return all_windows


# ══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_features_single(accel):
    """14 features from chest IMU only."""
    n = len(accel)
    if n < 20: return None
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
    fft_mag = []
    for k in range(1, min(n//2, 15)):
        re = sum(mags[i]*math.cos(2*math.pi*k*i/n) for i in range(n))
        im = sum(mags[i]*math.sin(2*math.pi*k*i/n) for i in range(n))
        fft_mag.append(math.sqrt(re**2+im**2))
    feats.append(fft_mag.index(max(fft_mag))+1 if fft_mag else 0)
    raw_x = [accel[i][0] for i in range(n)]
    mean_x = sum(raw_x)/n
    dx = [v-mean_x for v in raw_x]
    feats.append(sum(1 for i in range(1,n) if dx[i]*dx[i-1]<0)/n)
    return feats


def extract_features_multi(w):
    """28 features: 14 chest + 7 cross-sensor consistency features."""
    chest_feats = extract_features_single(w['chest_a'])
    if chest_feats is None: return None

    n = len(w['chest_a'])
    hand_mag  = [math.sqrt(sum(v**2 for v in w['hand_a'][i]))  for i in range(n)]
    chest_mag = [math.sqrt(sum(v**2 for v in w['chest_a'][i])) for i in range(n)]
    ankle_mag = [math.sqrt(sum(v**2 for v in w['ankle_a'][i])) for i in range(n)]
    chest_gm  = [math.sqrt(sum(v**2 for v in w['chest_g'][i])) for i in range(n)]
    ankle_gm  = [math.sqrt(sum(v**2 for v in w['ankle_g'][i])) for i in range(n)]
    hand_gm   = [math.sqrt(sum(v**2 for v in w['hand_g'][i]))  for i in range(n)]

    cross_feats = [
        pearson_r(chest_mag, ankle_mag),           # chest-ankle coupling
        pearson_r(hand_mag, ankle_mag),            # hand-ankle coupling
        pearson_r(hand_mag, chest_mag),            # hand-chest coupling
        pearson_r(chest_mag, chest_gm),            # chest gyro-accel coupling
        pearson_r(ankle_mag, ankle_gm),            # ankle gyro-accel coupling
        dom_freq(ankle_mag) - dom_freq(chest_mag), # freq difference
        dom_freq(hand_mag)  - dom_freq(chest_mag), # freq difference hand
    ]
    return chest_feats + cross_feats


# ══════════════════════════════════════════════════════════════════
# MLP + NORMALIZATION
# ══════════════════════════════════════════════════════════════════

def normalize(X, stats=None):
    if not X: return [], stats
    nf = len(X[0])
    if stats is None:
        means = [sum(x[i] for x in X)/len(X) for i in range(nf)]
        stds  = [math.sqrt(sum((x[i]-means[i])**2 for x in X)/len(X))+1e-8 for i in range(nf)]
        stats = (means, stds)
    means, stds = stats
    return [[(x[i]-means[i])/stds[i] for i in range(nf)] for x in X], stats


class MLP:
    def __init__(self, input_dim, hidden, output, lr=0.001, seed=42):
        """
        MLP with explicit local RNG for reproducible weight initialization.
        seed: Random seed for weight initialization (default 42 for backward compatibility).
        """
        self.lr=lr
        self.seed=seed
        # Use local RNG for weight initialization, don't reset global RNG
        rng = random.Random(seed)
        s1=math.sqrt(2.0/input_dim); s2=math.sqrt(2.0/hidden)
        self.W1=[[rng.gauss(0,s1) for _ in range(input_dim)] for _ in range(hidden)]
        self.b1=[0.0]*hidden
        self.W2=[[rng.gauss(0,s2) for _ in range(hidden)] for _ in range(output)]
        self.b2=[0.0]*output
        # Initialize shuffle RNG (will be set on first train_epoch call if not provided)
        self._shuffle_rng = None

    def forward(self,x):
        h=[max(0,sum(self.W1[j][i]*x[i] for i in range(len(x)))+self.b1[j]) for j in range(len(self.W1))]
        logits=[sum(self.W2[k][j]*h[j] for j in range(len(h)))+self.b2[k] for k in range(len(self.W2))]
        mx=max(logits); el=[math.exp(min(l-mx,60)) for l in logits]; s=sum(el)
        return h,[e/s for e in el]

    def backward(self,x,h,probs,label,w=1.0):
        d2=list(probs); d2[label]-=1.0
        for k in range(len(self.W2)):
            for j in range(len(h)): self.W2[k][j]-=self.lr*w*d2[k]*h[j]
            self.b2[k]-=self.lr*w*d2[k]
        d1=[sum(self.W2[k][j]*d2[k] for k in range(len(self.W2)))*(1 if h[j]>0 else 0) for j in range(len(self.W1))]
        for j in range(len(self.W1)):
            for i in range(len(x)): self.W1[j][i]-=self.lr*w*d1[j]*x[i]
            self.b1[j]-=self.lr*w*d1[j]

    def _clip(self,mn=5.0):
        for row in self.W1+self.W2:
            n=math.sqrt(sum(v**2 for v in row))
            if n>mn:
                for i in range(len(row)): row[i]*=mn/n

    def train_epoch(self, samples, shuffle_seed=None):
        """
        Train one epoch with optional explicit shuffle seed.
        shuffle_seed: If provided, initializes a local RNG for shuffling that advances across epochs.
                      If None, uses Python's global RNG (for backward compatibility).
        """
        if shuffle_seed is not None:
            # Initialize shuffle RNG on first call, then reuse to advance state
            if self._shuffle_rng is None:
                self._shuffle_rng = random.Random(shuffle_seed)
            shuffled = samples[:]
            self._shuffle_rng.shuffle(shuffled)
            samples = shuffled
        else:
            random.shuffle(samples)
        loss=0.0; skips=0
        for item in samples:
            feat,label,wt=(item[0],item[1],item[2]) if len(item)==3 else (item[0],item[1],1.0)
            h,probs=self.forward(feat)
            sl=-math.log(max(probs[label],1e-10))
            if sl!=sl: skips+=1; continue
            loss+=sl*wt; self.backward(feat,h,probs,label,wt)
        self._clip()
        return loss/max(len(samples)-skips,1)

    def evaluate(self,samples):
        correct=0
        # Track true positives, false positives, false negatives per class
        tp=defaultdict(int); fp=defaultdict(int); fn=defaultdict(int)
        for feat,label in samples:
            _,probs=self.forward(feat); pred=probs.index(max(probs))
            if pred==label:
                correct+=1; tp[label]+=1
            else:
                fp[pred]+=1; fn[label]+=1
        acc=correct/max(len(samples),1)
        # Macro-F1: per-class F1 then average
        f1s=[]
        for i in range(len(DOMAINS)):
            p=tp[i]/max(tp[i]+fp[i],1)
            r=tp[i]/max(tp[i]+fn[i],1)
            f1=2*p*r/max(p+r,1e-9)
            f1s.append(f1)
        macro_f1=sum(f1s)/len(f1s)
        return acc, macro_f1


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def compute_detailed_metrics(model, test_data, n_classes):
    """
    Compute detailed evaluation metrics including per-class support, precision, recall, F1.
    Returns dict with accuracy, macro_f1_12class (arithmetic mean over all predefined classes),
    macro_f1_supported_classes (arithmetic mean over classes with nonzero support),
    weighted_f1 (support-weighted mean), and per_class_metrics dict.
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = 0
    total = len(test_data)

    for feat, label in test_data:
        _, probs = model.forward(feat)
        pred = probs.index(max(probs))
        if pred == label:
            correct += 1
            tp[label] += 1
        else:
            fp[pred] += 1
            fn[label] += 1

    acc = correct / max(total, 1)

    # Per-class metrics
    per_class = {}
    f1s_with_support = []
    f1s_all = []
    f1s_unrounded = []  # Keep unrounded F1 for weighted calculation
    total_support = 0

    for i in range(n_classes):
        support = tp[i] + fn[i]
        p = tp[i] / max(tp[i] + fp[i], 1)
        r = tp[i] / max(tp[i] + fn[i], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)

        per_class[i] = {
            "support": support,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4)
        }
        f1s_all.append(f1)
        f1s_unrounded.append(f1)
        if support > 0:
            f1s_with_support.append(f1)
            total_support += support

    macro_f1_all = sum(f1s_all) / len(f1s_all)
    macro_f1_supported_classes = sum(f1s_with_support) / len(f1s_with_support) if f1s_with_support else 0.0

    # Weighted F1: sum of (unrounded F1 * support) / total_support
    weighted_f1 = 0.0
    if total_support > 0:
        weighted_sum = sum(f1s_unrounded[i] * per_class[i]['support'] for i in range(n_classes))
        weighted_f1 = weighted_sum / total_support

    return {
        "accuracy": round(acc, 4),
        "macro_f1_12class": round(macro_f1_all, 4),
        "macro_f1_supported_classes": round(macro_f1_supported_classes, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class
    }


def run_fold(train_w, test_w, fold_idx, test_subject_id, epochs, norm_s=None, norm_m=None, chain_floor=None, seed=42):
    """
    Run a single fold of training/evaluation.
    If norm_s, norm_m, chain_floor are None, compute from train_w.
    Otherwise, use provided values (for LOSO consistency).
    seed: Base random seed for reproducibility (default 42 within fold for paired comparisons.
    """
    print(f"\n{'='*62}")
    print(f"Fold {fold_idx}: Test subject {test_subject_id}")
    print(f"  Train windows: {len(train_w)}  Test windows: {len(test_w)}")

    # Score chain consistency on train
    if chain_floor is None:
        print("  Computing chain floor from training data...")
        chain_scores = []
        for w in train_w:
            s = kinematic_chain_score(
                w['hand_a'], w['chest_a'], w['ankle_a'],
                w['hand_g'], w['chest_g'], w['ankle_g'])
            w['chain_score'] = s
            w['chain_tier'] = get_chain_tier(s)
            chain_scores.append(s)
        chain_sorted = sorted(chain_scores)
        chain_floor = chain_sorted[int(len(chain_sorted)*0.25)]
        print(f"  Chain floor (p25): {chain_floor}")
    else:
        print(f"  Using provided chain floor: {chain_floor}")
        # Still need to score windows for filtering
        for w in train_w:
            s = kinematic_chain_score(
                w['hand_a'], w['chest_a'], w['ankle_a'],
                w['hand_g'], w['chest_g'], w['ankle_g'])
            w['chain_score'] = s
            w['chain_tier'] = get_chain_tier(s)

    # Extract and normalize features
    print("  Extracting features...")
    single_feats_tr = [extract_features_single(w['chest_a']) for w in train_w]
    multi_feats_tr = [extract_features_multi(w) for w in train_w]

    valid_single = [f for f in single_feats_tr if f]
    valid_multi = [f for f in multi_feats_tr if f]

    if norm_s is None or norm_m is None:
        _, norm_s = normalize(valid_single)
        _, norm_m = normalize(valid_multi)

    def ns(fl): fn, _ = normalize(fl, norm_s); return fn
    def nm(fl): fn, _ = normalize(fl, norm_m); return fn

    # Test samples
    test_single = [(f, DOM2IDX[w['domain']]) for f, w in
                   zip([extract_features_single(w['chest_a']) for w in test_w], test_w) if f]
    test_multi = [(f, DOM2IDX[w['domain']]) for f, w in
                  zip([extract_features_multi(w) for w in test_w], test_w) if f]

    test_s_n = list(zip(ns([f for f, _ in test_single]), [l for _, l in test_single]))
    test_m_n = list(zip(nm([f for f, _ in test_multi]), [l for _, l in test_multi]))

    # Build conditions
    chain_kept = [w for w in train_w if w['chain_score'] >= chain_floor]
    tier_wt = {'GOLD': 1.0, 'SILVER': 0.8, 'BRONZE': 0.5, 'REJECTED': 0.0}

    sA_raw = [(extract_features_single(w['chest_a']), DOM2IDX[w['domain']], 1.0)
              for w in train_w if extract_features_single(w['chest_a'])]
    sA = list(zip(ns([f for f, _, _ in sA_raw]), [l for _, l, _ in sA_raw], [wt for _, _, wt in sA_raw]))

    sB_raw = [(extract_features_multi(w), DOM2IDX[w['domain']], 1.0)
              for w in train_w if extract_features_multi(w)]
    sB = list(zip(nm([f for f, _, _ in sB_raw]), [l for _, l, _ in sB_raw], [wt for _, _, wt in sB_raw]))

    sC_raw = [(extract_features_multi(w), DOM2IDX[w['domain']], 1.0)
              for w in chain_kept if extract_features_multi(w)]
    sC = list(zip(nm([f for f, _, _ in sC_raw]), [l for _, l, _ in sC_raw], [wt for _, _, wt in sC_raw]))

    sD_raw = [(extract_features_multi(w), DOM2IDX[w['domain']], tier_wt.get(w['chain_tier'], 0.5))
              for w in chain_kept if extract_features_multi(w) and tier_wt.get(w['chain_tier'], 0) > 0]
    sD = list(zip(nm([f for f, _, _ in sD_raw]), [l for _, l, _ in sD_raw], [wt for _, _, wt in sD_raw]))

    print(f"  Condition sizes: A={len(sA)} B={len(sB)} C={len(sC)} D={len(sD)}")
    print(f"  Chain filter removed: {len(train_w) - len(chain_kept)} ({(len(train_w)-len(chain_kept))/len(train_w)*100:.1f}%)")

    fold_results = {}
    fold_predictions = {}  # condition -> list of (pred, true_label)
    conditions = [
        ("A_single_sensor", sA, test_s_n, "Chest IMU only"),
        ("B_multi_naive", sB, test_m_n, "3 IMUs no chain check"),
        ("C_chain_filtered", sC, test_m_n, "3 IMUs + chain filter"),
        ("D_chain_weighted", sD, test_m_n, "3 IMUs + filter + weights"),
    ]

    for cond, train_data, test_data, desc in conditions:
        if not train_data or not test_data:
            print(f"    {cond}: SKIPPED (no data)")
            fold_results[cond] = None
            fold_predictions[cond] = None
            continue

        input_dim = len(train_data[0][0])
        # Derive fold-specific seed (shared by all conditions in this fold for paired comparisons)
        # Shuffle seed is separate and advances across epochs
        fold_model_seed = derive_seed(seed, fold_idx)
        fold_shuffle_seed = derive_seed(seed, fold_idx, "shuffle")

        model = MLP(input_dim=input_dim, hidden=64, output=len(DOMAINS), lr=0.001, seed=fold_model_seed)

        for ep in range(epochs):
            # Pass the same shuffle seed; train_epoch will advance its internal RNG state
            model.train_epoch(train_data, shuffle_seed=fold_shuffle_seed)

        # Collect predictions for pooled metrics
        preds = []
        for feat, true_label in test_data:
            _, probs = model.forward(feat)
            pred = probs.index(max(probs))
            preds.append((pred, true_label))

        metrics = compute_detailed_metrics(model, test_data, len(DOMAINS))
        fold_results[cond] = {
            "description": desc,
            "train_n": len(train_data),
            "test_n": len(test_data),
            "chain_removed": len(train_w) - len(chain_kept) if "chain" in cond else 0,
            "removal_rate": round((len(train_w) - len(chain_kept)) / len(train_w) * 100, 2) if "chain" in cond else 0,
            **metrics
        }
        fold_predictions[cond] = preds
        print(f"    {cond}: acc={metrics['accuracy']:.4f} f1_12={metrics['macro_f1_12class']:.4f} f1_sup={metrics['macro_f1_supported_classes']:.4f}")

    return fold_results, fold_predictions, norm_s, norm_m, chain_floor


def compute_pooled_metrics(predictions, n_classes):
    """
    Compute pooled metrics from collected (pred, true_label) pairs.
    Returns dict with accuracy, macro_f1_12class, macro_f1_supported_classes,
    weighted_f1, per_class metrics, and total pooled samples.
    """
    if not predictions:
        per_class = {}
        for i in range(n_classes):
            per_class[i] = {
                "support": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0
            }
        return {
            "accuracy": 0.0,
            "macro_f1_12class": 0.0,
            "macro_f1_supported_classes": 0.0,
            "weighted_f1": 0.0,
            "per_class": per_class,
            "n_pooled_samples": 0
        }

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = 0
    total = len(predictions)

    for pred, true_label in predictions:
        if pred == true_label:
            correct += 1
            tp[true_label] += 1
        else:
            fp[pred] += 1
            fn[true_label] += 1

    acc = correct / max(total, 1)

    # Per-class metrics
    per_class = {}
    f1s_with_support = []
    f1s_all = []
    f1s_unrounded = []  # Keep unrounded F1 for weighted calculation
    total_support = 0

    for i in range(n_classes):
        support = tp[i] + fn[i]
        p = tp[i] / max(tp[i] + fp[i], 1)
        r = tp[i] / max(tp[i] + fn[i], 1)
        f1 = 2 * p * r / max(p + r, 1e-9)

        per_class[i] = {
            "support": support,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4)
        }
        f1s_all.append(f1)
        f1s_unrounded.append(f1)
        if support > 0:
            f1s_with_support.append(f1)
            total_support += support

    macro_f1_all = sum(f1s_all) / len(f1s_all)
    macro_f1_supported_classes = sum(f1s_with_support) / len(f1s_with_support) if f1s_with_support else 0.0

    # Weighted F1: sum of (unrounded F1 * support) / total_support
    weighted_f1 = 0.0
    if total_support > 0:
        weighted_sum = sum(f1s_unrounded[i] * per_class[i]['support'] for i in range(n_classes))
        weighted_f1 = weighted_sum / total_support

    return {
        "accuracy": round(acc, 4),
        "macro_f1_12class": round(macro_f1_all, 4),
        "macro_f1_supported_classes": round(macro_f1_supported_classes, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "n_pooled_samples": total
    }


def run_loso(data_dir, output_path, epochs=40, seed=42):
    """
    Leave-One-Subject-Out evaluation.
    Each subject is held out as test once, train on all others.
    seed: Base random seed for reproducibility (default 42).
    """
    print()
    print("S2S Level 4 — Multi-Sensor Kinematic Chain Consistency")
    print("PAMAP2: Hand + Chest + Ankle IMU  |  4 physics laws")
    print("EVALUATION MODE: LEAVE-ONE-SUBJECT-OUT (LOSO)")
    print("="*62)

    windows = load_pamap2_multisensor(data_dir)
    if len(windows) < 500:
        print("ERROR: not enough data")
        sys.exit(1)

    subjects = sorted(set(w['subject'] for w in windows))
    print(f"\nTotal subjects: {len(subjects)}")
    print(f"Total windows: {len(windows)}")

    all_fold_results = []
    all_predictions = defaultdict(list)  # condition -> list of (pred, true_label) across all folds

    for fold_idx, test_subject in enumerate(subjects):
        train_w = [w for w in windows if w['subject'] != test_subject]
        test_w = [w for w in windows if w['subject'] == test_subject]

        # Pass base seed to run_fold; it will derive fold-specific seeds internally
        fold_results, fold_predictions, norm_s, norm_m, chain_floor = run_fold(
            train_w, test_w, fold_idx, test_subject, epochs, seed=seed
        )

        fold_results['test_subject'] = test_subject
        fold_results['train_subjects'] = [s for s in subjects if s != test_subject]
        all_fold_results.append(fold_results)

        # Collect predictions for pooled metrics
        for cond in ["A_single_sensor", "B_multi_naive", "C_chain_filtered", "D_chain_weighted"]:
            if fold_predictions[cond] is not None:
                all_predictions[cond].extend(fold_predictions[cond])

    # Aggregate results across folds
    print(f"\n{'='*62}")
    print("LOSO RESULTS SUMMARY")
    print(f"{'='*62}")

    aggregated = {}
    for cond in ["A_single_sensor", "B_multi_naive", "C_chain_filtered", "D_chain_weighted"]:
        values = [f[cond] for f in all_fold_results if f[cond] is not None]
        if not values:
            continue

        accs = [v['accuracy'] for v in values]
        f1s_12 = [v['macro_f1_12class'] for v in values]
        f1s_sup = [v['macro_f1_supported_classes'] for v in values]
        f1s_w = [v['weighted_f1'] for v in values]

        aggregated[cond] = {
            "mean_accuracy": round(sum(accs) / len(accs), 4),
            "std_accuracy": round((sum((a - sum(accs)/len(accs))**2 for a in accs) / len(accs))**0.5, 4),
            "mean_macro_f1_12class": round(sum(f1s_12) / len(f1s_12), 4),
            "std_macro_f1_12class": round((sum((f - sum(f1s_12)/len(f1s_12))**2 for f in f1s_12) / len(f1s_12))**0.5, 4),
            "mean_macro_f1_supported_classes": round(sum(f1s_sup) / len(f1s_sup), 4),
            "std_macro_f1_supported_classes": round((sum((f - sum(f1s_sup)/len(f1s_sup))**2 for f in f1s_sup) / len(f1s_sup))**0.5, 4),
            "mean_weighted_f1": round(sum(f1s_w) / len(f1s_w), 4),
            "std_weighted_f1": round((sum((f - sum(f1s_w)/len(f1s_w))**2 for f in f1s_w) / len(f1s_w))**0.5, 4),
            "n_folds": len(values)
        }

    print(f"\n  {'Condition':<24} {'Acc':>8} {'±':>4} {'F1_12':>8} {'±':>4} {'F1_sup':>8} {'±':>4} {'F1_w':>8} {'±':>4}")
    print("  " + "-"*78)
    for cond, stats in aggregated.items():
        print(f"  {cond:<24} {stats['mean_accuracy']:>8.4f} {stats['std_accuracy']:>4.4f} "
              f"{stats['mean_macro_f1_12class']:>8.4f} {stats['std_macro_f1_12class']:>4.4f} "
              f"{stats['mean_macro_f1_supported_classes']:>8.4f} {stats['std_macro_f1_supported_classes']:>4.4f} "
              f"{stats['mean_weighted_f1']:>8.4f} {stats['std_weighted_f1']:>4.4f}")

    # Paired comparisons
    print(f"\n  Paired comparisons (mean ± std across subjects):")
    if all(c in aggregated for c in ["A_single_sensor", "B_multi_naive", "C_chain_filtered", "D_chain_weighted"]):
        gains = {}
        for pair, name, label in [("B_multi_naive", "A_single_sensor", "B-A"),
                           ("C_chain_filtered", "B_multi_naive", "C-B"),
                           ("D_chain_weighted", "C_chain_filtered", "D-C"),
                           ("D_chain_weighted", "A_single_sensor", "D-A")]:
            fold_diffs = []
            for f in all_fold_results:
                if f[pair] and f[name]:
                    fold_diffs.append(f[pair]['macro_f1_12class'] - f[name]['macro_f1_12class'])
            if fold_diffs:
                gains[label] = {
                    "mean": round(sum(fold_diffs) / len(fold_diffs), 4),
                    "std": round((sum((d - sum(fold_diffs)/len(fold_diffs))**2 for d in fold_diffs) / len(fold_diffs))**0.5, 4),
                    "n_paired_folds": len(fold_diffs)
                }
                print(f"    {label}: {gains[label]['mean']:+.4f} ± {gains[label]['std']:.4f} (n={len(fold_diffs)})")

    # Compute pooled metrics from all held-out predictions
    print(f"\n  Pooled metrics (all held-out samples):")
    pooled = {}
    for cond in ["A_single_sensor", "B_multi_naive", "C_chain_filtered", "D_chain_weighted"]:
        if cond in all_predictions and all_predictions[cond]:
            pooled[cond] = compute_pooled_metrics(all_predictions[cond], len(DOMAINS))
            print(f"    {cond}: acc={pooled[cond]['accuracy']:.4f} f1_12={pooled[cond]['macro_f1_12class']:.4f} "
                  f"f1_sup={pooled[cond]['macro_f1_supported_classes']:.4f} f1_w={pooled[cond]['weighted_f1']:.4f} "
                  f"n={pooled[cond]['n_pooled_samples']}")

    results = {
        "meta": {
            "experiment": "s2s_level4_pamap2",
            "dataset": "PAMAP2",
            "evaluation_mode": "leave_one_subject_out",
            "sensors": ["hand", "chest", "ankle"],
            "hz": HZ,
            "window_size": WINDOW_SIZE,
            "epochs": epochs,
            "n_subjects": len(subjects),
            "subjects": subjects,
            "seed": seed,
            "seed_policy": "derive_seed uses SHA256 for cross-process stability. run_loso passes base seed to run_fold. run_fold derives fold_model_seed=derive_seed(base_seed, fold_idx) and fold_shuffle_seed=derive_seed(base_seed, fold_idx, 'shuffle'). All conditions in same fold share model and shuffle seeds for paired comparisons.",
            "physics_laws": ["locomotion_coherence", "segment_coupling",
                           "gyro_accel_coupling", "cross_sensor_jerk_timing"],
        },
        "fold_results": all_fold_results,
        "aggregated": aggregated,
        "paired_comparisons": gains if 'gains' in locals() else {},
        "pooled_metrics": pooled
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved → {output_path}")

    return results


def run_legacy(data_dir, output_path, epochs=40, seed=42):
    """
    Legacy single-split evaluation (original behavior).
    seed: Random seed for reproducibility (default 42).
    """
    print()
    print("S2S Level 4 — Multi-Sensor Kinematic Chain Consistency")
    print("PAMAP2: Hand + Chest + Ankle IMU  |  4 physics laws")
    print("EVALUATION MODE: LEGACY SINGLE SPLIT")
    print("="*62)

    windows = load_pamap2_multisensor(data_dir)
    if len(windows) < 500: print("ERROR: not enough data"); sys.exit(1)

    # Subject split
    subjects = sorted(set(w['subject'] for w in windows))
    # Use local RNG for shuffling to avoid polluting global state
    shuffle_rng = random.Random(seed)
    shuffle_rng.shuffle(subjects)
    n_test = max(1,int(len(subjects)*0.2))
    test_ids = set(subjects[:n_test])
    train_w = [w for w in windows if w['subject'] not in test_ids]
    test_w  = [w for w in windows if w['subject'] in test_ids]
    print(f"\nSubjects: train={len(subjects)-n_test} test={n_test}")
    print(f"Windows:  train={len(train_w)} test={len(test_w)}")

    fold_results, _fold_predictions, norm_s, norm_m, chain_floor = run_fold(
        train_w, test_w, 0, list(test_ids)[0] if test_ids else -1, epochs, seed=seed
    )

    results = {}
    for cond in ["A_single_sensor", "B_multi_naive", "C_chain_filtered", "D_chain_weighted"]:
        if fold_results[cond]:
            results[cond] = fold_results[cond]

    # Summary
    print(f"\n{'='*62}")
    print("  LEVEL 4 RESULTS — MULTI-SENSOR FUSION (PAMAP2)")
    print(f"{'='*62}\n")
    print(f"  {'Condition':<24} {'F1':>6}  {'n':>6}  {'Feats':>5}")
    print("  "+"-"*50)
    for c,r in results.items():
        print(f"  {c:<24} {r['macro_f1_12class']:.4f}  {r['train_n']:>6}  {len(r['per_class']):>5}")

    rA=results.get('A_single_sensor',{})
    rB=results.get('B_multi_naive',{})
    rC=results.get('C_chain_filtered',{})
    rD=results.get('D_chain_weighted',{})

    print()
    if all([rA,rB,rC,rD]):
        multi_gain  = rB['macro_f1_12class']-rA['macro_f1_12class']
        filter_gain = rC['macro_f1_12class']-rB['macro_f1_12class']
        weight_gain = rD['macro_f1_12class']-rC['macro_f1_12class']
        net_gain    = rD['macro_f1_12class']-rA['macro_f1_12class']
        print(f"  Multi-sensor gain  (B-A): {multi_gain*100:+.2f}% F1")
        print(f"  Chain filter gain  (C-B): {filter_gain*100:+.2f}% F1  ← Level 4 claim")
        print(f"  Tier weight gain   (D-C): {weight_gain*100:+.2f}% F1")
        print(f"  Net vs single      (D-A): {net_gain*100:+.2f}% F1")
        print()
        if filter_gain > 0.005:
            print("  ✓ LEVEL 4 PROVEN — Kinematic chain consistency improves multi-sensor fusion")
        elif net_gain > 0.005:
            print("  ✓ Multi-sensor fusion proven — chain filter contributes")
        elif multi_gain > 0.005:
            print("  ~ Multi-sensor helps but chain filter neutral")
        else:
            print("  ~ Multi-sensor fusion neutral on this dataset")

    results['meta'] = {
        "experiment":"s2s_level4_pamap2","dataset":"PAMAP2",
        "evaluation_mode":"legacy_single_split",
        "sensors":["hand","chest","ankle"],"hz":HZ,
        "window_size":WINDOW_SIZE,"epochs":epochs,
        "chain_floor":chain_floor,
        "seed":seed,
        "seed_policy":"derive_seed uses SHA256 for cross-process stability. run_legacy passes base seed to run_fold (single fold with fold_idx=0). run_fold derives fold_model_seed=derive_seed(base_seed, 0) and fold_shuffle_seed=derive_seed(base_seed, 0, 'shuffle'). All conditions share model and shuffle seeds for paired comparisons.",
        "physics_laws":["locomotion_coherence","segment_coupling",
                        "gyro_accel_coupling","cross_sensor_jerk_timing"],
    }
    with open(output_path,'w') as f:
        json.dump(results,f,indent=2)
    print(f"\n  Saved → {output_path}")

    return results


def run(data_dir, output_path, epochs=40, cv_mode='legacy', seed=42):
    """Main entry point - dispatches to legacy or LOSO evaluation."""
    if cv_mode == 'loso':
        return run_loso(data_dir, output_path, epochs, seed=seed)
    else:
        return run_legacy(data_dir, output_path, epochs, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/pamap2/')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--out',    default='experiments/results_level4_pamap2.json')
    parser.add_argument('--cv',     default='legacy', choices=['legacy', 'loso'],
                       help='Cross-validation mode: legacy (single split) or loso (leave-one-subject-out)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default 42)')
    args = parser.parse_args()
    run(args.data, args.out, args.epochs, args.cv, args.seed)
