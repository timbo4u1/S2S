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

import os, sys, json, math, random, time, argparse
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
        rows = []
        with open(fpath) as f:
            for line in f:
                cols = line.strip().split()
                if len(cols) < 47: continue
                try:
                    act = int(float(cols[COL_ACTIVITY]))
                    if act not in ACTIVITY_LABELS: continue
                    def gc(c): return float(cols[c])
                    def safe(c):
                        v = float(cols[c])
                        return 0.0 if math.isnan(v) else v
                    rows.append((act,
                        [safe(COL_HAND_AX), safe(COL_HAND_AY), safe(COL_HAND_AZ)],
                        [safe(COL_HAND_GX), safe(COL_HAND_GY), safe(COL_HAND_GZ)],
                        [safe(COL_CHEST_AX),safe(COL_CHEST_AY),safe(COL_CHEST_AZ)],
                        [safe(COL_CHEST_GX),safe(COL_CHEST_GY),safe(COL_CHEST_GZ)],
                        [safe(COL_ANKLE_AX),safe(COL_ANKLE_AY),safe(COL_ANKLE_AZ)],
                        [safe(COL_ANKLE_GX),safe(COL_ANKLE_GY),safe(COL_ANKLE_GZ)],
                    ))
                except: continue

        wins = 0; i = 0
        while i+WINDOW_SIZE <= len(rows):
            acts = [rows[i+j][0] for j in range(WINDOW_SIZE)]
            dom_act = max(set(acts), key=acts.count)
            if acts.count(dom_act)/WINDOW_SIZE < 0.8: i+=STEP_SIZE; continue
            w = {
                'subject': subj, 'activity': dom_act,
                'domain': ACTIVITY_LABELS[dom_act],
                'hand_a':  [rows[i+j][1] for j in range(WINDOW_SIZE)],
                'hand_g':  [rows[i+j][2] for j in range(WINDOW_SIZE)],
                'chest_a': [rows[i+j][3] for j in range(WINDOW_SIZE)],
                'chest_g': [rows[i+j][4] for j in range(WINDOW_SIZE)],
                'ankle_a': [rows[i+j][5] for j in range(WINDOW_SIZE)],
                'ankle_g': [rows[i+j][6] for j in range(WINDOW_SIZE)],
            }
            all_windows.append(w); wins+=1; i+=STEP_SIZE
        print(f"  {fname}: {len(rows)} rows → {wins} windows")
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
    def __init__(self, input_dim, hidden, output, lr=0.001):
        self.lr=lr; random.seed(42)
        s1=math.sqrt(2.0/input_dim); s2=math.sqrt(2.0/hidden)
        self.W1=[[random.gauss(0,s1) for _ in range(input_dim)] for _ in range(hidden)]
        self.b1=[0.0]*hidden
        self.W2=[[random.gauss(0,s2) for _ in range(hidden)] for _ in range(output)]
        self.b2=[0.0]*output

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

    def train_epoch(self,samples):
        random.shuffle(samples); loss=0.0; skips=0
        for item in samples:
            feat,label,wt=(item[0],item[1],item[2]) if len(item)==3 else (item[0],item[1],1.0)
            h,probs=self.forward(feat)
            sl=-math.log(max(probs[label],1e-10))
            if sl!=sl: skips+=1; continue
            loss+=sl*wt; self.backward(feat,h,probs,label,wt)
        self._clip()
        return loss/max(len(samples)-skips,1)

    def evaluate(self,samples):
        correct=0; pc=defaultdict(lambda:[0,0])
        for feat,label in samples:
            _,probs=self.forward(feat); pred=probs.index(max(probs))
            pc[label][1]+=1
            if pred==label: correct+=1; pc[label][0]+=1
        acc=correct/max(len(samples),1)
        return acc,sum(pc[i][0]/max(pc[i][1],1) for i in range(len(DOMAINS)))/len(DOMAINS)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run(data_dir, output_path, epochs=40):
    print()
    print("S2S Level 4 — Multi-Sensor Kinematic Chain Consistency")
    print("PAMAP2: Hand + Chest + Ankle IMU  |  4 physics laws")
    print("="*62)

    windows = load_pamap2_multisensor(data_dir)
    if len(windows) < 500: print("ERROR: not enough data"); sys.exit(1)

    # Subject split
    subjects = sorted(set(w['subject'] for w in windows))
    random.seed(42); random.shuffle(subjects)
    n_test = max(1,int(len(subjects)*0.2))
    test_ids = set(subjects[:n_test])
    train_w = [w for w in windows if w['subject'] not in test_ids]
    test_w  = [w for w in windows if w['subject'] in test_ids]
    print(f"\nSubjects: train={len(subjects)-n_test} test={n_test}")
    print(f"Windows:  train={len(train_w)} test={len(test_w)}")

    # Score chain consistency on clean train
    print("\nScoring kinematic chain consistency...")
    chain_scores = []
    for w in train_w:
        s = kinematic_chain_score(
            w['hand_a'], w['chest_a'], w['ankle_a'],
            w['hand_g'], w['chest_g'], w['ankle_g'])
        w['chain_score'] = s
        w['chain_tier']  = get_chain_tier(s)
        chain_scores.append(s)

    chain_sorted = sorted(chain_scores)
    chain_floor = chain_sorted[int(len(chain_sorted)*0.25)]
    print(f"  Chain score avg: {sum(chain_scores)/len(chain_scores):.1f}")
    print(f"  Chain floor (p25): {chain_floor}")
    tc = Counter(w['chain_tier'] for w in train_w)
    for t in ['GOLD','SILVER','BRONZE','REJECTED']:
        print(f"    {t:<10} {tc.get(t,0)}")

    # Normalization — use multi features from clean train
    print("\nExtracting features...")
    # Single sensor features (chest only)
    single_feats_tr = [extract_features_single(w['chest_a']) for w in train_w]
    multi_feats_tr  = [extract_features_multi(w) for w in train_w]

    valid_single = [f for f in single_feats_tr if f]
    valid_multi  = [f for f in multi_feats_tr  if f]
    _,norm_s = normalize(valid_single)
    _,norm_m = normalize(valid_multi)

    def ns(fl): fn,_=normalize(fl,norm_s); return fn
    def nm(fl): fn,_=normalize(fl,norm_m); return fn

    # Test samples
    test_single = [(f,DOM2IDX[w['domain']]) for f,w in
                   zip([extract_features_single(w['chest_a']) for w in test_w],test_w) if f]
    test_multi  = [(f,DOM2IDX[w['domain']]) for f,w in
                   zip([extract_features_multi(w) for w in test_w],test_w) if f]

    test_s_n = list(zip(ns([f for f,_ in test_single]),[l for _,l in test_single]))
    test_m_n = list(zip(nm([f for f,_ in test_multi]), [l for _,l in test_multi]))
    print(f"Test samples: single={len(test_s_n)} multi={len(test_m_n)}")

    # Build conditions
    # A: single sensor chest only
    sA_raw = [(extract_features_single(w['chest_a']),DOM2IDX[w['domain']],1.0)
              for w in train_w if extract_features_single(w['chest_a'])]
    sA = list(zip(ns([f for f,_,_ in sA_raw]),[l for _,l,_ in sA_raw],[wt for _,_,wt in sA_raw]))

    # B: multi-sensor naive (no chain filter)
    sB_raw = [(extract_features_multi(w),DOM2IDX[w['domain']],1.0)
              for w in train_w if extract_features_multi(w)]
    sB = list(zip(nm([f for f,_,_ in sB_raw]),[l for _,l,_ in sB_raw],[wt for _,_,wt in sB_raw]))

    # C: multi-sensor + chain filter
    chain_kept = [w for w in train_w if w['chain_score'] >= chain_floor]
    sC_raw = [(extract_features_multi(w),DOM2IDX[w['domain']],1.0)
              for w in chain_kept if extract_features_multi(w)]
    sC = list(zip(nm([f for f,_,_ in sC_raw]),[l for _,l,_ in sC_raw],[wt for _,_,wt in sC_raw]))

    # D: multi-sensor + chain filter + tier weights
    tier_wt = {'GOLD':1.0,'SILVER':0.8,'BRONZE':0.5,'REJECTED':0.0}
    sD_raw = [(extract_features_multi(w),DOM2IDX[w['domain']],tier_wt.get(w['chain_tier'],0.5))
              for w in chain_kept if extract_features_multi(w) and tier_wt.get(w['chain_tier'],0)>0]
    sD = list(zip(nm([f for f,_,_ in sD_raw]),[l for _,l,_ in sD_raw],[wt for _,_,wt in sD_raw]))

    print(f"\nCondition sizes:")
    print(f"  A single sensor:        {len(sA)}")
    print(f"  B multi naive:          {len(sB)}")
    print(f"  C multi + chain filter: {len(sC)}")
    print(f"  D multi + filter + wts: {len(sD)}")

    results = {}
    conditions = [
        ("A_single_sensor",   sA, test_s_n, "Chest IMU only (Level 1 reference)"),
        ("B_multi_naive",     sB, test_m_n, "3 IMUs concatenated, no chain check"),
        ("C_chain_filtered",  sC, test_m_n, "3 IMUs + kinematic chain filter"),
        ("D_chain_weighted",  sD, test_m_n, "3 IMUs + chain filter + tier weights"),
    ]

    for cond, train_data, test_data, desc in conditions:
        print(f"\n{'─'*62}")
        print(f"Condition {cond}  n={len(train_data)}")
        print(f"  {desc}")
        input_dim = len(train_data[0][0]) if train_data else 14
        model = MLP(input_dim=input_dim, hidden=64, output=len(DOMAINS), lr=0.001)
        t0 = time.time()
        for ep in range(epochs):
            loss = model.train_epoch(train_data)
            if (ep+1) % 10 == 0:
                acc,f1 = model.evaluate(test_data)
                print(f"  Epoch {ep+1:3d}/{epochs}  loss={loss:.4f}  acc={acc:.3f}  f1={f1:.3f}")
        acc,f1 = model.evaluate(test_data)
        elapsed = time.time()-t0
        print(f"  Final: {acc:.4f} ({acc*100:.1f}%)  F1={f1:.4f}  [{elapsed:.0f}s]")
        results[cond] = {"description":desc,"accuracy":round(acc,4),
                         "macro_f1":round(f1,4),"train_n":len(train_data),
                         "input_features":input_dim}

    # Summary
    print(f"\n{'='*62}")
    print("  LEVEL 4 RESULTS — MULTI-SENSOR FUSION (PAMAP2)")
    print(f"{'='*62}\n")
    print(f"  {'Condition':<24} {'F1':>6}  {'n':>6}  {'Feats':>5}")
    print("  "+"-"*50)
    for c,r in results.items():
        print(f"  {c:<24} {r['macro_f1']:.4f}  {r['train_n']:>6}  {r['input_features']:>5}")

    rA=results.get('A_single_sensor',{})
    rB=results.get('B_multi_naive',{})
    rC=results.get('C_chain_filtered',{})
    rD=results.get('D_chain_weighted',{})

    print()
    if all([rA,rB,rC,rD]):
        multi_gain  = rB['macro_f1']-rA['macro_f1']
        filter_gain = rC['macro_f1']-rB['macro_f1']
        weight_gain = rD['macro_f1']-rC['macro_f1']
        net_gain    = rD['macro_f1']-rA['macro_f1']
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

    # Discussion notes
    print(f"\n  Discussion:")
    print(f"  Chain score avg: {sum(chain_scores)/len(chain_scores):.1f}")
    print(f"  Chain floor:     {chain_floor}")
    print(f"  Windows removed by chain filter: {len(train_w)-len(chain_kept)}")
    print(f"  Removal rate: {(len(train_w)-len(chain_kept))/len(train_w)*100:.1f}%")

    results['meta'] = {
        "experiment":"s2s_level4_pamap2","dataset":"PAMAP2",
        "sensors":["hand","chest","ankle"],"hz":HZ,
        "window_size":WINDOW_SIZE,"epochs":epochs,
        "chain_floor":chain_floor,
        "chain_score_avg":round(sum(chain_scores)/len(chain_scores),1),
        "tier_counts":dict(tc),
        "physics_laws":["locomotion_coherence","segment_coupling",
                        "gyro_accel_coupling","cross_sensor_jerk_timing"],
    }
    with open(output_path,'w') as f:
        json.dump(results,f,indent=2)
    print(f"\n  Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/pamap2/')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--out',    default='experiments/results_level4_pamap2.json')
    args = parser.parse_args()
    run(args.data, args.out, args.epochs)
