#!/usr/bin/env python3
"""
S2S Level 3 — Adaptive Reconstruction
Frequency-aware repair of BRONZE sensor windows.

Reconstruction methods chosen by sample rate:
  Hz ≤ 50:  Kalman RTS smoother       (current proven method)
  Hz ≥ 51:  Savitzky-Golay filter     (preserves peaks at high Hz)

Dual acceptance criterion (both must pass):
  1. Physics re-score ≥ 75 (SILVER threshold)
  2. Spectral similarity ≥ 0.8 (important freq bands preserved)

Repaired records carry full provenance:
  tier:           RECONSTRUCTED
  recon_method:   kalman | savgol
  score_before:   original BRONZE score
  score_after:    post-reconstruction score
  spectral_sim:   spectral similarity 0-1
  weight:         0.5 (half trust — it was repaired)

Tests:
  WISDM 20Hz  → Kalman path    (previously +0.65%)
  PAMAP2 100Hz → Savitzky path (previously -0.54% with Kalman)

Run from ~/S2S:
  python3 experiments/level3_adaptive_reconstruction.py \
    --dataset wisdm --data data/ --epochs 40 \
    --out experiments/results_level3_adaptive_wisdm.json

  python3 experiments/level3_adaptive_reconstruction.py \
    --dataset pamap2 --data data/pamap2/ --epochs 40 \
    --out experiments/results_level3_adaptive_pamap2.json
"""

import os, sys, json, math, random, time, argparse
from collections import defaultdict, Counter

# ══════════════════════════════════════════════════════════════════
# SAVITZKY-GOLAY FILTER (pure python, no scipy)
# Fits polynomial to local window, uses fitted value as smoothed
# ══════════════════════════════════════════════════════════════════

def savgol_smooth(signal, window_size, polyorder=2):
    """
    Savitzky-Golay smoothing via local polynomial fitting.
    Robust pure-Python implementation — fits poly to each window.
    """
    if window_size % 2 == 0:
        window_size += 1
    window_size = max(polyorder+2, min(window_size, len(signal)))
    if window_size % 2 == 0:
        window_size -= 1
    half = window_size // 2
    n = len(signal)
    result = list(signal)

    for i in range(half, n-half):
        # Extract local window
        xs = list(range(-half, half+1))
        ys = signal[i-half:i+half+1]
        m = len(xs)

        # Fit polynomial of degree polyorder via normal equations
        # Build A matrix [1, x, x^2, ...]
        deg = min(polyorder, m-1)
        A = [[xs[j]**k for k in range(deg+1)] for j in range(m)]

        # AtA and Aty
        AtA = [[sum(A[r][c1]*A[r][c2] for r in range(m))
                for c2 in range(deg+1)] for c1 in range(deg+1)]
        Aty = [sum(A[r][c]*ys[r] for r in range(m)) for c in range(deg+1)]

        # Solve via Cholesky-free Gaussian elimination with pivoting
        d = deg+1
        aug = [AtA[i][:] + [Aty[i]] for i in range(d)]
        ok = True
        for col in range(d):
            pivot = max(range(col, d), key=lambda r: abs(aug[r][col]))
            aug[col], aug[pivot] = aug[pivot], aug[col]
            if abs(aug[col][col]) < 1e-10:
                ok = False; break
            for row in range(d):
                if row == col: continue
                f = aug[row][col] / aug[col][col]
                aug[row] = [aug[row][j] - f*aug[col][j] for j in range(d+1)]

        if not ok:
            # Fallback: simple moving average
            result[i] = sum(ys)/len(ys)
        else:
            coeffs = [aug[i][d]/aug[i][i] for i in range(d)]
            # Evaluate polynomial at x=0 (center point)
            result[i] = coeffs[0]

    return result


# ══════════════════════════════════════════════════════════════════
# KALMAN RTS SMOOTHER (proven at 20Hz)
# ══════════════════════════════════════════════════════════════════

def kalman_smooth(signal, process_var=1e-3, obs_var=0.1):
    """1D Kalman forward-backward smoother."""
    n = len(signal)
    x_f = [0.0]*n; p_f = [1.0]*n
    # Forward pass
    x_f[0] = signal[0]; p_f[0] = 1.0
    for i in range(1, n):
        x_pred = x_f[i-1]
        p_pred = p_f[i-1] + process_var
        k = p_pred / (p_pred + obs_var)
        x_f[i] = x_pred + k*(signal[i]-x_pred)
        p_f[i] = (1-k)*p_pred
    # Backward pass
    x_s = list(x_f); p_s = list(p_f)
    for i in range(n-2, -1, -1):
        g = p_f[i] / (p_f[i] + process_var)
        x_s[i] = x_f[i] + g*(x_s[i+1]-x_f[i])
        p_s[i] = p_f[i] + g**2*(p_s[i+1]-p_f[i])
    return x_s


# ══════════════════════════════════════════════════════════════════
# ADAPTIVE RECONSTRUCTION — core of Level 3 v2
# ══════════════════════════════════════════════════════════════════

def spectral_similarity(original, reconstructed):
    """
    Compare dominant frequency content 0-1.
    Uses DFT magnitude correlation on first 20 harmonics.
    Preserving important frequency bands = motion preserved.
    """
    n = min(len(original), len(reconstructed), 64)
    def dft_mags(sig):
        mean = sum(sig)/n
        s = [v-mean for v in sig[:n]]
        mags = []
        for k in range(1, min(20, n//2)):
            re = sum(s[i]*math.cos(2*math.pi*k*i/n) for i in range(n))
            im = sum(s[i]*math.sin(2*math.pi*k*i/n) for i in range(n))
            mags.append(math.sqrt(re**2+im**2))
        return mags
    mo = dft_mags(original)
    mr = dft_mags(reconstructed)
    if not mo or not mr: return 0.0
    # Normalized cross-correlation
    mean_o = sum(mo)/len(mo); mean_r = sum(mr)/len(mr)
    num = sum((mo[i]-mean_o)*(mr[i]-mean_r) for i in range(len(mo)))
    do = math.sqrt(sum((v-mean_o)**2 for v in mo))
    dr = math.sqrt(sum((v-mean_r)**2 for v in mr))
    if do < 1e-10 or dr < 1e-10: return 0.0
    return max(0.0, min(1.0, (num/(do*dr)+1)/2))  # map -1..1 → 0..1


def adaptive_reconstruct(accel, hz, score_fn):
    """
    Reconstruct a BRONZE window using frequency-appropriate method.
    Returns: (reconstructed_accel, provenance_dict) or (None, reason)

    Provenance dict contains full repair audit trail.
    """
    n = len(accel)
    score_before = score_fn(accel)

    # Choose method by Hz
    if hz <= 50:
        method = 'kalman'
        recon = []
        for axis in range(3):
            axis_vals = [accel[i][axis] for i in range(n)]
            smoothed = kalman_smooth(axis_vals)
            recon.append(smoothed)
        recon_accel = [[recon[0][i], recon[1][i], recon[2][i]] for i in range(n)]
    else:
        method = 'savgol'
        # Window = 5% of signal, min 5 samples
        window = max(5, int(0.05 * hz))
        if window % 2 == 0: window += 1
        recon = []
        for axis in range(3):
            axis_vals = [accel[i][axis] for i in range(n)]
            smoothed = savgol_smooth(axis_vals, window_size=window, polyorder=2)
            recon.append(smoothed)
        recon_accel = [[recon[0][i], recon[1][i], recon[2][i]] for i in range(n)]

    score_after = score_fn(recon_accel)

    # Spectral similarity check — per axis, take average
    sim_scores = []
    for axis in range(3):
        orig_axis  = [accel[i][axis] for i in range(n)]
        recon_axis = [recon_accel[i][axis] for i in range(n)]
        sim_scores.append(spectral_similarity(orig_axis, recon_axis))
    spec_sim = sum(sim_scores)/len(sim_scores)

    provenance = {
        'recon_method':  method,
        'hz':            hz,
        'score_before':  score_before,
        'score_after':   score_after,
        'spectral_sim':  round(spec_sim, 3),
        'weight':        0.5,
    }

    # Dual acceptance: physics ≥75 AND spectral similarity ≥0.8
    if score_after >= 75 and spec_sim >= 0.8:
        provenance['tier'] = 'RECONSTRUCTED'
        provenance['accepted'] = True
        return recon_accel, provenance
    else:
        provenance['tier'] = 'REJECTED_RECON'
        provenance['accepted'] = False
        reason = []
        if score_after < 75:  reason.append(f"score={score_after}<75")
        if spec_sim < 0.8:    reason.append(f"sim={spec_sim:.2f}<0.8")
        provenance['reject_reason'] = ', '.join(reason)
        return None, provenance


# ══════════════════════════════════════════════════════════════════
# PHYSICS SCORERS
# ══════════════════════════════════════════════════════════════════

def score_window_20hz(accel):
    n = len(accel)
    if n < 20: return 0
    ax=[accel[i][0] for i in range(n)]
    ay=[accel[i][1] for i in range(n)]
    az=[accel[i][2] for i in range(n)]
    mx=sum(ax)/n; my=sum(ay)/n; mz=sum(az)/n
    tv=sum((v-mx)**2 for v in ax)/n+sum((v-my)**2 for v in ay)/n+sum((v-mz)**2 for v in az)/n
    if tv < 0.005: return 0
    scores = []
    if tv > 5.0:   scores.append(85)
    elif tv > 1.0: scores.append(75)
    elif tv > 0.2: scores.append(65)
    else:          scores.append(55)
    # ZCR on demeaned x
    dx = [v-mx for v in ax]
    zcr = sum(1 for i in range(1,n) if dx[i]*dx[i-1]<0)/n
    if zcr > 0.1:   scores.append(85)
    elif zcr > 0.05: scores.append(70)
    else:            scores.append(55)
    return int(sum(scores)/len(scores))


def score_window_100hz(accel):
    n = len(accel)
    if n < 20: return 0
    ax=[accel[i][0] for i in range(n)]
    ay=[accel[i][1] for i in range(n)]
    az=[accel[i][2] for i in range(n)]
    mx=sum(ax)/n; my=sum(ay)/n; mz=sum(az)/n
    tv=sum((v-mx)**2 for v in ax)/n+sum((v-my)**2 for v in ay)/n+sum((v-mz)**2 for v in az)/n
    if tv < 0.01: return 0
    for axis_vals in [ax,ay,az]:
        max_val=max(abs(v) for v in axis_vals)
        if max_val < 0.5: continue
        at_max=sum(1 for v in axis_vals if abs(abs(v)-max_val)<0.01)
        if at_max/n > 0.20: return 0
    dt=1.0/100; jerks=[]
    for av in [ax,ay,az]:
        vel=[(av[k+1]-av[k-1])/(2*dt) for k in range(1,n-1)]
        jerk=[(vel[k+1]-vel[k-1])/(2*dt) for k in range(1,len(vel)-1)]
        jerks.extend([abs(j) for j in jerk])
    if jerks:
        jerks.sort()
        if jerks[int(len(jerks)*0.95)] > 5000: return 0
    scores=[]
    if tv>10: scores.append(90)
    elif tv>2: scores.append(80)
    elif tv>0.5: scores.append(70)
    else: scores.append(58)
    return int(sum(scores)/len(scores)) if scores else 0


# ══════════════════════════════════════════════════════════════════
# DATA LOADERS
# ══════════════════════════════════════════════════════════════════

def load_wisdm(data_dir):
    """Load WISDM 20Hz accelerometer data."""
    print("Loading WISDM (20Hz)...")
    DOMAIN_MAP = {
        'A':'LOCOMOTION','B':'LOCOMOTION','C':'REST',
        'D':'ACTIVITY','E':'ACTIVITY','F':'ACTIVITY',
        'G':'ACTIVITY','H':'ACTIVITY','I':'ACTIVITY',
        'J':'ACTIVITY','K':'ACTIVITY','L':'ACTIVITY',
        'M':'ACTIVITY','O':'REST','P':'ACTIVITY',
        'Q':'ACTIVITY','R':'ACTIVITY','S':'ACTIVITY',
    }
    DOMAINS = sorted(set(DOMAIN_MAP.values()))
    windows = []
    WINDOW_SIZE = 200; STEP = 100
    raw_file = None
    # WISDM 2019 format: one file per subject
    # data_XXXX_accel_phone.txt: subject,activity,timestamp,ax,ay,az;
    accel_dir = os.path.join(data_dir,'wisdm','raw','phone','accel')
    if not os.path.isdir(accel_dir):
        accel_dir = os.path.join(data_dir,'raw','phone','accel')
    if not os.path.isdir(accel_dir):
        print("ERROR: WISDM accel dir not found")
        return None, None, None

    by_subj = defaultdict(list)
    for fname in sorted(os.listdir(accel_dir)):
        if not fname.endswith('.txt'): continue
        fpath = os.path.join(accel_dir, fname)
        with open(fpath) as f:
            for line in f:
                line = line.strip().rstrip(';')
                parts = line.split(',')
                if len(parts) < 6: continue
                try:
                    uid=int(parts[0]); act=parts[1].strip()
                    ax,ay,az=float(parts[3]),float(parts[4]),float(parts[5])
                    if any(math.isnan(v) for v in [ax,ay,az]): continue
                    if act not in DOMAIN_MAP: continue
                    by_subj[uid].append((act,[ax,ay,az]))
                except: continue

    for uid,rows in sorted(by_subj.items()):
        i=0
        while i+WINDOW_SIZE<=len(rows):
            acts=[rows[i+j][0] for j in range(WINDOW_SIZE)]
            dom_act=max(set(acts),key=acts.count)
            if acts.count(dom_act)/WINDOW_SIZE<0.8: i+=STEP; continue
            accel=[rows[i+j][1] for j in range(WINDOW_SIZE)]
            windows.append({'subject':uid,'activity':dom_act,
                           'domain':DOMAIN_MAP[dom_act],'accel':accel,'hz':20})
            i+=STEP

    print(f"  {len(windows)} windows from {len(by_subj)} subjects")
    return windows, DOMAINS, 20


PAMAP2_ACTS = {
    1:'lying',2:'sitting',3:'standing',4:'walking',5:'running',
    6:'cycling',7:'nordic_walking',12:'ascending_stairs',
    13:'descending_stairs',16:'vacuum_cleaning',17:'ironing',24:'rope_jumping'
}

def load_pamap2(data_dir):
    """Load PAMAP2 100Hz 3-domain data."""
    print("Loading PAMAP2 (100Hz)...")
    DOMAIN_MAP = {
        1:'REST',2:'REST',3:'REST',
        4:'LOCOMOTION',5:'LOCOMOTION',6:'LOCOMOTION',
        7:'LOCOMOTION',12:'LOCOMOTION',13:'LOCOMOTION',
        16:'ACTIVITY',17:'ACTIVITY',24:'ACTIVITY',
    }
    DOMAINS=['REST','LOCOMOTION','ACTIVITY']
    windows=[]; WINDOW_SIZE=256; STEP=128
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.dat'): continue
        fpath=os.path.join(data_dir,fname)
        if os.path.getsize(fpath)>400*1024*1024:
            print(f"  Skipping {fname}"); continue
        subj=int(''.join(filter(str.isdigit,fname.split('.')[0])))
        rows=[]
        with open(fpath) as f:
            for line in f:
                cols=line.strip().split()
                if len(cols)<24: continue
                try:
                    act=int(float(cols[1]))
                    if act not in DOMAIN_MAP: continue
                    ax,ay,az=float(cols[21]),float(cols[22]),float(cols[23])
                    if any(math.isnan(v) for v in [ax,ay,az]): continue
                    rows.append((act,[ax,ay,az]))
                except: continue
        i=0
        while i+WINDOW_SIZE<=len(rows):
            acts=[rows[i+j][0] for j in range(WINDOW_SIZE)]
            dom_act=max(set(acts),key=acts.count)
            if acts.count(dom_act)/WINDOW_SIZE<0.8: i+=STEP; continue
            accel=[rows[i+j][1] for j in range(WINDOW_SIZE)]
            windows.append({'subject':subj,'activity':dom_act,
                           'domain':DOMAIN_MAP[dom_act],'accel':accel,'hz':100})
            i+=STEP
    print(f"  {len(windows)} windows")
    return windows, DOMAINS, 100


# ══════════════════════════════════════════════════════════════════
# FEATURES + MLP
# ══════════════════════════════════════════════════════════════════

def extract_features(accel):
    n=len(accel)
    if n<20: return None
    feats=[]
    for axis in range(3):
        vals=[accel[i][axis] for i in range(n)]
        mean=sum(vals)/n
        std=math.sqrt(sum((v-mean)**2 for v in vals)/n)
        feats+=[mean,std,max(vals)-min(vals)]
    mags=[math.sqrt(sum(accel[i][j]**2 for j in range(3))) for i in range(n)]
    mag_mean=sum(mags)/n
    mag_std=math.sqrt(sum((v-mag_mean)**2 for v in mags)/n)
    feats+=[mag_mean,mag_std,max(mags)]
    fft_mag=[]
    for k in range(1,min(n//2,15)):
        re=sum(mags[i]*math.cos(2*math.pi*k*i/n) for i in range(n))
        im=sum(mags[i]*math.sin(2*math.pi*k*i/n) for i in range(n))
        fft_mag.append(math.sqrt(re**2+im**2))
    feats.append(fft_mag.index(max(fft_mag))+1 if fft_mag else 0)
    dx=[accel[i][0]-sum(accel[j][0] for j in range(n))/n for i in range(n)]
    feats.append(sum(1 for i in range(1,n) if dx[i]*dx[i-1]<0)/n)
    return feats


def normalize(X, stats=None):
    if not X: return [],stats
    nf=len(X[0])
    if stats is None:
        means=[sum(x[i] for x in X)/len(X) for i in range(nf)]
        stds=[math.sqrt(sum((x[i]-means[i])**2 for x in X)/len(X))+1e-8 for i in range(nf)]
        stats=(means,stds)
    means,stds=stats
    return [[(x[i]-means[i])/stds[i] for i in range(nf)] for x in X],stats


class MLP:
    def __init__(self,input_dim,hidden,output,lr=0.001):
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
        ndom=max(pc.keys())+1 if pc else 1
        return acc,sum(pc[i][0]/max(pc[i][1],1) for i in range(ndom))/ndom


# ══════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════

def run(dataset, data_dir, output_path, epochs=40):
    print()
    print(f"S2S Level 3 — Adaptive Reconstruction ({dataset.upper()})")
    print("Kalman ≤50Hz  |  Savitzky-Golay ≥51Hz  |  Dual acceptance")
    print("="*62)

    # Load data
    if dataset == 'wisdm':
        windows, DOMAINS, HZ = load_wisdm(data_dir)
        score_fn = score_window_20hz
    else:
        windows, DOMAINS, HZ = load_pamap2(data_dir)
        score_fn = score_window_100hz

    if windows is None or len(windows) < 200:
        print("ERROR: insufficient data"); sys.exit(1)

    DOM2IDX = {d:i for i,d in enumerate(DOMAINS)}
    method_name = 'kalman' if HZ <= 50 else 'savgol'
    print(f"\nHz={HZ}  Method={method_name}  Domains={len(DOMAINS)}")

    # Subject split
    subjects=sorted(set(w['subject'] for w in windows))
    random.seed(42); random.shuffle(subjects)
    n_test=max(1,int(len(subjects)*0.2))
    test_ids=set(subjects[:n_test])
    train_w=[w for w in windows if w['subject'] not in test_ids]
    test_w =[w for w in windows if w['subject'] in test_ids]
    print(f"Subjects: train={len(subjects)-n_test} test={n_test}")
    print(f"Windows:  train={len(train_w)} test={len(test_w)}")

    # Score clean train, establish floor
    print("\nScoring clean train windows...")
    scores=[score_fn(w['accel']) for w in train_w]
    nonzero=sorted(s for s in scores if s>0)
    floor=nonzero[int(len(nonzero)*0.25)] if nonzero else 60
    print(f"  Score avg (non-zero): {sum(nonzero)/max(len(nonzero),1):.1f}")
    print(f"  Adaptive floor (p25): {floor}")

    for i,w in enumerate(train_w):
        s=scores[i]
        if s>=87: w['tier']='GOLD'
        elif s>=75: w['tier']='SILVER'
        elif s>=floor: w['tier']='BRONZE'
        else: w['tier']='REJECTED'
        w['score']=s
    tc=Counter(w['tier'] for w in train_w)
    for t in ['GOLD','SILVER','BRONZE','REJECTED']:
        print(f"    {t:<10} {tc.get(t,0)}")

    # Normalization stats from clean
    clean_feats=[extract_features(w['accel']) for w in train_w]
    _,norm_stats=normalize([f for f in clean_feats if f])
    def norm(fl):
        fn,_=normalize(fl,norm_stats); return fn

    # Test set
    test_feats=[extract_features(w['accel']) for w in test_w]
    test_labels=[DOM2IDX[w['domain']] for w in test_w]
    tv=[(f,l) for f,l in zip(test_feats,test_labels) if f]
    test_eval=list(zip(norm([f for f,_ in tv]),[l for _,l in tv]))
    print(f"Test samples: {len(test_eval)}")

    # Inject corruption (degrade 30% of SILVER to BRONZE)
    import copy
    print("\nDegrading 30% of SILVER windows to BRONZE...")
    silver_idx=[i for i,w in enumerate(train_w) if w['tier']=='SILVER']
    random.seed(42)
    degrade_idx=set(random.sample(silver_idx, int(len(silver_idx)*0.30)))
    train_mixed=[]
    degraded=0
    for i,w in enumerate(train_w):
        wc=copy.deepcopy(w)
        if i in degrade_idx:
            noise_std=0.15*math.sqrt(sum(
                (wc['accel'][j][k]-sum(wc['accel'][m][k] for m in range(len(wc['accel'])))/len(wc['accel']))**2
                for j in range(len(wc['accel'])) for k in range(3)
            )/(len(wc['accel'])*3))
            wc['accel']=[[v+random.gauss(0,noise_std) for v in row] for row in wc['accel']]
            wc['tier']='BRONZE'; wc['score']=score_fn(wc['accel'])
            degraded+=1
        train_mixed.append(wc)
    print(f"  Degraded {degraded} SILVER→BRONZE")

    # CONDITION A: all SILVER (clean reference)
    silver_w=[w for w in train_mixed if w['tier']=='SILVER']
    # CONDITION B: SILVER + BRONZE (mixed, no repair)
    accepted_w=[w for w in train_mixed if w['tier'] in ('SILVER','BRONZE')]
    # CONDITION C: floor filter (drop BRONZE, keep SILVER only)
    floor_w=[w for w in train_mixed if w['tier']=='SILVER']
    # CONDITION D: adaptive reconstruction of BRONZE
    print(f"\nAdaptive reconstruction of BRONZE windows (method={method_name})...")
    bronze_w=[w for w in train_mixed if w['tier']=='BRONZE']
    accepted_recon=[]; rejected_recon=[]
    prov_log=[]
    for w in bronze_w:
        recon_accel, prov = adaptive_reconstruct(w['accel'], HZ, score_fn)
        prov_log.append(prov)
        if recon_accel is not None:
            wc=copy.deepcopy(w)
            wc['accel']=recon_accel
            wc['tier']='RECONSTRUCTED'
            wc['recon_method']=prov['recon_method']
            wc['score_before']=prov['score_before']
            wc['score_after']=prov['score_after']
            wc['spectral_sim']=prov['spectral_sim']
            wc['weight']=0.5
            accepted_recon.append(wc)
        else:
            rejected_recon.append(w)

    accept_rate=len(accepted_recon)/max(len(bronze_w),1)*100
    avg_sim=sum(p['spectral_sim'] for p in prov_log if p['accepted'])/max(len(accepted_recon),1)
    avg_score_before=sum(p['score_before'] for p in prov_log)/max(len(prov_log),1)
    avg_score_after=sum(p['score_after'] for p in prov_log if p['accepted'])/max(len(accepted_recon),1)

    print(f"  BRONZE windows:          {len(bronze_w)}")
    print(f"  Accepted (dual check):   {len(accepted_recon)} ({accept_rate:.1f}%)")
    print(f"  Rejected:                {len(rejected_recon)}")
    print(f"  Avg score before:        {avg_score_before:.1f}")
    print(f"  Avg score after:         {avg_score_after:.1f}")
    print(f"  Avg spectral similarity: {avg_sim:.3f}")
    print(f"  Method used:             {method_name}")

    # Show provenance sample
    print(f"\n  Provenance sample (first 3 accepted):")
    shown=0
    for p in prov_log:
        if p['accepted'] and shown<3:
            print(f"    method={p['recon_method']}  "
                  f"score {p['score_before']}→{p['score_after']}  "
                  f"sim={p['spectral_sim']:.3f}  "
                  f"tier=RECONSTRUCTED  weight=0.5")
            shown+=1

    recon_train=floor_w+accepted_recon

    def make_samples(windows, default_weight=1.0):
        out=[]
        for w in windows:
            f=extract_features(w['accel'])
            if f and w['domain'] in DOM2IDX:
                wt=w.get('weight',default_weight)
                out.append((f,DOM2IDX[w['domain']],wt))
        return out

    sA=make_samples(silver_w)
    sB=make_samples(accepted_w)
    sC=make_samples(floor_w)
    sD=make_samples(recon_train)

    def ns(samples):
        if not samples: return []
        fn=norm([f for f,_,_ in samples])
        return [(fn[i],samples[i][1],samples[i][2]) for i in range(len(fn))]

    sA_n=ns(sA); sB_n=ns(sB); sC_n=ns(sC); sD_n=ns(sD)

    print(f"\nCondition sizes:")
    print(f"  A (clean SILVER):         {len(sA_n)}")
    print(f"  B (SILVER+BRONZE mixed):  {len(sB_n)}")
    print(f"  C (floor, SILVER only):   {len(sC_n)}")
    print(f"  D (floor+recon w=0.5):    {len(sD_n)}  (+{len(accepted_recon)} RECONSTRUCTED)")

    results={}
    for cond, train_data, desc in [
        ("A_clean",      sA_n, "Clean SILVER reference"),
        ("B_mixed",      sB_n, "SILVER+BRONZE, no repair"),
        ("C_floor",      sC_n, "Floor only — BRONZE removed"),
        ("D_adaptive",   sD_n, f"Floor + {method_name} recon (w=0.5, dual check)"),
    ]:
        print(f"\n{'─'*62}")
        print(f"Condition {cond}  n={len(train_data)}")
        model=MLP(input_dim=14,hidden=64,output=len(DOMAINS),lr=0.001)
        t0=time.time()
        for ep in range(epochs):
            loss=model.train_epoch(train_data)
            if (ep+1)%10==0:
                acc,f1=model.evaluate(test_eval)
                print(f"  Epoch {ep+1:3d}/{epochs}  loss={loss:.4f}  acc={acc:.3f}  f1={f1:.3f}")
        acc,f1=model.evaluate(test_eval)
        print(f"  Final: {acc:.4f} ({acc*100:.1f}%)  F1={f1:.4f}  [{time.time()-t0:.0f}s]")
        results[cond]={"description":desc,"accuracy":round(acc,4),
                       "macro_f1":round(f1,4),"train_n":len(train_data)}

    # Summary
    print(f"\n{'='*62}")
    print(f"  LEVEL 3 ADAPTIVE RESULTS — {dataset.upper()} ({HZ}Hz)")
    print(f"  Reconstruction method: {method_name.upper()}")
    print(f"{'='*62}\n")
    print(f"  {'Condition':<22} {'F1':>6}  {'n':>6}")
    print("  "+"-"*38)
    for c,r in results.items():
        print(f"  {c:<22} {r['macro_f1']:.4f}  {r['train_n']:>6}")

    rA=results.get('A_clean',{}); rB=results.get('B_mixed',{})
    rC=results.get('C_floor',{}); rD=results.get('D_adaptive',{})
    if all([rA,rB,rC,rD]):
        damage  = rB['macro_f1']-rA['macro_f1']
        recover = rC['macro_f1']-rB['macro_f1']
        gain    = rD['macro_f1']-rC['macro_f1']
        net     = rD['macro_f1']-rA['macro_f1']
        print(f"\n  Degradation damage (B-A): {damage*100:+.2f}% F1")
        print(f"  Floor recovery     (C-B): {recover*100:+.2f}% F1")
        print(f"  Recon gain         (D-C): {gain*100:+.2f}% F1  ← Level 3 claim")
        print(f"  Net vs clean       (D-A): {net*100:+.2f}% F1")
        print()
        if gain > 0.003:
            print(f"  ✓ LEVEL 3 PROVEN on {dataset.upper()} — {method_name.upper()} reconstruction +{gain*100:.2f}%")
        elif gain > -0.003:
            print(f"  ~ Neutral — reconstruction matches floor")
        else:
            print(f"  ✗ Reconstruction did not improve on {dataset.upper()}")

    results['meta']={
        "experiment":f"s2s_level3_adaptive_{dataset}",
        "dataset":dataset,"hz":HZ,
        "recon_method":method_name,
        "dual_acceptance":{"physics_threshold":75,"spectral_similarity_threshold":0.8},
        "bronze_windows":len(bronze_w),
        "accepted_reconstructed":len(accepted_recon),
        "acceptance_rate":round(accept_rate,1),
        "avg_score_before":round(avg_score_before,1),
        "avg_score_after":round(avg_score_after,1),
        "avg_spectral_similarity":round(avg_sim,3),
        "adaptive_floor":floor,
    }
    with open(output_path,'w') as f:
        json.dump(results,f,indent=2)
    print(f"\n  Saved → {output_path}")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['wisdm','pamap2'], required=True)
    parser.add_argument('--data',    default='data/')
    parser.add_argument('--epochs',  type=int, default=40)
    parser.add_argument('--out',     default='experiments/results_level3_adaptive.json')
    args=parser.parse_args()
    run(args.dataset, args.data, args.out, args.epochs)
