#!/usr/bin/env python3
"""
S2S Level 2 — Curriculum Training on PAMAP2
Real 100Hz chest IMU  |  9 subjects  |  12 activities → 3 domains

Conditions:
  A — Clean baseline (all windows, no tier weighting)
  B — Corrupted 35% (flat, clip, label noise)
  C — S2S adaptive floor (REJECTED removed)
  D — Tier-weighted loss (GOLD=1.0, SILVER=0.8, BRONZE=0.4)
  E — Curriculum phases (GOLD→+SILVER→+BRONZE)

Run from ~/S2S:
  python3 experiments/level2_pamap2_curriculum.py \
    --data data/pamap2/ --epochs 40 \
    --out experiments/results_level2_pamap2.json
"""

import os, sys, json, math, random, time, argparse
from collections import defaultdict, Counter

# ── Activity mapping ──────────────────────────────────────────────
ACTIVITY_LABELS = {
    1:'REST', 2:'REST', 3:'REST',
    4:'LOCOMOTION', 5:'LOCOMOTION', 6:'LOCOMOTION',
    7:'LOCOMOTION', 12:'LOCOMOTION', 13:'LOCOMOTION',
    16:'ACTIVITY', 17:'ACTIVITY', 24:'ACTIVITY',
}
DOMAINS  = ['REST', 'LOCOMOTION', 'ACTIVITY']
DOM2IDX  = {d:i for i,d in enumerate(DOMAINS)}

COL_ACTIVITY = 1
COL_CHEST_AX = 21
COL_CHEST_AY = 22
COL_CHEST_AZ = 23

WINDOW_SIZE   = 256
STEP_SIZE     = 128
HZ            = 100
CORRUPT_RATE  = 0.35
TIER_WEIGHTS  = {'GOLD':1.0, 'SILVER':0.8, 'BRONZE':0.4, 'REJECTED':0.0}


# ══════════════════════════════════════════════════════════════════
# PHYSICS SCORER
# ══════════════════════════════════════════════════════════════════

def score_window(accel):
    n = len(accel)
    if n < 20: return 0
    ax = [accel[i][0] for i in range(n)]
    ay = [accel[i][1] for i in range(n)]
    az = [accel[i][2] for i in range(n)]
    mean_x = sum(ax)/n; mean_y = sum(ay)/n; mean_z = sum(az)/n
    var_x = sum((v-mean_x)**2 for v in ax)/n
    var_y = sum((v-mean_y)**2 for v in ay)/n
    var_z = sum((v-mean_z)**2 for v in az)/n
    total_var = var_x+var_y+var_z
    if total_var < 0.01: return 0
    for axis_vals in [ax,ay,az]:
        max_val = max(abs(v) for v in axis_vals)
        if max_val < 0.5: continue
        at_max = sum(1 for v in axis_vals if abs(abs(v)-max_val) < 0.01)
        if at_max/n > 0.20: return 0
    dt = 1.0/HZ
    jerks = []
    for axis_vals in [ax,ay,az]:
        vel = [(axis_vals[k+1]-axis_vals[k-1])/(2*dt) for k in range(1,n-1)]
        jerk = [(vel[k+1]-vel[k-1])/(2*dt) for k in range(1,len(vel)-1)]
        jerks.extend([abs(j) for j in jerk])
    if jerks:
        jerks.sort()
        p95 = jerks[int(len(jerks)*0.95)]
        if p95 > 5000: return 0
    scores = []
    if total_var > 10.0:   scores.append(90)
    elif total_var > 2.0:  scores.append(80)
    elif total_var > 0.5:  scores.append(70)
    else:                  scores.append(58)
    if jerks:
        p95 = jerks[int(len(jerks)*0.95)]
        if p95 < 100:   scores.append(90)
        elif p95 < 250: scores.append(80)
        else:           scores.append(65)
    diffs = [abs(ax[i]-ax[i-1]) for i in range(1,n)]
    max_diff = max(diffs) if diffs else 0
    if max_diff < 5.0:    scores.append(85)
    elif max_diff < 15.0: scores.append(70)
    else:                 scores.append(55)
    return int(sum(scores)/len(scores)) if scores else 0


def get_tier(score, floor):
    if score >= 87: return 'GOLD'
    if score >= 75: return 'SILVER'
    if score >= 60: return 'BRONZE'
    if score >= floor: return 'BRONZE'
    return 'REJECTED'


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_pamap2(data_dir):
    print(f"Loading PAMAP2 from {data_dir}...")
    all_windows = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.dat'): continue
        fpath = os.path.join(data_dir, fname)
        if os.path.getsize(fpath) > 400*1024*1024:
            print(f"  Skipping {fname} (corrupt)")
            continue
        rows = []
        with open(fpath) as f:
            for line in f:
                cols = line.strip().split()
                if len(cols) < 24: continue
                try:
                    act = int(float(cols[COL_ACTIVITY]))
                    if act not in ACTIVITY_LABELS: continue
                    ax,ay,az = float(cols[COL_CHEST_AX]),float(cols[COL_CHEST_AY]),float(cols[COL_CHEST_AZ])
                    if any(math.isnan(v) for v in [ax,ay,az]): continue
                    rows.append((act,[ax,ay,az]))
                except: continue
        subj = int(''.join(filter(str.isdigit, fname.split('.')[0])))
        wins = 0
        i = 0
        while i+WINDOW_SIZE <= len(rows):
            acts = [rows[i+j][0] for j in range(WINDOW_SIZE)]
            dom_act = max(set(acts), key=acts.count)
            if acts.count(dom_act)/WINDOW_SIZE < 0.8: i+=STEP_SIZE; continue
            accel = [rows[i+j][1] for j in range(WINDOW_SIZE)]
            all_windows.append({'subject':subj,'activity':dom_act,
                                 'domain':ACTIVITY_LABELS[dom_act],'accel':accel})
            wins+=1; i+=STEP_SIZE
        print(f"  {fname}: {len(rows)} rows → {wins} windows")
    print(f"  Total: {len(all_windows)} windows")
    return all_windows


# ══════════════════════════════════════════════════════════════════
# CORRUPTION
# ══════════════════════════════════════════════════════════════════

def inject_corruption(windows, rate=CORRUPT_RATE):
    import copy
    random.seed(123)
    n = int(len(windows)*rate)
    corrupt_idx = set(random.sample(range(len(windows)), n))
    counts = {'flat':0,'clip':0,'label':0}
    result = []
    for i,w in enumerate(windows):
        if i not in corrupt_idx: result.append(w); continue
        r = copy.deepcopy(w)
        ctype = i%3
        if ctype == 0:
            r['accel'] = [[random.gauss(0,0.001)]*3 for _ in range(len(r['accel']))]
            counts['flat']+=1
        elif ctype == 1:
            clip = 16.0
            r['accel'] = [[max(-clip,min(clip,v)) for v in row] for row in r['accel']]
            counts['clip']+=1
        else:
            acts = [a for a in ACTIVITY_LABELS if a != r['activity']]
            if acts:
                new_act = random.choice(acts)
                r['activity']=new_act; r['domain']=ACTIVITY_LABELS[new_act]
            counts['label']+=1
        result.append(r)
    print(f"  Injected {sum(counts.values())} ({rate*100:.0f}%): {counts}")
    return result


# ══════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION + NORMALIZATION
# ══════════════════════════════════════════════════════════════════

def extract_features(accel):
    n = len(accel)
    if n < 20: return None
    feats = []
    for axis in range(3):
        vals = [accel[i][axis] for i in range(n)]
        mean = sum(vals)/n
        std = math.sqrt(sum((v-mean)**2 for v in vals)/n)
        rng = max(vals)-min(vals)
        feats += [mean,std,rng]
    mags = [math.sqrt(sum(accel[i][j]**2 for j in range(3))) for i in range(n)]
    mag_mean = sum(mags)/n
    mag_std = math.sqrt(sum((v-mag_mean)**2 for v in mags)/n)
    mag_max = max(mags)
    feats += [mag_mean,mag_std,mag_max]
    N = n
    fft_mag = []
    for k in range(1,min(N//2,15)):
        re = sum(mags[i]*math.cos(2*math.pi*k*i/N) for i in range(N))
        im = sum(mags[i]*math.sin(2*math.pi*k*i/N) for i in range(N))
        fft_mag.append(math.sqrt(re**2+im**2))
    dom_freq = fft_mag.index(max(fft_mag))+1 if fft_mag else 0
    feats.append(dom_freq)
    raw_x = [accel[i][0] for i in range(n)]
    mean_x = sum(raw_x)/n
    dx = [v-mean_x for v in raw_x]
    zcr = sum(1 for i in range(1,n) if dx[i]*dx[i-1]<0)/n
    feats.append(zcr)
    return feats


def normalize(X, stats=None):
    if not X: return [],stats
    nf = len(X[0])
    if stats is None:
        means = [sum(x[i] for x in X)/len(X) for i in range(nf)]
        stds = [math.sqrt(sum((x[i]-means[i])**2 for x in X)/len(X))+1e-8 for i in range(nf)]
        stats = (means,stds)
    means,stds = stats
    return [[(x[i]-means[i])/stds[i] for i in range(nf)] for x in X],stats


# ══════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════

class MLP:
    def __init__(self, input_dim, hidden, output, lr=0.001):
        self.lr=lr
        random.seed(42)
        s1=math.sqrt(2.0/input_dim); s2=math.sqrt(2.0/hidden)
        self.W1=[[random.gauss(0,s1) for _ in range(input_dim)] for _ in range(hidden)]
        self.b1=[0.0]*hidden
        self.W2=[[random.gauss(0,s2) for _ in range(hidden)] for _ in range(output)]
        self.b2=[0.0]*output

    def forward(self,x):
        h=[max(0,sum(self.W1[j][i]*x[i] for i in range(len(x)))+self.b1[j]) for j in range(len(self.W1))]
        logits=[sum(self.W2[k][j]*h[j] for j in range(len(h)))+self.b2[k] for k in range(len(self.W2))]
        mx=max(logits); exp_l=[math.exp(min(l-mx,60)) for l in logits]; s=sum(exp_l)
        return h,[e/s for e in exp_l]

    def backward(self,x,h,probs,label):
        d2=list(probs); d2[label]-=1.0
        for k in range(len(self.W2)):
            for j in range(len(h)): self.W2[k][j]-=self.lr*d2[k]*h[j]
            self.b2[k]-=self.lr*d2[k]
        d1=[sum(self.W2[k][j]*d2[k] for k in range(len(self.W2)))*(1 if h[j]>0 else 0) for j in range(len(self.W1))]
        for j in range(len(self.W1)):
            for i in range(len(x)): self.W1[j][i]-=self.lr*d1[j]*x[i]
            self.b1[j]-=self.lr*d1[j]

    def _clip(self,max_norm=5.0):
        for row in self.W1+self.W2:
            n=math.sqrt(sum(v**2 for v in row))
            if n>max_norm:
                for i in range(len(row)): row[i]*=max_norm/n

    def train_epoch(self,samples):
        random.shuffle(samples)
        loss,skips=0.0,0
        for item in samples:
            feat,label,w=item if len(item)==3 else (*item,1.0)
            h,probs=self.forward(feat)
            sl=-math.log(max(probs[label],1e-10))
            if sl!=sl: skips+=1; continue
            loss+=sl*w; orig=self.lr; self.lr*=w
            self.backward(feat,h,probs,label); self.lr=orig
        self._clip()
        return loss/max(len(samples)-skips,1)

    def evaluate(self,samples):
        correct=0; pc=defaultdict(lambda:[0,0])
        for feat,label in samples:
            _,probs=self.forward(feat); pred=probs.index(max(probs))
            pc[label][1]+=1
            if pred==label: correct+=1; pc[label][0]+=1
        acc=correct/max(len(samples),1)
        f1s=[pc[i][0]/max(pc[i][1],1) for i in range(len(DOMAINS))]
        return acc,sum(f1s)/len(f1s)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run(data_dir, output_path, epochs=40):
    print()
    print("S2S Level 2 — Curriculum Training (PAMAP2)")
    print("Real 100Hz chest IMU  |  3 domains  |  Cross-dataset validation")
    print("="*62)

    all_windows = load_pamap2(data_dir)
    if len(all_windows) < 500: print("ERROR: not enough data"); sys.exit(1)

    subjects = sorted(set(w['subject'] for w in all_windows))
    random.seed(42); random.shuffle(subjects)
    n_test = max(1,int(len(subjects)*0.2))
    test_ids = set(subjects[:n_test])
    train_w = [w for w in all_windows if w['subject'] not in test_ids]
    test_w  = [w for w in all_windows if w['subject'] in test_ids]
    print(f"\nSubjects: train={len(subjects)-n_test} test={n_test}")
    print(f"Windows:  train={len(train_w)} test={len(test_w)}")

    # Score clean train
    print("\nScoring clean train windows...")
    scores = [score_window(w['accel']) for w in train_w]
    nonzero = sorted(s for s in scores if s > 0)
    floor = nonzero[int(len(nonzero)*0.25)] if nonzero else 60
    print(f"  Score avg (non-zero): {sum(nonzero)/max(len(nonzero),1):.1f}")
    print(f"  Adaptive floor (p25): {floor}")

    # Assign tiers
    tier_counts = Counter()
    for i,w in enumerate(train_w):
        t = get_tier(scores[i], floor)
        w['tier'] = t; w['score'] = scores[i]
        tier_counts[t]+=1
    for t in ['GOLD','SILVER','BRONZE','REJECTED']:
        print(f"    {t:<10} {tier_counts[t]}")

    # Normalization
    train_feats = [extract_features(w['accel']) for w in train_w]
    valid_feats = [f for f in train_feats if f]
    _,norm_stats = normalize(valid_feats)
    def norm(fl):
        fn,_=normalize(fl,norm_stats); return fn

    # Test samples
    test_feats = [extract_features(w['accel']) for w in test_w]
    test_labels = [DOM2IDX[w['domain']] for w in test_w]
    tv = [(f,l) for f,l in zip(test_feats,test_labels) if f]
    test_eval = list(zip(norm([f for f,_ in tv]),[l for _,l in tv]))
    print(f"Test samples: {len(test_eval)}")

    # Corrupt train
    print(f"\nInjecting corruption ({CORRUPT_RATE*100:.0f}%)...")
    train_corrupted = inject_corruption(train_w)

    # Re-score corrupted, apply floor
    print("Applying S2S adaptive floor to corrupted...")
    floor_kept = [w for w in train_corrupted if score_window(w['accel']) >= floor]
    print(f"  Kept {len(floor_kept)}/{len(train_corrupted)} (removed {len(train_corrupted)-len(floor_kept)})")

    # Build condition samples
    def make_samples(windows, use_tier_weight=False, w_default=1.0):
        out = []
        for w in windows:
            f = extract_features(w['accel'])
            if f and w['domain'] in DOM2IDX:
                wt = TIER_WEIGHTS.get(w.get('tier','SILVER'), w_default) if use_tier_weight else w_default
                if wt > 0:
                    out.append((f, DOM2IDX[w['domain']], wt))
        return out

    sA = make_samples(train_w)
    sB = make_samples(train_corrupted)
    sC = make_samples(floor_kept)
    sD = make_samples(floor_kept, use_tier_weight=True)

    # Curriculum: phase 1=GOLD, phase 2=GOLD+SILVER, phase 3=all
    gold_w    = [w for w in floor_kept if w.get('tier')=='GOLD']
    silver_w  = [w for w in floor_kept if w.get('tier') in ('GOLD','SILVER')]
    sE_p1 = make_samples(gold_w)
    sE_p2 = make_samples(silver_w)
    sE_p3 = make_samples(floor_kept)

    def norm_s(samples):
        if not samples: return []
        feats=[f for f,_,_ in samples]
        fn=norm(feats)
        return [(fn[i],samples[i][1],samples[i][2]) for i in range(len(fn))]

    sA_n=norm_s(sA); sB_n=norm_s(sB); sC_n=norm_s(sC); sD_n=norm_s(sD)
    sE_p1_n=norm_s(sE_p1); sE_p2_n=norm_s(sE_p2); sE_p3_n=norm_s(sE_p3)

    print(f"\nCondition sizes:")
    print(f"  A clean:      {len(sA_n)}")
    print(f"  B corrupted:  {len(sB_n)}")
    print(f"  C floor:      {len(sC_n)}")
    print(f"  D weighted:   {len(sD_n)}")
    print(f"  E curriculum: p1={len(sE_p1_n)} p2={len(sE_p2_n)} p3={len(sE_p3_n)}")

    ep1 = max(1, int(epochs*0.3))
    ep2 = max(1, int(epochs*0.3))
    ep3 = epochs - ep1 - ep2

    results = {}
    conditions = [
        ("A_clean",      sA_n,  None,   None,   "Clean baseline"),
        ("B_corrupted",  sB_n,  None,   None,   f"Corrupted {CORRUPT_RATE*100:.0f}%"),
        ("C_s2s_floor",  sC_n,  None,   None,   "Adaptive S2S floor"),
        ("D_weighted",   sD_n,  None,   None,   "Tier-weighted loss"),
        ("E_curriculum", None,  sE_p1_n,sE_p2_n,"Curriculum: GOLD→SILVER→ALL"),
    ]

    for cond, train_data, p1, p2, desc in conditions:
        print(f"\n{'─'*62}")
        print(f"Condition {cond}  |  {desc}")
        model = MLP(input_dim=14, hidden=64, output=len(DOMAINS), lr=0.001)
        t0 = time.time()

        if cond == "E_curriculum":
            print(f"  Phase 1 GOLD only      ({ep1} epochs, n={len(p1)})")
            for ep in range(ep1):
                loss = model.train_epoch(p1)
                if (ep+1) % 10 == 0:
                    acc,f1 = model.evaluate(test_eval)
                    print(f"  P1 Epoch {ep+1:3d}  loss={loss:.4f}  f1={f1:.3f}")

            print(f"  Phase 2 GOLD+SILVER    ({ep2} epochs, n={len(p2)})")
            for ep in range(ep2):
                loss = model.train_epoch(p2)
                if (ep+1) % 10 == 0:
                    acc,f1 = model.evaluate(test_eval)
                    print(f"  P2 Epoch {ep+1:3d}  loss={loss:.4f}  f1={f1:.3f}")

            print(f"  Phase 3 ALL tiers      ({ep3} epochs, n={len(sE_p3_n)})")
            for ep in range(ep3):
                loss = model.train_epoch(sE_p3_n)
                if (ep+1) % 10 == 0:
                    acc,f1 = model.evaluate(test_eval)
                    print(f"  P3 Epoch {ep+1:3d}  loss={loss:.4f}  f1={f1:.3f}")
        else:
            for ep in range(epochs):
                loss = model.train_epoch(train_data)
                if (ep+1) % 10 == 0:
                    acc,f1 = model.evaluate(test_eval)
                    print(f"  Epoch {ep+1:3d}/{epochs}  loss={loss:.4f}  acc={acc:.3f}  f1={f1:.3f}")

        acc,f1 = model.evaluate(test_eval)
        elapsed = time.time()-t0
        print(f"  Final: {acc:.4f} ({acc*100:.1f}%)  F1={f1:.4f}  [{elapsed:.0f}s]")
        results[cond] = {"description":desc,"accuracy":round(acc,4),
                         "macro_f1":round(f1,4),"train_n":len(train_data or sE_p3_n)}

    # Summary
    print(f"\n{'='*62}")
    print("  LEVEL 2 RESULTS — PAMAP2 (Cross-dataset validation)")
    print(f"{'='*62}\n")
    print(f"  {'Condition':<22} {'F1':>6}  {'n':>6}  Description")
    print("  "+"-"*55)
    for c,r in results.items():
        print(f"  {c:<22} {r['macro_f1']:.4f}  {r['train_n']:>6}  {r['description']}")

    rA=results.get('A_clean',{}); rB=results.get('B_corrupted',{})
    rC=results.get('C_s2s_floor',{}); rD=results.get('D_weighted',{})
    rE=results.get('E_curriculum',{})

    print()
    if all([rA,rB,rC,rD,rE]):
        damage  = rB['macro_f1']-rA['macro_f1']
        recover = rC['macro_f1']-rB['macro_f1']
        weight  = rD['macro_f1']-rC['macro_f1']
        curric  = rE['macro_f1']-rA['macro_f1']
        print(f"  Corruption damage  (B-A): {damage*100:+.2f}% F1")
        print(f"  Floor recovery     (C-B): {recover*100:+.2f}% F1")
        print(f"  Tier weights gain  (D-C): {weight*100:+.2f}% F1")
        print(f"  Curriculum vs clean(E-A): {curric*100:+.2f}% F1  ← Level 2 claim")
        print()
        if curric > 0.005:
            print("  ✓ LEVEL 2 PROVEN ON PAMAP2 — Curriculum beats clean baseline")
        elif curric > -0.005:
            print("  ~ Neutral — curriculum matches clean baseline")
        else:
            print("  ✗ Curriculum did not improve on PAMAP2")

    results['meta'] = {
        "experiment":"s2s_level2_pamap2","dataset":"PAMAP2",
        "hz":HZ,"window_size":WINDOW_SIZE,"epochs":epochs,
        "corrupt_rate":CORRUPT_RATE,"adaptive_floor":floor,
        "tier_counts":dict(tier_counts),
    }
    with open(output_path,'w') as f:
        json.dump(results,f,indent=2)
    print(f"\n  Saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/pamap2/')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--out',    default='experiments/results_level2_pamap2.json')
    args = parser.parse_args()
    run(args.data, args.out, args.epochs)
