#!/usr/bin/env python3
"""
S2S Level 2 — Adaptive Tier Boundaries (PAMAP2)
Fixes the GOLD=0 problem from level2_pamap2_curriculum.py

Root cause of original failure:
  - level2_pamap2_curriculum.py used a multi-component scorer
    that averaged several sub-scores, dragging all scores to ~72.5
  - This scorer uses clean variance-based scoring matching actual
    PAMAP2 signal characteristics → scores cluster at 58/70/80/90

Adaptive tier boundaries (percentile-anchored):
  - GOLD   = p75 of clean non-zero scores = 90  (39% of windows)
  - SILVER = p50                          = 80  (20% of windows)
  - floor  = p25                          = 58  (reject below)
  - BRONZE = floor ≤ score < SILVER            (41% of windows)

Conditions:
  A — Clean baseline (all windows, flat weight)
  B — Corrupted 35%
  C — Adaptive floor (REJECTED removed)
  D — Tier-weighted loss (GOLD=1.0, SILVER=0.8, BRONZE=0.4)
  E — Curriculum (GOLD only → GOLD+SILVER → all)

Run from ~/S2S:
  python3 experiments/level2_pamap2_adaptive_tiers.py \
    --data data/pamap2/ --epochs 40 \
    --out experiments/results_level2_pamap2_adaptive.json
"""

import os, sys, json, math, random, time, argparse, copy
from collections import defaultdict, Counter

# ── Activity mapping ───────────────────────────────────────────
ACTIVITY_LABELS = {
    1:'REST',  2:'REST',  3:'REST',
    4:'LOCOMOTION', 5:'LOCOMOTION', 6:'LOCOMOTION',
    7:'LOCOMOTION', 12:'LOCOMOTION', 13:'LOCOMOTION',
    16:'ACTIVITY', 17:'ACTIVITY', 24:'ACTIVITY',
}
DOMAINS  = ['REST','LOCOMOTION','ACTIVITY']
DOM2IDX  = {d:i for i,d in enumerate(DOMAINS)}

COL_ACTIVITY = 1
COL_CHEST_AX = 21; COL_CHEST_AY = 22; COL_CHEST_AZ = 23

WINDOW_SIZE  = 256
STEP_SIZE    = 128
HZ           = 100
CORRUPT_RATE = 0.35

# ── Adaptive tier thresholds (set after scoring clean data) ────
# Will be computed at runtime from p25/p50/p75 of clean scores
# Defaults shown here are PAMAP2-specific (from calibration run)
DEFAULT_GOLD_PCT   = 75   # p75 → GOLD threshold
DEFAULT_SILVER_PCT = 50   # p50 → SILVER threshold
DEFAULT_FLOOR_PCT  = 25   # p25 → floor (reject below)


# ══════════════════════════════════════════════════════════════════
# SCORER — calibrated to PAMAP2 100Hz signal characteristics
# Returns discrete values: 0, 58, 70, 80, 90
# ══════════════════════════════════════════════════════════════════

def score_window(accel):
    n = len(accel)
    if n < 20: return 0
    ax=[accel[i][0] for i in range(n)]
    ay=[accel[i][1] for i in range(n)]
    az=[accel[i][2] for i in range(n)]
    mx=sum(ax)/n; my=sum(ay)/n; mz=sum(az)/n
    tv = (sum((v-mx)**2 for v in ax) +
          sum((v-my)**2 for v in ay) +
          sum((v-mz)**2 for v in az)) / n
    if tv < 0.01: return 0
    # Check for clipping (saturation)
    for axis_vals in [ax, ay, az]:
        max_val = max(abs(v) for v in axis_vals)
        if max_val < 0.5: continue
        at_max = sum(1 for v in axis_vals if abs(abs(v)-max_val) < 0.01)
        if at_max/n > 0.20: return 0
    # Single variance score → discrete 58/70/80/90
    if tv > 10:  return 90
    if tv > 2:   return 80
    if tv > 0.5: return 70
    return 58


def compute_adaptive_tiers(scores_nonzero,
                            gold_pct=DEFAULT_GOLD_PCT,
                            silver_pct=DEFAULT_SILVER_PCT,
                            floor_pct=DEFAULT_FLOOR_PCT):
    """
    Compute tier thresholds from percentiles of clean non-zero scores.
    Returns (gold_threshold, silver_threshold, floor_threshold)
    """
    s = sorted(scores_nonzero)
    n = len(s)
    gold_thresh   = s[int(n * gold_pct   / 100)]
    silver_thresh = s[int(n * silver_pct / 100)]
    floor_thresh  = s[int(n * floor_pct  / 100)]
    return gold_thresh, silver_thresh, floor_thresh


def get_tier(score, gold_t, silver_t, floor_t):
    if score >= gold_t:   return 'GOLD'
    if score >= silver_t: return 'SILVER'
    if score >= floor_t:  return 'BRONZE'
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
            print(f"  Skipping {fname}"); continue
        subj = int(''.join(filter(str.isdigit, fname.split('.')[0])))
        rows = []
        with open(fpath) as f:
            for line in f:
                cols = line.strip().split()
                if len(cols) < 24: continue
                try:
                    act = int(float(cols[COL_ACTIVITY]))
                    if act not in ACTIVITY_LABELS: continue
                    ax,ay,az = (float(cols[COL_CHEST_AX]),
                                float(cols[COL_CHEST_AY]),
                                float(cols[COL_CHEST_AZ]))
                    if any(math.isnan(v) for v in [ax,ay,az]): continue
                    rows.append((act,[ax,ay,az]))
                except: continue
        i = 0
        wins = 0
        while i+WINDOW_SIZE <= len(rows):
            acts = [rows[i+j][0] for j in range(WINDOW_SIZE)]
            dom  = max(set(acts), key=acts.count)
            if acts.count(dom)/WINDOW_SIZE < 0.8: i+=STEP_SIZE; continue
            accel = [rows[i+j][1] for j in range(WINDOW_SIZE)]
            all_windows.append({'subject':subj,'activity':dom,
                                 'domain':ACTIVITY_LABELS[dom],
                                 'accel':accel})
            wins+=1; i+=STEP_SIZE
        print(f"  {fname}: {wins} windows")
    print(f"  Total: {len(all_windows)} windows")
    return all_windows


# ══════════════════════════════════════════════════════════════════
# CORRUPTION
# ══════════════════════════════════════════════════════════════════

def inject_corruption(windows, rate=CORRUPT_RATE):
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
            r['accel'] = [[random.gauss(0,0.001)]*3
                          for _ in range(len(r['accel']))]
            counts['flat']+=1
        elif ctype == 1:
            clip = 16.0
            r['accel'] = [[max(-clip,min(clip,v)) for v in row]
                           for row in r['accel']]
            counts['clip']+=1
        else:
            alts = [a for a in ACTIVITY_LABELS if a != r['activity']]
            if alts:
                new_act = random.choice(alts)
                r['activity']=new_act
                r['domain']=ACTIVITY_LABELS[new_act]
            counts['label']+=1
        result.append(r)
    print(f"  Injected {sum(counts.values())} ({rate*100:.0f}%): {counts}")
    return result


# ══════════════════════════════════════════════════════════════════
# FEATURES + NORMALIZATION
# ══════════════════════════════════════════════════════════════════

def extract_features(accel):
    n = len(accel)
    if n < 20: return None
    feats = []
    for axis in range(3):
        vals = [accel[i][axis] for i in range(n)]
        mean = sum(vals)/n
        std  = math.sqrt(sum((v-mean)**2 for v in vals)/n)
        feats += [mean, std, max(vals)-min(vals)]
    mags = [math.sqrt(sum(accel[i][j]**2 for j in range(3)))
            for i in range(n)]
    mag_mean = sum(mags)/n
    mag_std  = math.sqrt(sum((v-mag_mean)**2 for v in mags)/n)
    feats   += [mag_mean, mag_std, max(mags)]
    fft_mag  = []
    for k in range(1, min(n//2, 15)):
        re = sum(mags[i]*math.cos(2*math.pi*k*i/n) for i in range(n))
        im = sum(mags[i]*math.sin(2*math.pi*k*i/n) for i in range(n))
        fft_mag.append(math.sqrt(re**2+im**2))
    feats.append(fft_mag.index(max(fft_mag))+1 if fft_mag else 0)
    dx = [accel[i][0] - sum(accel[j][0] for j in range(n))/n
          for i in range(n)]
    feats.append(sum(1 for i in range(1,n) if dx[i]*dx[i-1]<0)/n)
    return feats


def normalize(X, stats=None):
    if not X: return [], stats
    nf = len(X[0])
    if stats is None:
        means = [sum(x[i] for x in X)/len(X) for i in range(nf)]
        stds  = [math.sqrt(sum((x[i]-means[i])**2 for x in X)/len(X))+1e-8
                 for i in range(nf)]
        stats = (means, stds)
    means, stds = stats
    return [[(x[i]-means[i])/stds[i] for i in range(nf)] for x in X], stats


# ══════════════════════════════════════════════════════════════════
# MLP
# ══════════════════════════════════════════════════════════════════

class MLP:
    def __init__(self, input_dim, hidden, output, lr=0.001):
        self.lr=lr; random.seed(42)
        s1=math.sqrt(2.0/input_dim); s2=math.sqrt(2.0/hidden)
        self.W1=[[random.gauss(0,s1) for _ in range(input_dim)]
                 for _ in range(hidden)]
        self.b1=[0.0]*hidden
        self.W2=[[random.gauss(0,s2) for _ in range(hidden)]
                 for _ in range(output)]
        self.b2=[0.0]*output

    def forward(self, x):
        h=[max(0,sum(self.W1[j][i]*x[i] for i in range(len(x)))+self.b1[j])
           for j in range(len(self.W1))]
        logits=[sum(self.W2[k][j]*h[j] for j in range(len(h)))+self.b2[k]
                for k in range(len(self.W2))]
        mx=max(logits); el=[math.exp(min(l-mx,60)) for l in logits]
        s=sum(el)
        return h,[e/s for e in el]

    def backward(self, x, h, probs, label, w=1.0):
        d2=list(probs); d2[label]-=1.0
        for k in range(len(self.W2)):
            for j in range(len(h)):
                self.W2[k][j]-=self.lr*w*d2[k]*h[j]
            self.b2[k]-=self.lr*w*d2[k]
        d1=[sum(self.W2[k][j]*d2[k] for k in range(len(self.W2)))*
            (1 if h[j]>0 else 0) for j in range(len(self.W1))]
        for j in range(len(self.W1)):
            for i in range(len(x)):
                self.W1[j][i]-=self.lr*w*d1[j]*x[i]
            self.b1[j]-=self.lr*w*d1[j]

    def _clip(self, mn=5.0):
        for row in self.W1+self.W2:
            n=math.sqrt(sum(v**2 for v in row))
            if n>mn:
                for i in range(len(row)): row[i]*=mn/n

    def train_epoch(self, samples):
        random.shuffle(samples); loss=0.0; skips=0
        for item in samples:
            feat,label,wt = (item[0],item[1],item[2]) if len(item)==3 \
                             else (item[0],item[1],1.0)
            h,probs = self.forward(feat)
            sl = -math.log(max(probs[label],1e-10))
            if sl!=sl: skips+=1; continue
            loss+=sl*wt; self.backward(feat,h,probs,label,wt)
        self._clip()
        return loss/max(len(samples)-skips,1)

    def evaluate(self, samples):
        correct=0; pc=defaultdict(lambda:[0,0])
        for feat,label in samples:
            _,probs=self.forward(feat)
            pred=probs.index(max(probs))
            pc[label][1]+=1
            if pred==label: correct+=1; pc[label][0]+=1
        acc=correct/max(len(samples),1)
        f1s=[pc[i][0]/max(pc[i][1],1) for i in range(len(DOMAINS))]
        return acc, sum(f1s)/len(f1s)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run(data_dir, output_path, epochs=40):
    print()
    print("S2S Level 2 — Adaptive Tier Boundaries (PAMAP2)")
    print("Percentile-anchored GOLD/SILVER/BRONZE thresholds")
    print("="*62)

    all_windows = load_pamap2(data_dir)
    if len(all_windows) < 500:
        print("ERROR: not enough data"); sys.exit(1)

    # Subject split
    subjects = sorted(set(w['subject'] for w in all_windows))
    random.seed(42); random.shuffle(subjects)
    n_test   = max(1, int(len(subjects)*0.2))
    test_ids = set(subjects[:n_test])
    train_w  = [w for w in all_windows if w['subject'] not in test_ids]
    test_w   = [w for w in all_windows if w['subject'] in test_ids]
    print(f"\nSubjects: train={len(subjects)-n_test}  test={n_test}")
    print(f"Windows:  train={len(train_w)}  test={len(test_w)}")

    # Score clean train windows
    print("\nScoring clean train windows...")
    scores = [score_window(w['accel']) for w in train_w]
    nonzero = [s for s in scores if s > 0]

    # Compute adaptive thresholds from percentiles
    gold_t, silver_t, floor_t = compute_adaptive_tiers(nonzero)
    print(f"  Score distribution (non-zero): {Counter(nonzero).most_common()}")
    print(f"\n  Adaptive tier thresholds:")
    print(f"    GOLD   >= {gold_t}  (p{DEFAULT_GOLD_PCT})")
    print(f"    SILVER >= {silver_t}  (p{DEFAULT_SILVER_PCT})")
    print(f"    floor  >= {floor_t}  (p{DEFAULT_FLOOR_PCT})")

    # Assign tiers
    tier_counts = Counter()
    for i,w in enumerate(train_w):
        s = scores[i]
        w['tier']  = get_tier(s, gold_t, silver_t, floor_t)
        w['score'] = s
        tier_counts[w['tier']] += 1
    print(f"\n  Tier distribution:")
    for t in ['GOLD','SILVER','BRONZE','REJECTED']:
        bar = '█' * (tier_counts[t]//100)
        print(f"    {t:<10} {tier_counts[t]:5d}  {bar}")

    # Normalization stats from clean train
    clean_feats = [extract_features(w['accel']) for w in train_w]
    _, norm_stats = normalize([f for f in clean_feats if f])

    def norm(samples):
        if not samples: return []
        feats = [f for f,_,_ in samples]
        fn, _ = normalize(feats, norm_stats)
        return [(fn[i],samples[i][1],samples[i][2])
                for i in range(len(fn))]

    # Test set
    test_feats  = [extract_features(w['accel']) for w in test_w]
    test_labels = [DOM2IDX[w['domain']] for w in test_w]
    tv = [(f,l) for f,l in zip(test_feats,test_labels) if f]
    test_eval = list(zip(
        normalize([f for f,_ in tv], norm_stats)[0],
        [l for _,l in tv]
    ))
    print(f"\nTest samples: {len(test_eval)}")

    # Inject corruption
    print(f"\nInjecting corruption ({CORRUPT_RATE*100:.0f}%)...")
    train_corrupted = inject_corruption(train_w)

    # Re-score and apply floor to corrupted set
    print("Applying adaptive floor to corrupted...")
    for w in train_corrupted:
        s = score_window(w['accel'])
        w['score'] = s
        w['tier']  = get_tier(s, gold_t, silver_t, floor_t)
    floor_kept = [w for w in train_corrupted if w['tier'] != 'REJECTED']
    print(f"  Kept {len(floor_kept)}/{len(train_corrupted)}")

    # Build condition sample sets
    TIER_W = {'GOLD':1.0,'SILVER':0.8,'BRONZE':0.4,'REJECTED':0.0}

    def make_samples(windows, use_tier_weight=False):
        out = []
        for w in windows:
            f = extract_features(w['accel'])
            if f and w['domain'] in DOM2IDX:
                wt = TIER_W.get(w.get('tier','SILVER'),1.0) \
                     if use_tier_weight else 1.0
                if wt > 0:
                    out.append((f, DOM2IDX[w['domain']], wt))
        return out

    sA = make_samples(train_w)
    sB = make_samples(train_corrupted)
    sC = make_samples(floor_kept)
    sD = make_samples(floor_kept, use_tier_weight=True)

    # Curriculum tiers
    gold_w   = [w for w in floor_kept if w['tier']=='GOLD']
    silver_w = [w for w in floor_kept if w['tier'] in ('GOLD','SILVER')]
    sE_p1 = make_samples(gold_w)
    sE_p2 = make_samples(silver_w)
    sE_p3 = make_samples(floor_kept)

    sA=norm(sA); sB=norm(sB); sC=norm(sC); sD=norm(sD)
    sE_p1=norm(sE_p1); sE_p2=norm(sE_p2); sE_p3=norm(sE_p3)

    ep1 = max(1, int(epochs*0.3))
    ep2 = max(1, int(epochs*0.3))
    ep3 = epochs - ep1 - ep2

    print(f"\nCondition sizes:")
    print(f"  A clean:       {len(sA)}")
    print(f"  B corrupted:   {len(sB)}")
    print(f"  C floor:       {len(sC)}")
    print(f"  D weighted:    {len(sD)}")
    print(f"  E curriculum:  p1={len(sE_p1)} p2={len(sE_p2)} p3={len(sE_p3)}")

    results = {}
    for cond, train_data, p1, p2, desc in [
        ("A_clean",     sA,   None,    None,    "Clean baseline"),
        ("B_corrupted", sB,   None,    None,    f"Corrupted {CORRUPT_RATE*100:.0f}%"),
        ("C_floor",     sC,   None,    None,    "Adaptive floor"),
        ("D_weighted",  sD,   None,    None,    "Tier-weighted loss"),
        ("E_curriculum",None, sE_p1,   sE_p2,   "Curriculum GOLD→SILVER→ALL"),
    ]:
        print(f"\n{'─'*62}")
        print(f"Condition {cond}  |  {desc}")
        model = MLP(input_dim=14, hidden=64, output=len(DOMAINS), lr=0.001)
        t0 = time.time()

        if cond == "E_curriculum":
            print(f"  Phase 1 GOLD only   ({ep1} epochs, n={len(p1)})")
            for ep in range(ep1):
                loss = model.train_epoch(p1)
                if (ep+1)%10==0:
                    acc,f1=model.evaluate(test_eval)
                    print(f"  P1 ep{ep+1:3d}  loss={loss:.4f}  f1={f1:.4f}")
            print(f"  Phase 2 +SILVER     ({ep2} epochs, n={len(p2)})")
            for ep in range(ep2):
                loss = model.train_epoch(p2)
                if (ep+1)%10==0:
                    acc,f1=model.evaluate(test_eval)
                    print(f"  P2 ep{ep+1:3d}  loss={loss:.4f}  f1={f1:.4f}")
            print(f"  Phase 3 ALL tiers   ({ep3} epochs, n={len(sE_p3)})")
            for ep in range(ep3):
                loss = model.train_epoch(sE_p3)
                if (ep+1)%10==0:
                    acc,f1=model.evaluate(test_eval)
                    print(f"  P3 ep{ep+1:3d}  loss={loss:.4f}  f1={f1:.4f}")
            n_train = len(sE_p3)
        else:
            for ep in range(epochs):
                loss = model.train_epoch(train_data)
                if (ep+1)%10==0:
                    acc,f1=model.evaluate(test_eval)
                    print(f"  ep{ep+1:3d}/{epochs}  loss={loss:.4f}  "
                          f"acc={acc:.3f}  f1={f1:.4f}")
            n_train = len(train_data)

        acc,f1 = model.evaluate(test_eval)
        elapsed = time.time()-t0
        print(f"  Final: acc={acc:.4f}  F1={f1:.4f}  [{elapsed:.0f}s]")
        results[cond] = {"description":desc,"accuracy":round(acc,4),
                         "macro_f1":round(f1,4),"train_n":n_train}

    # Summary
    print(f"\n{'='*62}")
    print("  LEVEL 2 ADAPTIVE TIERS — PAMAP2")
    print(f"  gold_threshold={gold_t}  silver_threshold={silver_t}  floor={floor_t}")
    print(f"{'='*62}\n")
    print(f"  {'Condition':<22} {'F1':>6}  {'n':>6}  Description")
    print("  "+"-"*58)
    for c,r in results.items():
        print(f"  {c:<22} {r['macro_f1']:.4f}  "
              f"{r['train_n']:>6}  {r['description']}")

    rA=results.get('A_clean',{})
    rB=results.get('B_corrupted',{})
    rC=results.get('C_floor',{})
    rD=results.get('D_weighted',{})
    rE=results.get('E_curriculum',{})
    print()
    if all([rA,rB,rC,rD,rE]):
        damage  = rB['macro_f1']-rA['macro_f1']
        recover = rC['macro_f1']-rB['macro_f1']
        weight  = rD['macro_f1']-rC['macro_f1']
        curric  = rE['macro_f1']-rA['macro_f1']
        print(f"  Corruption damage   (B-A): {damage*100:+.2f}% F1")
        print(f"  Floor recovery      (C-B): {recover*100:+.2f}% F1")
        print(f"  Tier weight gain    (D-C): {weight*100:+.2f}% F1")
        print(f"  Curriculum vs clean (E-A): {curric*100:+.2f}% F1  ← Level 2 claim")
        print()
        if curric > 0.005:
            print("  ✓ LEVEL 2 PROVEN ON PAMAP2 — adaptive tiers unlocked curriculum")
        elif curric > -0.005:
            print("  ~ Neutral — curriculum matches clean with adaptive tiers")
        else:
            print("  ✗ Curriculum did not improve with adaptive tiers")

    results['meta'] = {
        "experiment":     "s2s_level2_pamap2_adaptive_tiers",
        "dataset":        "PAMAP2",
        "hz":             HZ,
        "window_size":    WINDOW_SIZE,
        "epochs":         epochs,
        "corrupt_rate":   CORRUPT_RATE,
        "adaptive_tiers": {
            "gold_threshold":   gold_t,
            "silver_threshold": silver_t,
            "floor":            floor_t,
            "gold_pct":         DEFAULT_GOLD_PCT,
            "silver_pct":       DEFAULT_SILVER_PCT,
            "floor_pct":        DEFAULT_FLOOR_PCT,
        },
        "tier_counts": dict(tier_counts),
    }
    with open(output_path,'w') as f:
        json.dump(results,f,indent=2)
    print(f"\n  Saved → {output_path}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default='data/pamap2/')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--out',
        default='experiments/results_level2_pamap2_adaptive.json')
    args = parser.parse_args()
    run(args.data, args.out, args.epochs)
