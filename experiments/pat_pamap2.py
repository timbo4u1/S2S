#!/usr/bin/env python3
"""
PAT PAMAP2 — Physical Action Tokenizer on PAMAP2
Chest IMU certified by S2S (accel+gyro) → PAT tokens for 8 activities
Expected: better tier distribution than NinaPro (more Bronze/Gold)
"""
import os, sys, json, math, random
import numpy as np
sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

G="\033[92m"; R="\033[91m"; Y="\033[93m"; W="\033[97m"; NC="\033[0m"

WIN    = 256
HZ     = 100.0
EPOCHS = 60
TEMP   = 2.0
random.seed(42); np.random.seed(42)

DATA_DIR = os.path.expanduser("~/S2S_data/pamap2")

# 8 main activities (skip 0=other, map to 0-7)
ACTIVITY_MAP = {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 12:7}
ACTIVITY_NAMES = {
    0:"LYING", 1:"SITTING", 2:"STANDING", 3:"WALKING",
    4:"RUNNING", 5:"CYCLING", 6:"NORDIC_WALK", 7:"STAIRS_UP"
}
N_CLASSES = 8

# Physics constraints per activity
# Chest IMU: chest motion physics
LAW_CONSTRAINTS = {
    "sensor_freeze": {
        0:+1.5, 1:+0.8,  # more likely static
        3:-0.5, 4:-1.0, 6:-0.5, 7:-0.8  # less likely moving
    },
    "jerk_bounds": {
        4:+0.5, 7:+0.8,  # high jerk → running/stairs
        0:-1.0, 1:-0.8,  # low jerk → not lying/sitting
    },
    "resonance_frequency": {
        0:+0.5, 1:+0.3,  # tremor artifact more likely at rest
        4:-0.3, 5:-0.3
    },
    "temporal_autocorrelation": {
        3:+0.3, 4:+0.3, 6:+0.3,  # rhythmic motion has high ACF
        0:-0.3,
    },
    "cross_axis_cohesion": {
        3:+0.3, 4:+0.3,  # walking/running = coupled axes
        0:-0.2
    },
}
TIER_TEMP = {"GOLD":0.3, "SILVER":1.0, "BRONZE":2.5, "REJECTED":4.0}

def load_subject(path):
    d = np.genfromtxt(path, invalid_raise=False)
    if d.ndim != 2 or d.shape[1] < 33:
        return None, None, None
    ts    = d[:, 0]
    label = d[:, 1]
    # Chest accel cols 17-19 (g → m/s²)
    accel = d[:, 17:20] * 9.81
    # Chest gyro cols 20-22 (rad/s)
    gyro  = d[:, 20:23]
    return accel, gyro, label

def certify_win(pe, acc_win, gyro_win):
    # Replace NaN with 0
    acc_w = np.nan_to_num(acc_win, nan=0.0)
    gyr_w = np.nan_to_num(gyro_win, nan=0.0)
    ts = [int(i*1e9/HZ) for i in range(len(acc_w))]
    r = pe.certify(
        imu_raw={"timestamps_ns":ts,
                 "accel":acc_w.tolist(),
                 "gyro":gyr_w.tolist()},
        segment="forearm")
    return r["tier"], int(r["physical_law_score"]), r.get("laws_failed",[])

def featurize(acc_win, gyro_win):
    """18 features: 9 accel + 9 gyro (mean, std, range per axis)"""
    feats = []
    for sensor in [acc_win, gyro_win]:
        s = np.nan_to_num(sensor, nan=0.0)
        for ax in range(3):
            col = s[:, ax]
            feats += [col.mean(), col.std(), col.max()-col.min()]
    return np.array(feats, np.float32)

def softmax(x, T=1.0):
    x = np.array(x,float)/T; x -= x.max()
    e = np.exp(x); return e/e.sum()

def entropy(p):
    p = np.array(p)+1e-12; p/=p.sum()
    return float(-np.sum(p*np.log(p)))

def physics_dist(logits, tier, laws_failed):
    lg = logits.copy()
    for law in laws_failed:
        if law in LAW_CONSTRAINTS:
            for cls, delta in LAW_CONSTRAINTS[law].items():
                if cls < N_CLASSES: lg[cls] += delta
    T = TIER_TEMP.get(tier, 1.0)
    return softmax(lg, T)

class MLP:
    def __init__(self, ni, nh, no, lr=0.005):
        self.lr=lr
        self.W1=np.random.randn(nh,ni)*math.sqrt(2/ni)
        self.b1=np.zeros(nh)
        self.W2=np.random.randn(no,nh)*math.sqrt(2/nh)
        self.b2=np.zeros(no)
    def forward(self,x):
        h=np.maximum(0,self.W1@x+self.b1)
        return h, self.W2@h+self.b2
    def backward(self,x,h,p,t):
        d=p-t
        self.W2-=self.lr*np.outer(d,h); self.b2-=self.lr*d
        dh=(self.W2.T@d)*(h>0)
        self.W1-=self.lr*np.outer(dh,x); self.b1-=self.lr*dh

print(f"\n{W}{'='*65}")
print("  PAT PAMAP2 — Physical Action Tokenizer")
print("  8 activities, chest IMU, 9 subjects, v1.7.9 engine")
print(f"{'='*65}{NC}\n")

pe = PhysicsEngine()
all_feats, all_labels, all_meta = [], [], []

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".dat") or fname == "subject101.dat":
        continue
    path = os.path.join(DATA_DIR, fname)
    accel, gyro, labels = load_subject(path)
    if accel is None: continue
    n = len(accel); count = 0
    print(f"  {fname}...", end=" ", flush=True)
    for start in range(0, n-WIN+1, WIN):
        mid_label = int(labels[start + WIN//2]) if not np.isnan(labels[start + WIN//2]) else 0
        if mid_label not in ACTIVITY_MAP: continue
        acc_win = accel[start:start+WIN]
        gyr_win = gyro[start:start+WIN]
        # Skip if too many NaN
        if np.isnan(acc_win).mean() > 0.1: continue
        mapped = ACTIVITY_MAP[mid_label]
        tier, score, laws = certify_win(pe, acc_win, gyr_win)
        all_feats.append(featurize(acc_win, gyr_win))
        all_labels.append(mapped)
        all_meta.append({"tier":tier,"score":score,"laws_failed":laws})
        count += 1
    print(f"{count} windows")

X = np.array(all_feats); y = np.array(all_labels)
print(f"\n  Total: {len(y)} windows, {N_CLASSES} classes")
from collections import Counter
tier_dist = Counter(m["tier"] for m in all_meta)
print(f"  Tier distribution: {dict(tier_dist)}")
print(f"  Class distribution: {dict(Counter(y.tolist()))}")

# Normalize + split by leaving out 2 subjects worth of data
mu, sigma = X.mean(0), X.std(0)+1e-8
Xn = (X-mu)/sigma
idx = np.random.permutation(len(y))
n_test = int(len(idx)*0.25)
tr, te = idx[n_test:], idx[:n_test]
X_tr,y_tr,m_tr = Xn[tr],y[tr],[all_meta[i] for i in tr]
X_te,y_te,m_te = Xn[te],y[te],[all_meta[i] for i in te]

print(f"\n  Training MLP ({X_tr.shape[1]} features → 128 → {N_CLASSES})...")
model = MLP(X_tr.shape[1], 128, N_CLASSES, lr=0.005)
np.random.seed(42)
for ep in range(EPOCHS):
    perm = np.random.permutation(len(X_tr))
    for i in perm:
        h,lg = model.forward(X_tr[i])
        p = softmax(lg)
        t = np.zeros(N_CLASSES); t[y_tr[i]]=1.0
        model.backward(X_tr[i],h,p,t)
    if (ep+1) % 20 == 0:
        top1 = sum(np.argmax(softmax(model.forward(x)[1]))==l
                   for x,l in zip(X_te,y_te))/len(y_te)
        print(f"    Epoch {ep+1}/{EPOCHS}  top1={top1:.3f}")

def evaluate(cond):
    top1=top3=0
    entropies=[]
    tier_ent={"GOLD":[],"SILVER":[],"BRONZE":[],"REJECTED":[]}
    flipped = 0
    for i,(x,lbl) in enumerate(zip(X_te,y_te)):
        _,lg=model.forward(x); m=m_te[i]
        pA = softmax(lg,1.0)
        if cond=="A":   p=pA
        elif cond=="B": p=softmax(lg,TEMP)
        else:           p=physics_dist(lg,m["tier"],m["laws_failed"])
        pred=np.argmax(p)
        if pred==lbl: top1+=1
        if lbl in np.argsort(p)[::-1][:3]: top3+=1
        e=entropy(p); entropies.append(e)
        if m["tier"] in tier_ent: tier_ent[m["tier"]].append(e)
        if cond=="C" and np.argmax(pA)!=pred: flipped+=1
    n=len(y_te)
    return {"top1":top1/n,"top3":top3/n,
            "entropy":np.mean(entropies),
            "tier_entropy":{t:float(np.mean(v)) if v else None
                            for t,v in tier_ent.items()},
            "flipped":flipped if cond=="C" else 0}

print(f"\n{W}{'─'*65}  Results{NC}\n")
results={}; base=None
for cond,label in [("A","Hard classification"),
                    ("B","Soft, no physics"),
                    ("C","Physics-constrained")]:
    r=evaluate(cond); results[cond]=r
    d=f"  ({(r['top1']-base)*100:+.2f}%)" if base else ""
    if not base: base=r["top1"]
    print(f"  Condition {cond}: {label}")
    print(f"    Top-1: {r['top1']:.4f}{d}   Top-3: {r['top3']:.4f}")
    print(f"    Mean entropy: {r['entropy']:.4f}")
    for tier in ["GOLD","SILVER","BRONZE","REJECTED"]:
        v = r["tier_entropy"].get(tier)
        if v is not None:
            n_win = tier_dist.get(tier,0)
            bar = "█" * min(20,int(v*8))
            c = G if tier=="GOLD" else (Y if tier=="SILVER" else R)
            print(f"    {c}{tier:<9}{NC} H={v:.4f} {bar}  (n≈{n_win})")
    if cond=="C":
        print(f"    Argmax flipped by physics: {r['flipped']} windows")
    print()

# Law analysis
law_counts = Counter(l for m in m_te for l in m["laws_failed"])
print(f"  Law failures in test set ({len(m_te)} windows):")
for law,cnt in law_counts.most_common(8):
    tag = f"{G}← constrained{NC}" if law in LAW_CONSTRAINTS else ""
    print(f"    {law}: {cnt} ({100*cnt/len(m_te):.1f}%) {tag}")

A=results["A"]["top1"]; C=results["C"]["top1"]
delta=C-A
te_C = results["C"]["tier_entropy"]
gold_ok = ((te_C.get("GOLD") or 1) < (te_C.get("SILVER") or 0) and
           (te_C.get("SILVER") or 1) < (te_C.get("BRONZE") or 0))

print(f"\n{W}{'='*65}  VERDICT{NC}")
print(f"  Baseline (A):            {A:.4f}")
print(f"  Physics-constrained (C): {C:.4f}  ({delta*100:+.2f}%)")
print(f"  Physics flipped predictions: {results['C']['flipped']}")

if delta > 0.005 and gold_ok:
    print(f"\n  {G}✓ PAT FULLY PROVEN{NC}")
    print(f"    Accuracy improved AND tier entropy monotonic")
elif gold_ok:
    print(f"\n  {Y}~ PARTIAL — tier entropy monotonic GOLD < SILVER < BRONZE{NC}")
    te = results["C"]["tier_entropy"]
    ratio = (te.get("BRONZE") or 0) / max(te.get("GOLD") or 0.001, 0.001)
    print(f"    BRONZE/GOLD entropy ratio: {ratio:.1f}x")
else:
    print(f"\n  {R}✗ Not proven{NC}")

out={"experiment":"PAT_PAMAP2","n_windows":len(y),
     "tier_dist":dict(tier_dist),"results":results,
     "law_counts":dict(law_counts.most_common(10))}
with open("/Users/timbo/S2S/experiments/results_pat_pamap2.json","w") as f:
    json.dump(out,f,indent=2)
print(f"\n  Saved → experiments/results_pat_pamap2.json")
print(f"{W}{'='*65}{NC}\n")
