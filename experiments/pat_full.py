#!/usr/bin/env python3
"""
PAT Full — Physical Action Tokenizer on all 10 NinaPro DB5 subjects
Uses E1 exercise (12 gesture classes + rest)
Certifies fresh with v1.7.9 engine, then runs 3-condition experiment
"""
import os, sys, json, math, random
import numpy as np
import scipy.io
sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

G="\033[92m"; R="\033[91m"; Y="\033[93m"; W="\033[97m"; NC="\033[0m"

WIN    = 256
HZ     = 200.0
EPOCHS = 50
TEMP   = 2.0
random.seed(42); np.random.seed(42)

# E1 gesture names (NinaPro DB5)
GESTURE_NAMES = {
    0:"REST", 1:"THUMB_FLEX", 2:"INDEX_FLEX", 3:"MIDDLE_FLEX",
    4:"RING_FLEX", 5:"LITTLE_FLEX", 6:"WRIST_FLEX", 7:"WRIST_EXT",
    8:"RADIAL_DEV", 9:"ULNAR_DEV", 10:"HAND_OPEN", 11:"POWER_GRASP",
    12:"LATERAL_PINCH"
}
N_CLASSES = 13

LAW_CONSTRAINTS = {
    "resonance_frequency": {0:+0.8, 6:-0.3, 7:-0.3, 11:-0.4, 12:-0.4},
    "sensor_freeze":       {0:+1.5, 1:-0.5, 6:-0.8, 11:-1.0},
    "jerk_bounds":         {0:-1.0, 11:+0.5, 12:+0.5},
    "temporal_autocorrelation": {0:-0.5, 6:+0.3, 11:+0.3},
    "cross_axis_cohesion": {0:+0.5, 6:+0.2, 7:+0.2},
    "innovation_kurtosis": {0:+0.3, 6:-0.1, 11:-0.2},
}
TIER_TEMP = {"GOLD":0.4, "SILVER":1.0, "BRONZE":2.0, "REJECTED":3.0}

def load_subject(s):
    path = os.path.expanduser(f"~/ninapro_db5/s{s}/S{s}_E1_A1.mat")
    if not os.path.exists(path):
        path = os.path.expanduser(f"~/ninapro_db5/s{s}/s{s}/S{s}_E1_A1.mat")
    d = scipy.io.loadmat(path)
    return (d["acc"].astype(np.float32),
            d["emg"].astype(np.float32),
            d["stimulus"].flatten().astype(int))

def featurize(acc_win, emg_win):
    """76 features: 12 ACC + 64 EMG (4 stats x 16 channels)."""
    # ACC features (12)
    w = acc_win * 9.81
    feats = []
    for ax in range(3):
        col = w[:, ax]
        feats += [col.mean(), col.std(), col.max()-col.min()]
    mag = np.linalg.norm(w, axis=1)
    feats += [mag.mean(), mag.std(), mag.max()]

    # EMG features (64 = 16 channels x 4 stats)
    # Standard hand gesture recognition features
    for ch in range(emg_win.shape[1]):
        col = emg_win[:, ch].astype(float)
        mav = np.abs(col).mean()          # Mean Absolute Value
        rms = np.sqrt((col**2).mean())     # Root Mean Square
        wl  = np.abs(np.diff(col)).sum()   # Waveform Length
        zc  = np.sum(np.diff(np.sign(col)) != 0) / len(col)  # Zero Crossing
        feats += [mav, rms, wl, zc]

    return np.array(feats, np.float32)

def certify_window(pe, acc_win):
    ts = [int(i*1e9/HZ) for i in range(len(acc_win))]
    r = pe.certify(imu_raw={"timestamps_ns":ts,
                              "accel":acc_win.tolist(),
                              "gyro":[[0.0,0.0,0.0]]*len(acc_win)},
                   segment="forearm")
    return r["tier"], int(r["physical_law_score"]), r.get("laws_failed",[])

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
        d=p-t; self.W2-=self.lr*np.outer(d,h); self.b2-=self.lr*d
        dh=(self.W2.T@d)*(h>0)
        self.W1-=self.lr*np.outer(dh,x); self.b1-=self.lr*dh

print(f"\n{W}{'='*65}")
print("  PAT Full — 10 subjects, NinaPro DB5 E1, 13 classes, EMG+ACC")
print(f"{'='*65}{NC}\n")

# Load all subjects
pe = PhysicsEngine()
all_feats, all_labels, all_meta = [], [], []

for s in range(1, 11):
    print(f"  Subject s{s}...", end=" ", flush=True)
    try:
        acc, emg, stim = load_subject(s)
        n, count = len(acc), 0
        for start in range(0, n-WIN+1, WIN//2):
            win_acc = acc[start:start+WIN]
            win_emg = emg[start:start+WIN]
            label = int(stim[start + WIN//2])
            if label >= N_CLASSES: continue
            tier, score, laws = certify_window(pe, win_acc)
            all_feats.append(featurize(win_acc, win_emg))
            all_labels.append(label)
            all_meta.append({"tier":tier,"score":score,"laws_failed":laws})
            count += 1
        print(f"{count} windows")
    except Exception as e:
        print(f"ERROR: {e}")

X = np.array(all_feats); y = np.array(all_labels)
print(f"\n  Total: {len(y)} windows, {N_CLASSES} classes")
from collections import Counter
print(f"  Tiers: {dict(Counter(m['tier'] for m in all_meta))}")
print(f"  Classes: {len(set(y.tolist()))} unique")

# Normalize + split
mu, sigma = X.mean(0), X.std(0)+1e-8
X = (X-mu)/sigma
idx = np.random.permutation(len(y))
n_test = int(len(idx)*0.25)
tr, te = idx[n_test:], idx[:n_test]
X_tr,y_tr,m_tr = X[tr],y[tr],[all_meta[i] for i in tr]
X_te,y_te,m_te = X[te],y[te],[all_meta[i] for i in te]

# Train
print("\n  Training MLP...")
model = MLP(X_tr.shape[1], 256, N_CLASSES, lr=0.002)
np.random.seed(42)
for ep in range(EPOCHS):
    perm = np.random.permutation(len(X_tr))
    for i in perm:
        h,lg = model.forward(X_tr[i])
        p = softmax(lg)
        t = np.zeros(N_CLASSES); t[y_tr[i]] = 1.0
        model.backward(X_tr[i],h,p,t)
    if (ep+1) % 10 == 0:
        top1 = sum(np.argmax(softmax(model.forward(x)[1])) == l
                   for x,l in zip(X_te,y_te)) / len(y_te)
        print(f"    Epoch {ep+1}/{EPOCHS}  top1={top1:.3f}")

# Evaluate 3 conditions
def evaluate(cond):
    top1=top3=0; entropies=[]; tier_ent={"GOLD":[],"SILVER":[],"BRONZE":[]}
    for i,(x,lbl) in enumerate(zip(X_te,y_te)):
        _,lg = model.forward(x)
        m = m_te[i]
        if cond=="A":   p = softmax(lg, 1.0)
        elif cond=="B": p = softmax(lg, TEMP)
        else:           p = physics_dist(lg, m["tier"], m["laws_failed"])
        if np.argmax(p)==lbl: top1+=1
        if lbl in np.argsort(p)[::-1][:3]: top3+=1
        e=entropy(p); entropies.append(e)
        if m["tier"] in tier_ent: tier_ent[m["tier"]].append(e)
    n=len(y_te)
    return {"top1":top1/n,"top3":top3/n,
            "entropy":np.mean(entropies),
            "tier_entropy":{t:float(np.mean(v)) if v else None
                            for t,v in tier_ent.items()}}

print(f"\n{W}{'─'*65}  Results{NC}\n")
results = {}
base_top1 = None
for cond, label in [("A","Hard classification"),
                     ("B","Soft, no physics"),
                     ("C","Physics-constrained")]:
    r = evaluate(cond); results[cond] = r
    d = f"  ({(r['top1']-base_top1)*100:+.2f}%)" if base_top1 else ""
    if not base_top1: base_top1 = r["top1"]
    print(f"  Condition {cond}: {label}")
    print(f"    Top-1: {r['top1']:.4f}{d}   Top-3: {r['top3']:.4f}")
    print(f"    Entropy: {r['entropy']:.4f}")
    gold = r["tier_entropy"].get("GOLD")
    silv = r["tier_entropy"].get("SILVER")
    bron = r["tier_entropy"].get("BRONZE")
    if gold: print(f"    {G}GOLD entropy: {gold:.4f}{NC}")
    if silv: print(f"    SILVER entropy: {silv:.4f}")
    if bron: print(f"    BRONZE entropy: {bron:.4f}")
    print()

# Law failure analysis
from collections import Counter
law_counts = Counter(l for m in m_te for l in m["laws_failed"])
print(f"  Law failures in test set ({len(m_te)} windows):")
for law, cnt in law_counts.most_common(8):
    tag = f"{G}← constrained{NC}" if law in LAW_CONSTRAINTS else ""
    print(f"    {law}: {cnt} ({100*cnt/len(m_te):.1f}%) {tag}")

# Verdict
A = results["A"]["top1"]; C = results["C"]["top1"]
delta = C - A
gold_ok = (results["C"]["tier_entropy"].get("GOLD") or 1) < \
          (results["C"]["tier_entropy"].get("SILVER") or 0)

print(f"\n{W}{'='*65}  VERDICT{NC}")
print(f"  Baseline (A):            {A:.4f}")
print(f"  Physics-constrained (C): {C:.4f}  ({delta*100:+.2f}%)")
if delta > 0.005 and gold_ok:
    print(f"\n  {G}✓ PAT PROVEN — physics improves accuracy + tier correlates{NC}")
elif gold_ok:
    print(f"\n  {Y}~ PARTIAL — tier entropy correct, accuracy unchanged{NC}")
else:
    print(f"\n  {R}✗ Not proven on this data{NC}")

out = {"experiment":"PAT_full_10subjects",
       "n_windows":len(y),"n_classes":N_CLASSES,
       "results":{k:v for k,v in results.items()},
       "law_counts":dict(law_counts.most_common(10))}
with open("experiments/results_pat_full.json","w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved → experiments/results_pat_full.json")
print(f"{W}{'='*65}{NC}\n")
