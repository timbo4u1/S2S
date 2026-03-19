#!/usr/bin/env python3
"""
Step 3 — Retrieval by Instruction Text
========================================
Given: "pick up cup" (any text)
Returns: top-K certified windows ranked by physics quality + motion similarity
How:
  1. Load all certified windows + their motion embeddings from trained encoder
  2. Map instruction text → motion space via keyword similarity
  3. Rank by (semantic_match * 0.4) + (physics_score * 0.4) + (smoothness * 0.2)
  4. Save retrieval index as JSON for demo
"""

import os, sys, json, glob, pickle
import numpy as np
import scipy.io
sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
import torch
import torch.nn as nn

# ── Load trained model ────────────────────────────────────────────────────────
class DualHead(nn.Module):
    def __init__(self, ni, nc, nm):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(ni,128),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(128,256),nn.ReLU(),nn.Dropout(0.3),
            nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2),
            nn.Linear(128,64),nn.ReLU())
        self.qh = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,nc))
        self.mh = nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,nm))
    def forward(self,x): e=self.enc(x); return self.qh(e),self.mh(e)
    def encode(self,x): return self.enc(x)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model  = DualHead(13, 3, 9).to(device)
model.load_state_dict(torch.load(
    os.path.expanduser("~/S2S/experiments/level5_corrected_best.pt"),
    map_location=device))
model.eval()
print(f"Model loaded | device={device}")

engine = PhysicsEngine()

# ── Feature extraction (same as training) ────────────────────────────────────
def extract_features(accel, gyro, hz):
    f = []
    f.append(float(np.sqrt(np.mean(accel**2))))
    f.append(float(np.std(accel)))
    f.append(float(np.max(np.abs(accel))))
    f.append(float(np.sqrt(np.mean(gyro**2))))
    f.append(float(np.std(gyro)))
    if len(accel) > 3:
        jerk = np.diff(accel, axis=0) * hz
        f.append(float(np.sqrt(np.mean(jerk**2))))
        f.append(float(np.percentile(np.abs(jerk), 95)))
    else:
        f.extend([0.0, 0.0])
    for axis in range(min(3, accel.shape[1])):
        fft   = np.abs(np.fft.rfft(accel[:, axis]))
        freqs = np.fft.rfftfreq(len(accel), 1/hz)
        f.append(float(freqs[np.argmax(fft)] if len(fft) > 0 else 0))
    c = np.corrcoef(accel[:,0], accel[:,1])[0,1]
    f.append(float(c) if not np.isnan(c) else 0.0)
    f.append(float(np.linalg.norm(np.mean(accel, axis=0))))
    hist, _ = np.histogram(accel.flatten(), bins=20)
    hist = hist / (hist.sum() + 1e-10)
    f.append(float(-np.sum(hist * np.log(hist + 1e-10))))
    return np.array(f, dtype=np.float32)

# ── Instruction keyword map ───────────────────────────────────────────────────
# Maps any free-text query → closest instruction category
# Based on actual RoboTurk commands + semantic expansion
INSTRUCTION_KEYWORDS = {
    "object search": [
        "pick", "grab", "get", "take", "reach", "find", "search",
        "fetch", "retrieve", "collect", "cup", "object", "item", "thing"
    ],
    "create tower": [
        "stack", "build", "tower", "block", "place on top", "pile",
        "assemble", "construct", "lift", "raise", "vertical"
    ],
    "layout laundry": [
        "lay", "spread", "arrange", "fold", "laundry", "cloth",
        "flat", "place", "put", "set", "organize", "sort", "horizontal"
    ]
}

def text_to_instruction(query):
    """Map free text query to instruction category + confidence scores."""
    query_lower = query.lower()
    scores = {}
    for instr, keywords in INSTRUCTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        scores[instr] = score / len(keywords)
    # Normalize
    total = sum(scores.values()) + 1e-8
    return {k: v/total for k, v in scores.items()}

# ── Build retrieval index ─────────────────────────────────────────────────────
print("\nBuilding retrieval index from all certified data...")

index = []  # list of dicts

# ── NinaPro ───────────────────────────────────────────────────────────────────
print("[1/3] NinaPro DB5...")
nin_count = 0
for subj_dir in sorted(glob.glob(os.path.expanduser("~/ninapro_db5/s*"))):
    accel_all = []
    for mat_path in sorted(glob.glob(os.path.join(subj_dir,"*.mat"))):
        try:
            mat = scipy.io.loadmat(mat_path)
            acc = mat.get("acc", mat.get("ACC"))
            if acc is not None: accel_all.append(acc)
        except: continue
    if not accel_all: continue
    accel = np.vstack(accel_all).astype(np.float64)
    gyro  = np.zeros_like(accel)
    for i in range(0, len(accel)-4000, 2000):
        ca = accel[i:i+2000,:3]; cg = gyro[i:i+2000,:3]
        ts = [int(j*1e9/2000) for j in range(len(ca))]
        r  = engine.certify(imu_raw={"timestamps_ns":ts,"accel":ca.tolist(),"gyro":cg.tolist()},segment="forearm")
        if r.get("tier","REJECTED") == "REJECTED": continue
        feats = extract_features(ca, cg, 2000)
        jerk  = np.diff(ca, axis=0)*2000
        smoothness = float(1.0/(1.0+np.sqrt(np.mean(jerk**2)+1e-8)))
        index.append({
            "source":      "ninapro_db5",
            "instruction": "hand_motion",
            "semantic_tags": ["grab", "pick", "reach", "hand", "finger", "grip"],
            "tier":        r.get("tier"),
            "score":       r.get("physical_law_score", 0),
            "smoothness":  round(smoothness, 4),
            "features":    feats.tolist(),
            "laws_passed": r.get("laws_passed", []),
        })
        nin_count += 1
        if nin_count >= 500: break
    if nin_count >= 500: break
print(f"  → {nin_count} windows indexed")

# ── EMG Amputee ───────────────────────────────────────────────────────────────
print("[2/3] EMG Amputee...")
amp_count = 0
for subj in sorted(os.listdir(os.path.expanduser("~/S2S_Project/EMG_Amputee"))):
    inner = os.path.expanduser(f"~/S2S_Project/EMG_Amputee/{subj}/{subj}")
    if not os.path.isdir(inner): continue
    for csv_path in sorted(glob.glob(os.path.join(inner,"mov *","accelerometer-*.csv"))):
        try:
            data  = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            if data.ndim < 2 or data.shape[1] < 4: continue
            accel = data[:,1:4].astype(np.float64)
            gyro  = np.zeros_like(accel)
            for i in range(0, len(accel)-400, 200):
                ca = accel[i:i+200]; cg = gyro[i:i+200]
                ts = [int(j*1e9/200) for j in range(len(ca))]
                r  = engine.certify(imu_raw={"timestamps_ns":ts,"accel":ca.tolist(),"gyro":cg.tolist()},segment="forearm")
                if r.get("tier","REJECTED") == "REJECTED": continue
                feats = extract_features(ca, cg, 200)
                jerk  = np.diff(ca, axis=0)*200
                smoothness = float(1.0/(1.0+np.sqrt(np.mean(jerk**2)+1e-8)))
                index.append({
                    "source":      "emg_amputee",
                    "instruction": "hand_motion",
                    "semantic_tags": ["grab", "pick", "reach", "hand", "prosthetic", "grip"],
                    "tier":        r.get("tier"),
                    "score":       r.get("physical_law_score", 0),
                    "smoothness":  round(smoothness, 4),
                    "features":    feats.tolist(),
                    "laws_passed": r.get("laws_passed", []),
                })
                amp_count += 1
        except: continue
print(f"  → {amp_count} windows indexed")

# ── RoboTurk ──────────────────────────────────────────────────────────────────
print("[3/3] RoboTurk (with real instruction labels)...")
NYU_COUNT = 14
HZ = 15
WINDOW = 30
files = sorted(glob.glob(os.path.expanduser(
    "~/S2S/openx_data/sample_*.data.pickle")))[NYU_COUNT:]
rt_count = 0
for path in files:
    with open(path,"rb") as f: d = pickle.load(f)
    steps = d.get("steps",[])
    instr = "unknown"
    for step in steps:
        if not isinstance(step,dict): continue
        raw = step.get("observation",{}).get("natural_language_instruction","")
        if raw:
            instr = raw.decode() if isinstance(raw,bytes) else str(raw)
            break
    wvs, rds = [], []
    for step in steps:
        if not isinstance(step,dict): continue
        wv = step.get("action",{}).get("world_vector")
        rd = step.get("action",{}).get("rotation_delta")
        if wv is not None and rd is not None:
            wvs.append(np.array(wv,dtype=np.float32))
            rds.append(np.array(rd,dtype=np.float32))
    if len(wvs) < 10: continue
    wvs = np.array(wvs); rds = np.array(rds)
    pos  = np.cumsum(wvs, axis=0)
    vel  = np.diff(pos, axis=0)*HZ
    accel= np.diff(vel, axis=0)*HZ
    rots = np.cumsum(rds, axis=0)
    gv   = np.diff(rots, axis=0)*HZ
    gyro = np.diff(gv, axis=0)*HZ
    if len(gyro) != len(accel): gyro = np.zeros_like(accel)
    for i in range(0, len(accel)-WINDOW, WINDOW//2):
        ca = accel[i:i+WINDOW]; cg = gyro[i:i+WINDOW]
        if len(ca) < 10: continue
        ts = [int(j*1e9/HZ) for j in range(len(ca))]
        r  = engine.certify(imu_raw={"timestamps_ns":ts,"accel":ca.tolist(),"gyro":cg.tolist()},segment="forearm")
        if r.get("tier","REJECTED") == "REJECTED": continue
        feats = extract_features(ca, cg, HZ)
        jerk  = np.diff(ca, axis=0)*HZ
        smoothness = float(1.0/(1.0+np.sqrt(np.mean(jerk**2)+1e-8)))
        index.append({
            "source":      "roboturk",
            "instruction": instr,
            "semantic_tags": INSTRUCTION_KEYWORDS.get(instr, [instr]),
            "tier":        r.get("tier"),
            "score":       r.get("physical_law_score", 0),
            "smoothness":  round(smoothness, 4),
            "features":    feats.tolist(),
            "laws_passed": r.get("laws_passed", []),
            "pos_start":   [round(float(x),4) for x in pos[i]],
        })
        rt_count += 1

print(f"  → {rt_count} windows indexed")
print(f"\nTotal index size: {len(index)} certified windows")

# ── Compute motion embeddings for all windows ─────────────────────────────────
print("Computing motion embeddings...")
all_feats = np.array([w["features"] for w in index], dtype=np.float32)
with torch.no_grad():
    embeddings = model.encode(torch.FloatTensor(all_feats).to(device)).cpu().numpy()
for i, emb in enumerate(embeddings):
    index[i]["embedding"] = emb.tolist()
print(f"Embeddings: {embeddings.shape}")

# ── Retrieval function ────────────────────────────────────────────────────────
def retrieve(query, top_k=10):
    """
    Given a text query, return top-K certified windows.
    Ranking: semantic_match(0.35) + physics_score(0.35) + smoothness(0.20) + tier_bonus(0.10)
    """
    instr_scores = text_to_instruction(query)
    tier_bonus   = {"GOLD": 1.0, "SILVER": 0.66, "BRONZE": 0.33}

    results = []
    for i, w in enumerate(index):
        # Semantic match
        sem = instr_scores.get(w["instruction"], 0.0)
        # Also check raw keyword overlap
        q_words = set(query.lower().split())
        tag_overlap = len(q_words & set(w.get("semantic_tags",[]))) / max(len(q_words),1)
        semantic = max(sem, tag_overlap)

        # Physics score (normalized 0-1)
        phys = min(1.0, w["score"] / 100.0)

        # Smoothness (already 0-1 roughly, cap at 1)
        smooth = min(1.0, w["smoothness"] * 30)  # scale up tiny values

        # Tier bonus
        tb = tier_bonus.get(w["tier"], 0.0)

        # Final rank score
        rank = 0.35*semantic + 0.35*phys + 0.20*smooth + 0.10*tb

        results.append({
            "rank_score":  round(rank, 4),
            "semantic":    round(semantic, 3),
            "physics":     w["score"],
            "smoothness":  w["smoothness"],
            "tier":        w["tier"],
            "source":      w["source"],
            "instruction": w["instruction"],
            "laws_passed": w.get("laws_passed", []),
            "pos_start":   w.get("pos_start", [0,0,0]),
        })

    results.sort(key=lambda x: x["rank_score"], reverse=True)
    return results[:top_k]

# ── Test retrieval ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RETRIEVAL TEST")
print("="*60)

test_queries = [
    "pick up the cup",
    "grab object from table",
    "stack blocks to build tower",
    "fold and arrange laundry",
    "reach and grasp",
    "place item carefully",
]

retrieval_results = {}
for query in test_queries:
    results = retrieve(query, top_k=5)
    retrieval_results[query] = results
    print(f"\nQuery: '{query}'")
    print(f"  {'Rank':>6} {'Score':>7} {'Tier':>8} {'Source':>12} {'Instruction'}")
    for j, r in enumerate(results):
        print(f"  #{j+1:>4}  {r['rank_score']:>6.3f}  {r['tier']:>8}  "
              f"{r['source']:>12}  {r['instruction']}")

# ── Save index + results ──────────────────────────────────────────────────────
# Save compact index (no embeddings for JSON size)
index_compact = [{k:v for k,v in w.items() if k != "embedding"} for w in index]

out_index = os.path.expanduser("~/S2S/experiments/retrieval_index.json")
out_results = os.path.expanduser("~/S2S/experiments/retrieval_results.json")

with open(out_index, "w") as f:
    json.dump({"total": len(index), "windows": index_compact}, f, indent=2)
with open(out_results, "w") as f:
    json.dump({"queries": retrieval_results}, f, indent=2)

# Save embeddings separately as numpy
np.save(os.path.expanduser("~/S2S/experiments/retrieval_embeddings.npy"), embeddings)

print(f"\n{'='*60}")
print(f"Index saved  → {out_index} ({os.path.getsize(out_index)/1024:.0f} KB)")
print(f"Results saved→ {out_results}")
print(f"Embeddings   → retrieval_embeddings.npy {embeddings.shape}")
print(f"\nReady for demo: {len(index)} certified windows, {len(test_queries)} queries tested")
