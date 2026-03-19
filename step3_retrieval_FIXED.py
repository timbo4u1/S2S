#!/usr/bin/env python3
"""
Step 3 — Retrieval by Instruction Text (SEMANTIC FIX)
=========================================================
CRITICAL BUG FIX (March 19, 2026):
  - Replaced physics-trained encoder with proper semantic embeddings
  - OLD: Used 64-dim encoder trained for quality prediction → wrong!
  - NEW: Uses sentence-transformers for true semantic similarity
  - Result: "drink water" no longer matches "folding clothes"

Given: "pick up cup" (any text)
Returns: top-K certified windows ranked by physics quality + motion similarity
How:
  1. Load all certified windows + their motion embeddings
  2. Map instruction text → motion space via SEMANTIC similarity (FIXED!)
  3. Rank by (semantic_match * 0.4) + (physics_score * 0.4) + (smoothness * 0.2)
  4. Save retrieval index as JSON for demo

INSTALLATION REQUIRED:
  pip install sentence-transformers

MODEL USED:
  - all-MiniLM-L6-v2: 384-dim semantic embeddings
  - Trained on 1B+ text pairs for semantic similarity
  - Works out of the box, no training needed
"""

import os, sys, json, glob, pickle
import numpy as np
import scipy.io
sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

# SEMANTIC FIX: Import sentence-transformers for proper semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_MODEL_AVAILABLE = True
except ImportError:
    print("⚠️  WARNING: sentence-transformers not installed!")
    print("   Install with: pip install sentence-transformers")
    print("   Falling back to keyword matching only...")
    SEMANTIC_MODEL_AVAILABLE = False

# ── Load semantic encoder ─────────────────────────────────────────────────────
if SEMANTIC_MODEL_AVAILABLE:
    print("Loading semantic encoder (all-MiniLM-L6-v2)...")
    semantic_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"✅ Semantic encoder loaded | dim=384 | device={semantic_encoder.device}")
else:
    semantic_encoder = None

engine = PhysicsEngine()

# ── Feature extraction (for physics scoring only, NOT for semantic matching) ──
def extract_features(accel, gyro, hz):
    """Extract physics features for quality scoring (not for semantic matching!)"""
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

# ── Instruction keyword map (still useful as fallback) ────────────────────────
INSTRUCTION_KEYWORDS = {
    "pick_up": ["pick", "grab", "get", "take", "reach", "fetch", "retrieve", "cup", "object"],
    "stack": ["stack", "build", "tower", "block", "place", "pile", "assemble", "lift"],
    "fold": ["fold", "lay", "spread", "arrange", "laundry", "cloth", "flat"],
    "drink": ["drink", "sip", "water", "beverage", "liquid"],
    "pour": ["pour", "fill", "liquid", "container"],
    "push": ["push", "press", "button", "force"],
    "pull": ["pull", "drag", "tug"],
    "twist": ["twist", "turn", "rotate", "screw"],
    "hand_motion": ["hand", "finger", "grip", "grasp", "pinch"],
}

def keyword_similarity(query, tags):
    """Fallback keyword matching (works when semantic model unavailable)"""
    query_lower = query.lower()
    matches = sum(1 for tag in tags if tag.lower() in query_lower)
    return matches / max(len(tags), 1)

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
        
        # SEMANTIC FIX: Store natural language description instead of just tags
        description = "hand grasping and gripping motion for object manipulation"
        
        index.append({
            "source":      "ninapro_db5",
            "description": description,
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
                
                description = "prosthetic hand reaching and grasping objects"
                
                index.append({
                    "source":      "emg_amputee",
                    "description": description,
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
        
        # Use actual instruction as description
        description = instr if instr != "unknown" else "robotic manipulation task"
        
        index.append({
            "source":      "roboturk",
            "description": description,
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

# ── SEMANTIC FIX: Compute semantic embeddings ─────────────────────────────────
if SEMANTIC_MODEL_AVAILABLE:
    print("\n🔧 SEMANTIC FIX: Computing proper semantic embeddings...")
    print("   (Using sentence-transformers, NOT physics encoder)")
    
    # Extract all descriptions
    descriptions = [w["description"] for w in index]
    
    # Generate semantic embeddings (384-dim, trained for semantic similarity)
    semantic_embeddings = semantic_encoder.encode(
        descriptions,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Store embeddings
    for i, emb in enumerate(semantic_embeddings):
        index[i]["semantic_embedding"] = emb.tolist()
    
    print(f"✅ Semantic embeddings: {semantic_embeddings.shape}")
    print(f"   Model: all-MiniLM-L6-v2 (384-dim)")
    print(f"   Purpose: Semantic similarity (NOT physics quality)")
else:
    print("\n⚠️  Semantic embeddings not available (sentence-transformers not installed)")
    print("   Falling back to keyword matching only")
    semantic_embeddings = None

# ── Retrieval function ────────────────────────────────────────────────────────
def retrieve(query, top_k=10):
    """
    Given a text query, return top-K certified windows.
    
    SEMANTIC FIX:
    - Now uses proper semantic similarity (sentence-transformers)
    - "drink water" will NOT match "folding clothes" anymore
    - Falls back to keyword matching if semantic model unavailable
    
    Ranking: semantic_match(0.35) + physics_score(0.35) + smoothness(0.20) + tier_bonus(0.10)
    """
    tier_bonus = {"GOLD": 1.0, "SILVER": 0.66, "BRONZE": 0.33}
    
    # Compute semantic similarity if available
    if SEMANTIC_MODEL_AVAILABLE and semantic_embeddings is not None:
        # Encode query
        query_embedding = semantic_encoder.encode([query], convert_to_numpy=True)[0]
        
        # Compute cosine similarity with all windows
        norms = np.linalg.norm(semantic_embeddings, axis=1)
        query_norm = np.linalg.norm(query_embedding)
        similarities = np.dot(semantic_embeddings, query_embedding) / (norms * query_norm + 1e-10)
    else:
        similarities = None

    results = []
    for i, w in enumerate(index):
        # Semantic match
        if similarities is not None:
            semantic = float(similarities[i])
        else:
            # Fallback to keyword matching
            semantic = keyword_similarity(query, w.get("semantic_tags", []))

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
            "description": w["description"],
            "laws_passed": w.get("laws_passed", []),
            "pos_start":   w.get("pos_start", [0,0,0]),
        })

    results.sort(key=lambda x: x["rank_score"], reverse=True)
    return results[:top_k]

# ── Test retrieval ────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("RETRIEVAL TEST (SEMANTIC FIX VALIDATION)")
print("="*60)

test_queries = [
    "pick up the cup",
    "grab object from table",
    "stack blocks to build tower",
    "fold and arrange laundry",
    "drink water",  # CRITICAL TEST: Should NOT match folding anymore!
    "reach and grasp",
]

retrieval_results = {}
for query in test_queries:
    results = retrieve(query, top_k=5)
    retrieval_results[query] = results
    print(f"\nQuery: '{query}'")
    print(f"  {'Rank':>6} {'Score':>7} {'Sem':>5} {'Tier':>8} {'Description'}")
    for j, r in enumerate(results):
        print(f"  #{j+1:>4}  {r['rank_score']:>6.3f}  {r['semantic']:>5.3f}  "
              f"{r['tier']:>8}  {r['description'][:60]}")

# Validate the fix
if SEMANTIC_MODEL_AVAILABLE:
    print("\n" + "="*60)
    print("🔧 SEMANTIC FIX VALIDATION:")
    print("="*60)
    drink_results = retrieve("drink water", top_k=5)
    fold_in_results = any("fold" in r["description"].lower() for r in drink_results)
    
    if fold_in_results:
        print("❌ FAILED: 'drink water' still matches 'folding' (bug not fixed)")
    else:
        print("✅ PASSED: 'drink water' no longer matches 'folding clothes'")
        print("   Semantic embeddings working correctly!")

# ── Save index + results ──────────────────────────────────────────────────────
index_compact = [{k:v for k,v in w.items() if k != "semantic_embedding"} for w in index]

out_index = os.path.expanduser("~/S2S/experiments/retrieval_index.json")
out_results = os.path.expanduser("~/S2S/experiments/retrieval_results.json")

with open(out_index, "w") as f:
    json.dump({"total": len(index), "windows": index_compact}, f, indent=2)
with open(out_results, "w") as f:
    json.dump({"queries": retrieval_results}, f, indent=2)

# Save semantic embeddings separately
if SEMANTIC_MODEL_AVAILABLE:
    np.save(os.path.expanduser("~/S2S/experiments/semantic_embeddings.npy"), semantic_embeddings)
    print(f"\nSemantic embeddings → semantic_embeddings.npy {semantic_embeddings.shape}")

print(f"\n{'='*60}")
print(f"Index saved  → {out_index} ({os.path.getsize(out_index)/1024:.0f} KB)")
print(f"Results saved→ {out_results}")
print(f"\n✅ Ready for demo: {len(index)} certified windows, {len(test_queries)} queries tested")
print(f"✅ Semantic fix applied: Using sentence-transformers (NOT physics encoder)")
