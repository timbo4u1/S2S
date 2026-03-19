#!/usr/bin/env python3
"""
Step 3 v2 — Multi-Dataset Retrieval by Instruction
Datasets: WISDM + PAMAP2 + UCI HAR + NinaPro + EMG Amputee + RoboTurk
"""
import os, sys, json, glob, pickle, re
import numpy as np
sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
import torch, torch.nn as nn

# ── Model ─────────────────────────────────────────────────────────────────────
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
model  = DualHead(13,3,9).to(device)
model.load_state_dict(torch.load(
    os.path.expanduser("~/S2S/experiments/level5_corrected_best.pt"),
    map_location=device))
model.eval()
print(f"Model loaded | device={device}")
engine = PhysicsEngine()

# ── Activity → semantic tags ───────────────────────────────────────────────────
WISDM_LABELS = {
    "A": ("walking",        ["walk","step","stroll","move forward","locomotion"]),
    "B": ("jogging",        ["jog","run","sprint","fast walk","cardio"]),
    "C": ("stairs",         ["climb stairs","go up stairs","ascending","descending stairs"]),
    "D": ("sitting",        ["sit","seated","rest","chair","stationary"]),
    "E": ("standing",       ["stand","upright","idle","wait"]),
    "F": ("typing",         ["type","keyboard","finger","press","computer"]),
    "G": ("brushing teeth", ["brush","teeth","mouth","dental"]),
    "H": ("eating soup",    ["eat","soup","spoon","bowl","feed"]),
    "I": ("eating chips",   ["eat","chips","snack","hand to mouth","reach"]),
    "J": ("eating pasta",   ["eat","pasta","fork","meal","feed"]),
    "K": ("drinking",       ["drink","cup","glass","raise","reach","pick up cup","grab cup"]),
    "L": ("eating sandwich",["eat","sandwich","hand","bite","reach","grab","pick up"]),
    "M": ("kicking",        ["kick","leg","foot","strike"]),
    "O": ("catching",       ["catch","grab","reach","grasp","pick","retrieve"]),
    "P": ("dribbling",      ["dribble","ball","hand","bounce","basketball"]),
    "Q": ("writing",        ["write","pen","pencil","fine motor","hand"]),
    "R": ("clapping",       ["clap","hands","applaud","rhythmic"]),
    "S": ("folding",        ["fold","laundry","cloth","arrange","layout","organize"]),
}

PAMAP2_LABELS = {
    1:  ("lying",           ["lie down","rest","prone","supine","lay"]),
    2:  ("sitting",         ["sit","seated","chair","stationary","rest"]),
    3:  ("standing",        ["stand","upright","idle","wait"]),
    4:  ("walking",         ["walk","step","stroll","move forward","locomotion"]),
    5:  ("running",         ["run","jog","sprint","fast","cardio"]),
    6:  ("cycling",         ["cycle","bike","pedal","rotate"]),
    7:  ("nordic walking",  ["walk","poles","stride","outdoor"]),
    12: ("ascending stairs",["climb","stairs","up","ascend","step up"]),
    13: ("descending stairs",["descend","stairs","down","step down"]),
    16: ("vacuum cleaning", ["vacuum","clean","push","sweep","household"]),
    17: ("ironing",         ["iron","press","cloth","laundry","fold","arrange"]),
    24: ("rope jumping",    ["jump","rope","skip","cardio","bounce"]),
}

UCI_LABELS = {
    1: ("walking",            ["walk","step","stroll","move forward","locomotion"]),
    2: ("walking upstairs",   ["climb","stairs","up","ascend","step up"]),
    3: ("walking downstairs", ["descend","stairs","down","step down"]),
    4: ("sitting",            ["sit","seated","chair","stationary","rest"]),
    5: ("standing",           ["stand","upright","idle","wait"]),
    6: ("laying",             ["lie down","rest","prone","supine","lay"]),
}

ROBOTURK_LABELS = {
    "object search":  ["pick","grab","get","take","reach","find","search","fetch","retrieve","cup","object","pick up"],
    "create tower":   ["stack","build","tower","block","place on top","pile","assemble","lift","raise"],
    "layout laundry": ["lay","spread","arrange","fold","laundry","cloth","flat","place","organize","sort"],
}

# ── Feature extraction ─────────────────────────────────────────────────────────
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
        f.append(float(freqs[np.argmax(fft)]) if len(fft) > 0 else 0.0)
    c = np.corrcoef(accel[:,0], accel[:,1])[0,1]
    f.append(float(c) if not np.isnan(c) else 0.0)
    f.append(float(np.linalg.norm(np.mean(accel, axis=0))))
    hist, _ = np.histogram(accel.flatten(), bins=20)
    hist = hist / (hist.sum() + 1e-10)
    f.append(float(-np.sum(hist * np.log(hist + 1e-10))))
    return np.array(f[:13], dtype=np.float32)

def smoothness(accel, hz):
    if len(accel) < 3: return 0.0
    jerk = np.diff(accel, axis=0) * hz
    return float(1.0 / (1.0 + np.sqrt(np.mean(jerk**2) + 1e-8)))

def certify_window(accel, gyro, hz):
    ts = [int(i*1e9/hz) for i in range(len(accel))]
    try:
        r = engine.certify(
            imu_raw={"timestamps_ns": ts,
                     "accel": accel.tolist(),
                     "gyro":  gyro.tolist()},
            segment="forearm")
        return r
    except:
        return {"tier": "REJECTED", "physical_law_score": 0, "laws_passed": []}

index = []
stats = {}

# ── 1. WISDM ──────────────────────────────────────────────────────────────────
print("\n[1/6] WISDM...")
HZ = 20
WIN = 80  # 4 seconds
wisdm_count = 0
arff_dir = os.path.expanduser("~/S2S_Project/wisdm-dataset/arff_files/phone/accel/")
for arff_path in sorted(glob.glob(os.path.join(arff_dir,"*.arff")))[:50]:
    rows = []
    label = None
    with open(arff_path) as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.lower() == "@data":
                in_data = True; continue
            if not in_data or not line or line.startswith("%"): continue
            parts = line.split(",")
            if len(parts) < 4: continue
            try:
                lbl = parts[0].strip().strip("'\"")
                vals = [float(x) for x in parts[1:4]]
                rows.append((lbl, vals))
            except: continue
    if not rows: continue
    label = rows[0][0]
    data  = np.array([r[1] for r in rows], dtype=np.float32)
    if data.shape[0] < WIN: continue
    meta  = WISDM_LABELS.get(label)
    if not meta: continue
    activity_name, tags = meta
    for i in range(0, len(data)-WIN, WIN//2):
        ca = data[i:i+WIN]
        if ca.shape[1] < 3: continue
        cg = np.zeros_like(ca)
        r  = certify_window(ca, cg, HZ)
        if r.get("tier","REJECTED") == "REJECTED": continue
        index.append({
            "source": "wisdm",
            "instruction": activity_name,
            "semantic_tags": tags,
            "tier": r["tier"],
            "score": r.get("physical_law_score", 0),
            "smoothness": round(smoothness(ca, HZ), 4),
            "features": extract_features(ca, cg, HZ).tolist(),
            "laws_passed": r.get("laws_passed", []),
        })
        wisdm_count += 1
        if wisdm_count >= 400: break
    if wisdm_count >= 400: break
print(f"  → {wisdm_count} windows")
stats["wisdm"] = wisdm_count

# ── 2. PAMAP2 ─────────────────────────────────────────────────────────────────
print("[2/6] PAMAP2...")
HZ = 100
WIN = 200
pamap_count = 0
for dat_path in sorted(glob.glob(
        os.path.expanduser("~/S2S_Project/pamap2_final/PAMAP2_Dataset/Protocol/subject10*.dat"))):
    try:
        data = np.genfromtxt(dat_path, invalid_raise=False)
        if data.ndim < 2 or data.shape[1] < 20: continue
        act_ids = data[:, 1].astype(int)
        # Hand IMU cols 4-6 (accel) 7-9 (gyro)
        accel_all = data[:, 4:7]
        gyro_all  = data[:, 7:10]
        for act_id, (act_name, tags) in PAMAP2_LABELS.items():
            mask = act_ids == act_id
            if mask.sum() < WIN: continue
            idx  = np.where(mask)[0]
            segs = np.split(idx, np.where(np.diff(idx) > 5)[0]+1)
            for seg in segs:
                if len(seg) < WIN: continue
                ca = accel_all[seg[:WIN]]
                cg = gyro_all[seg[:WIN]]
                ca = np.nan_to_num(ca)
                cg = np.nan_to_num(cg)
                r  = certify_window(ca, cg, HZ)
                if r.get("tier","REJECTED") == "REJECTED": continue
                index.append({
                    "source": "pamap2",
                    "instruction": act_name,
                    "semantic_tags": tags,
                    "tier": r["tier"],
                    "score": r.get("physical_law_score", 0),
                    "smoothness": round(smoothness(ca, HZ), 4),
                    "features": extract_features(ca, cg, HZ).tolist(),
                    "laws_passed": r.get("laws_passed", []),
                })
                pamap_count += 1
    except Exception as e:
        print(f"  skip {dat_path}: {e}")
        continue
print(f"  → {pamap_count} windows")
stats["pamap2"] = pamap_count

# ── 3. UCI HAR ────────────────────────────────────────────────────────────────
print("[3/6] UCI HAR...")
HZ = 50
WIN = 128
uci_count = 0
uci_base = os.path.expanduser("~/S2S_Project/uci_har/UCI HAR Dataset/train/Inertial Signals/")
try:
    ax = np.loadtxt(os.path.join(uci_base,"total_acc_x_train.txt"))
    ay = np.loadtxt(os.path.join(uci_base,"total_acc_y_train.txt"))
    az = np.loadtxt(os.path.join(uci_base,"total_acc_z_train.txt"))
    gx = np.loadtxt(os.path.join(uci_base,"body_gyro_x_train.txt"))
    gy = np.loadtxt(os.path.join(uci_base,"body_gyro_y_train.txt"))
    gz = np.loadtxt(os.path.join(uci_base,"body_gyro_z_train.txt"))
    labels = np.loadtxt(os.path.expanduser(
        "~/S2S_Project/uci_har/UCI HAR Dataset/train/y_train.txt")).astype(int)
    for i in range(min(len(labels), 500)):
        act_id = labels[i]
        meta = UCI_LABELS.get(act_id)
        if not meta: continue
        act_name, tags = meta
        ca = np.stack([ax[i], ay[i], az[i]], axis=1).astype(np.float32)
        cg = np.stack([gx[i], gy[i], gz[i]], axis=1).astype(np.float32)
        r  = certify_window(ca, cg, HZ)
        if r.get("tier","REJECTED") == "REJECTED": continue
        index.append({
            "source": "uci_har",
            "instruction": act_name,
            "semantic_tags": tags,
            "tier": r["tier"],
            "score": r.get("physical_law_score", 0),
            "smoothness": round(smoothness(ca, HZ), 4),
            "features": extract_features(ca, cg, HZ).tolist(),
            "laws_passed": r.get("laws_passed", []),
        })
        uci_count += 1
except Exception as e:
    print(f"  UCI HAR error: {e}")
print(f"  → {uci_count} windows")
stats["uci_har"] = uci_count

# ── 4. NinaPro DB5 ────────────────────────────────────────────────────────────
print("[4/6] NinaPro DB5...")
import scipy.io
HZ = 2000; WIN = 2000
nin_count = 0
for subj_dir in sorted(glob.glob(os.path.expanduser("~/ninapro_db5/s*")))[:3]:
    accel_all = []
    for mat_path in sorted(glob.glob(os.path.join(subj_dir,"*.mat"))):
        try:
            mat = scipy.io.loadmat(mat_path)
            acc = mat.get("acc", mat.get("ACC"))
            if acc is not None: accel_all.append(acc)
        except: continue
    if not accel_all: continue
    accel = np.vstack(accel_all).astype(np.float64)
    for i in range(0, min(len(accel)-WIN, WIN*5), WIN):
        ca = accel[i:i+WIN,:3].astype(np.float32)
        cg = np.zeros_like(ca)
        r  = certify_window(ca, cg, HZ)
        if r.get("tier","REJECTED") == "REJECTED": continue
        index.append({
            "source": "ninapro_db5",
            "instruction": "grasping",
            "semantic_tags": ["grab","grip","grasp","pick","reach","hand","finger","catch","retrieve"],
            "tier": r["tier"],
            "score": r.get("physical_law_score", 0),
            "smoothness": round(smoothness(ca, HZ), 4),
            "features": extract_features(ca, cg, HZ).tolist(),
            "laws_passed": r.get("laws_passed", []),
        })
        nin_count += 1
print(f"  → {nin_count} windows")
stats["ninapro"] = nin_count

# ── 5. EMG Amputee ────────────────────────────────────────────────────────────
print("[5/6] EMG Amputee...")
HZ = 200; WIN = 200
amp_count = 0
for subj in sorted(os.listdir(os.path.expanduser("~/S2S_Project/EMG_Amputee")))[:3]:
    inner = os.path.expanduser(f"~/S2S_Project/EMG_Amputee/{subj}/{subj}")
    if not os.path.isdir(inner): continue
    for csv_path in sorted(glob.glob(os.path.join(inner,"mov *","accelerometer-*.csv")))[:5]:
        try:
            data  = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
            if data.ndim < 2 or data.shape[1] < 4: continue
            ca = data[:,1:4].astype(np.float32)
            cg = np.zeros_like(ca)
            if len(ca) < WIN: continue
            r  = certify_window(ca[:WIN], cg[:WIN], HZ)
            if r.get("tier","REJECTED") == "REJECTED": continue
            index.append({
                "source": "emg_amputee",
                "instruction": "prosthetic grasping",
                "semantic_tags": ["grab","grip","grasp","pick","reach","prosthetic","hand","finger","catch"],
                "tier": r["tier"],
                "score": r.get("physical_law_score", 0),
                "smoothness": round(smoothness(ca[:WIN], HZ), 4),
                "features": extract_features(ca[:WIN], cg[:WIN], HZ).tolist(),
                "laws_passed": r.get("laws_passed", []),
            })
            amp_count += 1
        except: continue
print(f"  → {amp_count} windows")
stats["emg_amputee"] = amp_count

# ── 6. RoboTurk ───────────────────────────────────────────────────────────────
print("[6/6] RoboTurk...")
HZ = 15; WIN = 30; NYU = 14
rt_count = 0
for path in sorted(glob.glob(os.path.expanduser(
        "~/S2S/openx_data/sample_*.data.pickle")))[NYU:]:
    try:
        with open(path,"rb") as f: d = pickle.load(f)
        steps = d.get("steps",[])
        instr = "unknown"
        for step in steps:
            if not isinstance(step,dict): continue
            raw = step.get("observation",{}).get("natural_language_instruction","")
            if raw:
                instr = raw.decode() if isinstance(raw,bytes) else str(raw)
                break
        if instr not in ROBOTURK_LABELS: continue
        tags = ROBOTURK_LABELS[instr]
        wvs, rds = [], []
        for step in steps:
            if not isinstance(step,dict): continue
            wv = step.get("action",{}).get("world_vector")
            rd = step.get("action",{}).get("rotation_delta")
            if wv is not None and rd is not None:
                wvs.append(np.array(wv,dtype=np.float32))
                rds.append(np.array(rd,dtype=np.float32))
        if len(wvs) < WIN+2: continue
        wvs = np.array(wvs); rds = np.array(rds)
        pos   = np.cumsum(wvs, axis=0)
        vel   = np.diff(pos, axis=0)*HZ
        accel = np.diff(vel, axis=0)*HZ
        rots  = np.cumsum(rds, axis=0)
        gv    = np.diff(rots, axis=0)*HZ
        gyro  = np.diff(gv, axis=0)*HZ
        if len(gyro) != len(accel): gyro = np.zeros_like(accel)
        for i in range(0, len(accel)-WIN, WIN//2):
            ca = accel[i:i+WIN]; cg = gyro[i:i+WIN]
            if len(ca) < WIN: continue
            r  = certify_window(ca, cg, HZ)
            if r.get("tier","REJECTED") == "REJECTED": continue
            index.append({
                "source": "roboturk",
                "instruction": instr,
                "semantic_tags": tags,
                "tier": r["tier"],
                "score": r.get("physical_law_score", 0),
                "smoothness": round(smoothness(ca, HZ), 4),
                "features": extract_features(ca, cg, HZ).tolist(),
                "laws_passed": r.get("laws_passed", []),
            })
            rt_count += 1
    except: continue
print(f"  → {rt_count} windows")
stats["roboturk"] = rt_count

print(f"\nTotal index: {len(index)} certified windows")
print("By dataset:", {k:v for k,v in stats.items()})

# ── Compute embeddings ─────────────────────────────────────────────────────────
print("\nComputing embeddings...")
all_feats = np.array([w["features"] for w in index], dtype=np.float32)
with torch.no_grad():
    embs = model.encode(torch.FloatTensor(all_feats).to(device)).cpu().numpy()
for i,e in enumerate(embs): index[i]["embedding"] = e.tolist()

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query, top_k=10):
    q = query.lower()
    q_words = set(re.findall(r'\w+', q))
    tier_bonus = {"GOLD":1.0,"SILVER":0.66,"BRONZE":0.33}
    results = []
    for i,w in enumerate(index):
        tags     = set(t.lower() for t in w.get("semantic_tags",[]))
        instr_w  = set(re.findall(r'\w+', w["instruction"].lower()))
        overlap  = len(q_words & (tags | instr_w))
        semantic = min(1.0, overlap / max(len(q_words), 1))
        phys     = min(1.0, w["score"] / 100.0)
        sm       = min(1.0, w["smoothness"] * 20)
        tb       = tier_bonus.get(w["tier"], 0.0)
        rank     = 0.35*semantic + 0.35*phys + 0.20*sm + 0.10*tb
        results.append({
            "rank_score":  round(rank, 4),
            "semantic":    round(semantic, 3),
            "physics":     w["score"],
            "smoothness":  w["smoothness"],
            "tier":        w["tier"],
            "source":      w["source"],
            "instruction": w["instruction"],
            "laws_passed": w.get("laws_passed",[]),
        })
    results.sort(key=lambda x: x["rank_score"], reverse=True)
    return results[:top_k]

# ── Test ──────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
queries = [
    "pick up the cup",
    "walk forward",
    "run fast",
    "fold laundry",
    "climb stairs",
    "grab and grasp object",
    "sit down",
    "reach and catch",
    "stack blocks build tower",
    "write on paper",
]
retrieval_results = {}
for query in queries:
    res = retrieve(query, top_k=5)
    retrieval_results[query] = res
    print(f"\n'{query}'")
    print(f"  {'#':>2} {'Score':>6} {'Tier':>8} {'Source':>12}  Instruction")
    for j,r in enumerate(res):
        print(f"  #{j+1} {r['rank_score']:>6.3f}  {r['tier']:>8}  {r['source']:>12}  {r['instruction']}")

# ── Save ──────────────────────────────────────────────────────────────────────
compact = [{k:v for k,v in w.items() if k!="embedding"} for w in index]
with open(os.path.expanduser("~/S2S/experiments/retrieval_index_v2.json"),"w") as f:
    json.dump({"total":len(index),"stats":stats,"windows":compact},f,indent=2)
with open(os.path.expanduser("~/S2S/experiments/retrieval_results_v2.json"),"w") as f:
    json.dump({"queries":retrieval_results},f,indent=2)
np.save(os.path.expanduser("~/S2S/experiments/retrieval_embeddings_v2.npy"), embs)

print(f"\n{'='*65}")
print(f"Index: {len(index)} windows across {len(stats)} datasets")
for ds,n in stats.items(): print(f"  {ds:>14}: {n}")
print(f"\nFiles saved to ~/S2S/experiments/")
print(f"\nNow type any command — try it interactively:")
print(f"  python3 -c \"")
print(f"  import json")
print(f"  # Load and query\"")

# ── Interactive mode ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("INTERACTIVE — type a command, see results. Ctrl+C to exit.")
print("="*65)
while True:
    try:
        q = input("\n> ").strip()
        if not q: continue
        res = retrieve(q, top_k=8)
        print(f"\n  Top matches for '{q}':")
        print(f"  {'#':>2} {'Score':>6} {'Tier':>8} {'Source':>14}  Instruction")
        print(f"  {'-'*60}")
        for j,r in enumerate(res):
            print(f"  #{j+1} {r['rank_score']:>6.3f}  {r['tier']:>8}  "
                  f"{r['source']:>14}  {r['instruction']}")
    except KeyboardInterrupt:
        print("\nDone.")
        break
# PATCH — run this to diagnose
