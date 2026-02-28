#!/usr/bin/env python3
"""
s2s_dataset_adapter.py  —  S2S v1.3
Convert public motion datasets → certified S2S records

DATASETS SUPPORTED:
  uci_har    UCI HAR Dataset (30 subjects, 6 activities, accel+gyro 50Hz)
  pamap2     PAMAP2 (9 subjects, 18 activities, wrist IMU 100Hz)
  berkeley   Berkeley MHAD (12 subjects, 11 actions, IMU+MoCap)
  movi       MoVi (90 subjects, manipulation actions, 9-DOF IMU)
  custom     Any CSV with accel+gyro columns

DOWNLOAD:
  UCI HAR:   wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
  PAMAP2:    wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip"
  Berkeley:  https://tele-immersion.citris-uc.org/berkeley_mhad
  MoVi:      https://dmrepo.ict.usc.edu/dataset/movi/ (registration)

USAGE:
  python3 s2s_dataset_adapter.py --dataset uci_har --input "UCI HAR Dataset/" --out s2s_dataset/
  python3 s2s_dataset_adapter.py --dataset pamap2  --input PAMAP2_Dataset/    --out s2s_dataset/
  python3 s2s_dataset_adapter.py --taxonomy        # show full motion taxonomy
"""

import sys, os, json, csv, math, glob, argparse, random
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════
#  MOTION DOMAIN TAXONOMY
#  Based on: Flash-Hogan 1985, Bernstein 1967, Fitts 1954,
#            Wolpert 1998 (MOSAIC model), Shadmehr & Mussa-Ivaldi 1994
# ═══════════════════════════════════════════════════════════════════
MOTION_DOMAINS = {
    "PRECISION": {
        "jerk_max_ms3":    80,
        "coupling_r_min":  0.30,
        "description":     "Fine motor — surgery, assembly, prosthetics, writing",
        "robot_use":       "Surgical robots, prosthetic hands, PCB assembly arms",
        "science":         "Fitts Law: MT = a + b·log2(2D/W). Low jerk = high endpoint accuracy.",
        "actions": ["point","write","type","pinch","thread","reach_precise",
                    "grasp_small","solder","suture","turn_screw","pick_up"],
    },
    "POWER": {
        "jerk_max_ms3":    200,
        "coupling_r_min":  0.30,
        "description":     "Force generation — lifting, pushing, industrial manipulation",
        "robot_use":       "Warehouse robots, exoskeletons, industrial arms, bipeds",
        "science":         "Bernstein synergies: arm coordinates as rigid unit. High coupling r.",
        "actions": ["grasp","lift","push","pull","carry","throw","punch",
                    "kick","open_jar","turn_valve","hammer","dig"],
    },
    "SOCIAL": {
        "jerk_max_ms3":    180,
        "coupling_r_min":  0.15,
        "description":     "Communication gestures — HRI, service robots",
        "robot_use":       "Social robots, assistants, HRI (human-robot interaction)",
        "science":         "McNeill 1992: gestures co-produced with speech. Temporal not spatial accuracy.",
        "actions": ["wave","nod","shake_head","beckon","clap","thumbs_up",
                    "point_social","shrug","facepalm"],
    },
    "LOCOMOTION": {
        "jerk_max_ms3":    300,
        "coupling_r_min":  0.15,
        "description":     "Whole-body movement — walking, running, climbing",
        "robot_use":       "Bipedal robots, exoskeletons, prosthetic legs, quadrupeds",
        "science":         "BCG: heart stroke couples to wrist at heel-strike. Periodic = Fourier-checkable.",
        "actions": ["walk","run","jog","climb_stairs","descend_stairs","jump",
                    "squat","sit_down","stand_up","bend","nordic_walking","cycle"],
    },
    "DAILY_LIVING": {
        "jerk_max_ms3":    150,
        "coupling_r_min":  0.20,
        "description":     "ADL — cooking, cleaning, self-care, home tasks",
        "robot_use":       "Home robots, elder care, rehabilitation, domestic helpers",
        "science":         "Mixed precision+power per task phase. Flash-Hogan minimum jerk within sub-moves.",
        "actions": ["pour_liquid","stir","chop","open_door","wash_hands",
                    "brush_teeth","fold_clothes","vacuum","iron","eat","drink",
                    "use_phone","computer_work","car_driving","watch_tv"],
    },
    "SPORT": {
        "jerk_max_ms3":    500,
        "coupling_r_min":  0.10,
        "description":     "Athletic motion — maximum speed and force",
        "robot_use":       "Sports training robots, motion analysis, athletic exoskeletons",
        "science":         "Maximum jerk budget — motor cortex at limits. Explosive = high jerk.",
        "actions": ["tennis_swing","golf_swing","football_kick","basketball_throw",
                    "swim_stroke","row","jump_rope","play_soccer","rope_jumping"],
    },
}

# Flat action → domain lookup
ACTION_DOMAIN = {}
for dom, info in MOTION_DOMAINS.items():
    for act in info["actions"]:
        ACTION_DOMAIN[act] = dom

ACTION_DOMAIN.update({
    "laying": "DAILY_LIVING",
    "lying": "DAILY_LIVING",
    "sitting": "DAILY_LIVING",
    "standing": "DAILY_LIVING",
    "house_cleaning": "DAILY_LIVING",
    "sit_down": "LOCOMOTION",
    "stand_up": "LOCOMOTION",
})

# Fix missing labels from UCI HAR and PAMAP2 datasets
ACTION_DOMAIN.update({
    "laying":          "DAILY_LIVING",   # UCI HAR: lying down
    "lying":           "DAILY_LIVING",   # PAMAP2: lying down
    "sitting":         "DAILY_LIVING",   # PAMAP2: seated posture
    "standing":        "DAILY_LIVING",   # PAMAP2: upright posture
    "house_cleaning":  "DAILY_LIVING",   # PAMAP2: general cleaning
    "sit_down":        "LOCOMOTION",     # UCI HAR: transition
    "stand_up":        "LOCOMOTION",     # UCI HAR: transition
})

# ═══════════════════════════════════════════════════════════════════
#  DATASET LABEL MAPS
# ═══════════════════════════════════════════════════════════════════
UCI_HAR_LABELS = {
    1:"walk", 2:"climb_stairs", 3:"descend_stairs",
    4:"sit_down", 5:"stand_up", 6:"laying"
}

PAMAP2_LABELS = {
    1:"lying", 2:"sitting", 3:"standing", 4:"walk", 5:"run", 6:"cycle",
    7:"nordic_walking", 9:"watch_tv", 10:"computer_work", 11:"car_driving",
    12:"climb_stairs", 13:"descend_stairs", 16:"vacuum", 17:"iron",
    18:"fold_clothes", 19:"house_cleaning", 20:"play_soccer", 24:"jump_rope"
}

BERKELEY_LABELS = {
    1:"jump", 2:"jumping_jacks", 3:"bend", 4:"punch", 5:"wave_two_hands",
    6:"wave_one_hand", 7:"clap", 8:"throw", 9:"sit_down", 10:"sit_up", 11:"walk"
}

# ═══════════════════════════════════════════════════════════════════
#  LOADERS — one per dataset
# ═══════════════════════════════════════════════════════════════════
def load_uci_har(input_dir):
    """
    UCI HAR: train/ and test/ each contain:
      Inertial Signals/body_acc_{x,y,z}_{split}.txt  — 128-sample windows, m/s²
      Inertial Signals/body_gyro_{x,y,z}_{split}.txt — 128-sample windows, rad/s
      y_{split}.txt  — integer activity label per window
      subject_{split}.txt — subject ID per window
    Sample rate: 50 Hz. Window: 128 samples = 2.56s
    """
    segments = []
    DT_NS = int(1/50 * 1e9)
    BASE  = 1_700_000_000_000_000_000

    for split in ['train', 'test']:
        sig_dir = os.path.join(input_dir, split, 'Inertial Signals')
        if not os.path.isdir(sig_dir):
            print(f"  ! Missing {sig_dir} — skipping {split} raw signals")
            continue

        def load_sig(name):
            p = os.path.join(sig_dir, f'{name}_{split}.txt')
            with open(p) as f:
                return [[float(v) for v in line.split()] for line in f]

        try:
            bax=load_sig('body_acc_x'); bay=load_sig('body_acc_y'); baz=load_sig('body_acc_z')
            bgx=load_sig('body_gyro_x');bgy=load_sig('body_gyro_y');bgz=load_sig('body_gyro_z')
        except FileNotFoundError as e:
            print(f"  ! {e}"); continue

        y_file   = os.path.join(input_dir, split, f'y_{split}.txt')
        sub_file = os.path.join(input_dir, split, f'subject_{split}.txt')
        with open(y_file)   as f: labels   = [int(l) for l in f]
        with open(sub_file) as f: subjects = [int(l) for l in f]

        for wi, (lbl, subj) in enumerate(zip(labels, subjects)):
            ns = len(bax[wi])
            ts = [BASE + wi*ns*DT_NS + i*DT_NS for i in range(ns)]
            segments.append({
                "action":   UCI_HAR_LABELS.get(lbl, f"act_{lbl}"),
                "person":   f"uci_s{subj:02d}",
                "dataset":  "UCI_HAR",
                "ts": ts,
                "accel": [[bax[wi][i]*9.81, bay[wi][i]*9.81, baz[wi][i]*9.81] for i in range(ns)],
                "gyro":  [[bgx[wi][i],      bgy[wi][i],      bgz[wi][i]]      for i in range(ns)],
            })

    print(f"  UCI HAR: {len(segments)} windows loaded")
    return segments


def load_pamap2(input_dir):
    """
    PAMAP2: Protocol/subject10{1-9}.dat
    Columns (space-separated):
      0=timestamp(s), 1=activityID, 2=heartrate
      Hand IMU (cols 3-19): temp, accel1_xyz, accel2_xyz, gyro_xyz, mag_xyz, orient_xyzw
        accel1 (g-units):  cols 4,5,6  → multiply by 9.81
        gyro   (rad/s):    cols 10,11,12
    Sample rate: 100 Hz
    """
    segments = []
    BASE = 1_700_000_000_000_000_000

    files = (glob.glob(os.path.join(input_dir, 'Protocol', 'subject*.dat')) or
             glob.glob(os.path.join(input_dir, 'subject*.dat')))
    if not files:
        print(f"  ! No .dat files in {input_dir}"); return segments

    for fpath in sorted(files):
        sid = os.path.basename(fpath).replace('subject','').replace('.dat','')
        print(f"  Loading {os.path.basename(fpath)}...", end=' ', flush=True)

        rows = []
        with open(fpath) as f:
            for line in f:
                vals = line.split()
                if len(vals) < 13: continue
                try:
                    row = [float(v) if v!='NaN' else float('nan') for v in vals]
                    rows.append(row)
                except: continue

        # Group consecutive rows with same activityID
        cur_act, cur_rows = None, []
        def flush_seg():
            if not cur_rows or cur_act is None or cur_act==0: return
            n = len(cur_rows)
            if n < 100: return
            ts = [BASE + int(cur_rows[j][0]*1e9) for j in range(n)]
            def safe(r, i): return r[i] if not math.isnan(r[i]) else 0.0
            accel = [[safe(cur_rows[j],4)*9.81, safe(cur_rows[j],5)*9.81, safe(cur_rows[j],6)*9.81] for j in range(n)]
            gyro  = [[safe(cur_rows[j],10),     safe(cur_rows[j],11),     safe(cur_rows[j],12)]     for j in range(n)]
            segments.append({
                "action": PAMAP2_LABELS.get(cur_act, f"act_{cur_act}"),
                "person": f"pamap2_s{sid}",
                "dataset": "PAMAP2",
                "ts": ts, "accel": accel, "gyro": gyro,
            })

        for row in rows:
            act = int(row[1]) if not math.isnan(row[1]) else 0
            if act != cur_act:
                flush_seg()
                cur_act, cur_rows = act, []
            cur_rows.append(row)
        flush_seg()
        print(f"→ {len(segments)} segs so far")

    print(f"  PAMAP2: {len(segments)} segments loaded")
    return segments


def load_berkeley(input_dir):
    """
    Berkeley MHAD IMU data: accel CSV files per subject/action/trial
    Expected structure: Accelerometer/S{subj}/A{act}/T{trial}/accData.csv
    Format: row per sample, columns = sensor channels
    """
    segments = []
    BASE = 1_700_000_000_000_000_000
    DT_NS = int(1/100*1e9)  # assume 100Hz

    pat = os.path.join(input_dir, 'Accelerometer', 'S*', 'A*', 'T*', '*.csv')
    files = glob.glob(pat)
    if not files:
        # Try flat structure
        files = glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True)

    for fpath in sorted(files)[:500]:  # cap at 500 files
        try:
            parts = fpath.replace('\\','/').split('/')
            subj = next((p for p in parts if p.startswith('S')), 'S00')[1:]
            act  = next((p for p in parts if p.startswith('A')), 'A00')[1:]
            lbl  = BERKELEY_LABELS.get(int(act), f"act_{act}")
        except: lbl='unknown'; subj='00'

        rows = []
        with open(fpath) as f:
            for line in f:
                vals = [float(v) for v in line.replace(',',' ').split() if v.strip()]
                if len(vals) >= 3: rows.append(vals[:3])
        if len(rows) < 20: continue

        n  = len(rows)
        ts = [BASE + i*DT_NS for i in range(n)]
        segments.append({
            "action": lbl, "person": f"berk_s{subj}", "dataset": "Berkeley_MHAD",
            "ts": ts,
            "accel": [[r[0], r[1], r[2]] for r in rows],
            "gyro":  [[0.0, 0.0, 0.0]] * n,   # Berkeley accel-only version
        })

    print(f"  Berkeley MHAD: {len(segments)} segments loaded")
    return segments


# ═══════════════════════════════════════════════════════════════════
#  PHYSICS DOMAIN CLASSIFIER
# ═══════════════════════════════════════════════════════════════════
def classify_domain(action, jerk_p95, coupling_r):
    """Label → domain first, physics as fallback."""
    for key, dom in ACTION_DOMAIN.items():
        if key in (action or '').lower():
            return dom
    # Physics fallback
    j = jerk_p95 or 100
    r = coupling_r or 0
    if j < 80  and r > 0.30: return "PRECISION"
    if j < 200 and r > 0.25: return "POWER"
    if j < 180:               return "DAILY_LIVING"
    if j < 300:               return "LOCOMOTION"
    return "SPORT"


# ═══════════════════════════════════════════════════════════════════
#  CORE CERTIFIER
# ═══════════════════════════════════════════════════════════════════
def certify_and_save(seg, out_dir, signer, counters):
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine, check_rigid_body, check_jerk
    from s2s_standard_v1_3.s2s_stream_certify_v1_3 import StreamCertifier

    imu_raw = {"timestamps_ns": seg["ts"], "accel": seg["accel"], "gyro": seg["gyro"]}
    n = len(seg["ts"])

    try:
        phys       = PhysicsEngine().certify(imu_raw=imu_raw, segment="forearm")
        _, _, d_rb = check_rigid_body(imu_raw)
        _, _, d_jk = check_jerk(imu_raw)
    except Exception as e:
        return None, f"physics_error: {e}"

    if phys['tier'] == 'REJECTED':
        return None, "REJECTED"

    # IMU stream cert
    win = min(256, n // 2)
    if win >= 32:
        sc = StreamCertifier(['ax','ay','az','gx','gy','gz'], window=win, step=win)
        imu_cert = None
        for i in range(n):
            r = sc.push_frame(seg["ts"][i], seg["accel"][i] + seg["gyro"][i])
            if r: imu_cert = r
        imu_tier = imu_cert['tier'] if imu_cert else 'UNRATED'
    else:
        imu_tier = 'UNRATED'

    jerk     = d_jk.get('p95_jerk_ms3')
    coupling = d_rb.get('pearson_r_measured_vs_predicted')
    action   = seg["action"]
    domain   = classify_domain(action, jerk, coupling)
    dur      = (seg["ts"][-1] - seg["ts"][0]) / 1e9

    record = {
        "schema":              "s2s_action_record_v1",
        "action":              action,
        "domain":              domain,
        "domain_description":  MOTION_DOMAINS[domain]["description"],
        "robot_use":           MOTION_DOMAINS[domain]["robot_use"],
        "person_id":           seg["person"],
        "dataset_source":      seg["dataset"],
        "duration_s":          round(dur, 3),
        "n_samples":           n,
        "imu_tier":            imu_tier,
        "physics_tier":        phys['tier'],
        "physics_score":       phys['physical_law_score'],
        "physics_laws_passed": phys['laws_passed'],
        "physics_laws_failed": phys['laws_failed'],
        "jerk_p95_ms3":        jerk,
        "imu_coupling_r":      coupling,
        # Strip raw sensor data to save space (keep metadata only)
        # Uncomment below to include raw data:
        # "sensor_data": imu_raw,
    }

    signed = signer.sign_cert(record)

    # Save: out_dir/DOMAIN/action/action_person_NNNN.json
    action_dir = os.path.join(out_dir, domain, action)
    os.makedirs(action_dir, exist_ok=True)
    key = f"{action}_{seg['person']}"
    counters[key] = counters.get(key, 0) + 1
    fname = f"{action}_{seg['person']}_{counters[key]:04d}.json"
    with open(os.path.join(action_dir, fname), 'w') as f:
        json.dump(signed, f, separators=(',',':'))  # compact

    return fname, "OK"


# ═══════════════════════════════════════════════════════════════════
#  MAIN CONVERTER
# ═══════════════════════════════════════════════════════════════════
def run_conversion(dataset, input_dir, out_dir, max_segs=None):
    from s2s_standard_v1_3.s2s_signing_v1_3 import CertSigner
    signer, _ = CertSigner.generate()

    print(f"\nConverting {dataset} → {out_dir}")
    print("─"*50)

    if   dataset == 'uci_har':   segments = load_uci_har(input_dir)
    elif dataset == 'pamap2':    segments = load_pamap2(input_dir)
    elif dataset == 'berkeley':  segments = load_berkeley(input_dir)
    else:
        print(f"Unknown dataset: {dataset}"); return

    if max_segs: segments = segments[:max_segs]
    total = len(segments)
    print(f"\nCertifying {total} segments...")

    ok = err = rej = 0
    counters = {}
    by_domain = {}
    by_action = {}

    for i, seg in enumerate(segments):
        if i % 200 == 0 and i > 0:
            print(f"  {i}/{total}  ok={ok} rej={rej} err={err}")
        fname, status = certify_and_save(seg, out_dir, signer, counters)
        if status == "OK":
            ok += 1
            rec_path = os.path.join(out_dir,
                classify_domain(seg['action'], None, None), seg['action'],
                fname)
            # Read back to get domain
            try:
                with open(rec_path) as f: rec = json.load(f)
                dom = rec.get('domain','?')
                act = rec.get('action','?')
                by_domain[dom] = by_domain.get(dom,0) + 1
                by_action[act] = by_action.get(act,0) + 1
            except: pass
        elif status == "REJECTED": rej += 1
        else: err += 1

    print(f"\n{'='*50}")
    print(f"  DONE — {dataset}")
    print(f"  Certified: {ok}  Rejected: {rej}  Errors: {err}")
    print(f"\n  By Domain:")
    for d,c in sorted(by_domain.items()): print(f"    {d:<15} {c:>6}")
    print(f"\n  By Action (top 10):")
    for a,c in sorted(by_action.items(), key=lambda x:-x[1])[:10]:
        print(f"    {a:<25} {c:>5}")
    print(f"{'='*50}")
    return {"ok":ok,"rej":rej,"err":err,"by_domain":by_domain}


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════
def print_taxonomy():
    print("\nS2S MOTION DOMAIN TAXONOMY — Physical AI Training")
    print("Based on: Flash-Hogan 1985, Bernstein 1967, Fitts 1954, Wolpert 1998")
    print("="*65)
    for dom, info in MOTION_DOMAINS.items():
        print(f"\n  ▶ {dom}")
        print(f"    {info['description']}")
        print(f"    Jerk ≤ {info['jerk_max_ms3']} m/s³  |  Coupling r ≥ {info['coupling_r_min']}")
        print(f"    Science: {info['science']}")
        print(f"    Robot:   {info['robot_use']}")
        acts = ', '.join(info['actions'][:6]) + ('...' if len(info['actions'])>6 else '')
        print(f"    Actions: {acts}")
    print()

def main():
    p = argparse.ArgumentParser(description='S2S Dataset Adapter')
    p.add_argument('--dataset',  choices=['uci_har','pamap2','berkeley','movi'])
    p.add_argument('--input',    help='Dataset root directory')
    p.add_argument('--out',      default='s2s_dataset/', help='Output directory')
    p.add_argument('--max',      type=int, help='Max segments (for testing)')
    p.add_argument('--taxonomy', action='store_true', help='Print motion taxonomy and exit')
    args = p.parse_args()

    if args.taxonomy:
        print_taxonomy(); return

    if not args.dataset or not args.input:
        p.print_help(); return

    run_conversion(args.dataset, args.input, args.out, args.max)

if __name__ == '__main__':
    main()
