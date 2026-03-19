"""
S2S Auto Router v2 — Full dataset coverage.
Fingerprints every format you have and routes to the correct pipeline head.

Fingerprint priority:
  1. File extension + directory structure
  2. Filename keywords (emg-, accelerometer-, gyro-)
  3. CSV header columns
  4. .mat metadata (Double Myo override)
  5. Amplitude + channel count fallback
"""

import os, sys, glob, json, csv
import numpy as np
import scipy.io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
W = "\033[97m"; D = "\033[2m";  X = "\033[0m"; C = "\033[96m"

HEADS = {
    "GESTURE": {
        "description": "Hand gesture / prosthetic control",
        "segment":     "forearm",
        "use_cases":   "Prosthetics, exoskeletons, gesture recognition",
    },
    "ACTIVITY": {
        "description": "Human activity recognition / rehabilitation",
        "segment":     "wrist",
        "use_cases":   "Rehab monitoring, fall detection, HAR",
    },
    "CARDIAC": {
        "description": "Cardiac / stress / HRV monitoring",
        "segment":     "wrist",
        "use_cases":   "Stress detection, HRV analysis, sleep staging",
    },
    "LOCOMOTION": {
        "description": "Walking / running / phone IMU",
        "segment":     "ankle",
        "use_cases":   "Step counting, gait analysis, phone apps",
    },
    "CERTIFIED": {
        "description": "Already S2S-certified — skip intake",
        "segment":     "n/a",
        "use_cases":   "Pre-certified JSON output",
    },
    "SKIP": {
        "description": "Binary/compressed — cannot auto-parse",
        "segment":     "n/a",
        "use_cases":   "Requires format-specific parser",
    },
    "GENERIC": {
        "description": "Unknown — certify and flag",
        "segment":     "forearm",
        "use_cases":   "Fallback",
    },
}

# ── CSV header fingerprints ───────────────────────────────────────────────────
CSV_FINGERPRINTS = [
    # NinaproDB1: emg_0..9, exercise, stimulus
    {
        "name":    "NinaproDB1",
        "require": ["emg_0", "emg_1", "exercise"],
        "head":    "GESTURE",
        "hz":      100,
        "note":    "NinaproDB1 — 10ch EMG, normalized float, 100Hz",
    },
    # EMG_Amputee emg file: timestamp, emg1..8
    {
        "name":    "EMG_Amputee_EMG",
        "require": ["timestamp", "emg1", "emg2"],
        "head":    "GESTURE",
        "hz":      200,
        "note":    "EMG Amputee — 8ch EMG, Myo armband, 200Hz",
    },
    # EMG_Gestures: time, channel1..8, class
    {
        "name":    "EMG_Gestures",
        "require": ["time", "channel1", "channel2", "class"],
        "head":    "GESTURE",
        "hz":      100,
        "note":    "EMG Gestures dataset — 8ch EMG, 100Hz",
    },
    # EMG_Amputee accelerometer: timestamp, x, y, z  (or acc_x etc)
    {
        "name":    "EMG_Amputee_Accel",
        "require": ["timestamp"],
        "any_of":  ["x", "y", "z", "acc_x", "acceleration_x"],
        "head":    "ACTIVITY",
        "hz":      50,
        "note":    "EMG Amputee accelerometer — 3-axis, 50Hz",
    },
    # EMG_Amputee gyro
    {
        "name":    "EMG_Amputee_Gyro",
        "require": ["timestamp"],
        "any_of":  ["gyro_x", "gyroscope_x", "wx", "wy"],
        "head":    "ACTIVITY",
        "hz":      50,
        "note":    "EMG Amputee gyro — 3-axis, 50Hz",
    },
]

def read_csv_header(path):
    try:
        with open(path, 'r', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader, [])
            # Skip BOM
            if header and header[0].startswith('\ufeff'):
                header[0] = header[0][1:]
            return [h.strip().lower() for h in header]
    except:
        return []

def fingerprint_csv(path):
    fname    = os.path.basename(path).lower()
    header   = read_csv_header(path)
    header_s = set(header)

    # Filename-first fingerprinting (EMG_Amputee sensors named by file)
    if fname.startswith('emg-'):
        return "GESTURE", 200, "EMG_Amputee_EMG", "filename=emg-*"
    if fname.startswith('accelerometer-'):
        return "ACTIVITY", 50, "EMG_Amputee_Accel", "filename=accelerometer-*"
    if fname.startswith('gyro-'):
        return "ACTIVITY", 50, "EMG_Amputee_Gyro", "filename=gyro-*"
    if fname.startswith('orientation'):
        return "SKIP", 0, "Orientation", "orientation data — not used for certification"

    # Header fingerprinting
    for fp in CSV_FINGERPRINTS:
        req = fp.get("require", [])
        any_of = fp.get("any_of", [])
        if all(r in header_s for r in req):
            if not any_of or any(a in header_s for a in any_of):
                return fp["head"], fp["hz"], fp["name"], fp["note"]

    # Fallback: count numeric columns
    n_cols = len(header)
    if n_cols >= 8:
        return "GESTURE",    100, "Unknown_EMG",   f"{n_cols}-column CSV — likely EMG"
    if 3 <= n_cols <= 6:
        return "ACTIVITY",    50, "Unknown_IMU",   f"{n_cols}-column CSV — likely IMU"
    return "GENERIC", 0, "Unknown", f"header={header[:5]}"

def fingerprint_mat(path):
    try:
        mat = scipy.io.loadmat(path)
        keys = {k.lower(): v for k, v in mat.items() if not k.startswith('_')}
        sensor_field = str(mat.get('sensor', ''))
        has_emg  = any('emg' in k for k in keys)
        has_acc  = any(k in ('acc','accel','acceleration') for k in keys)
        has_gyro = any('gyro' in k or 'gyr' in k for k in keys)
        freq_val = float(mat['frequency'].flat[0]) if 'frequency' in mat else 0

        # NinaproDB5 Double Myo override
        if 'Myo' in sensor_field and has_emg:
            return "GESTURE", 2000, "NinaproDB5", \
                   f"Double Myo EMG 16ch — metadata Hz={freq_val} overridden to 2000"

        # Generic mat routing
        if has_emg:
            return "GESTURE", int(freq_val) or 1000, "MAT_EMG", "EMG in mat file"
        if has_gyro and has_acc:
            return "ACTIVITY", int(freq_val) or 100, "MAT_IMU", "Accel+Gyro mat"
        if has_acc:
            return "LOCOMOTION", int(freq_val) or 50, "MAT_Accel", "Accel-only mat"
        return "GENERIC", 0, "MAT_Unknown", f"keys={list(keys.keys())[:5]}"
    except Exception as e:
        return "SKIP", 0, "MAT_Error", str(e)

def fingerprint_npy(path):
    try:
        arr  = np.load(path)
        cols = arr.shape[1] if arr.ndim > 1 else 1
        if cols > 20:
            return "CARDIAC", 500, "PTT_PPG", \
                   f"Pre-windowed {cols}-col npy — PTT-PPG wrist cardiac"
        if cols == 3:
            return "LOCOMOTION", 50, "NPY_Accel3", "3-axis accel npy"
        return "GENERIC", 0, "NPY_Unknown", f"{cols} columns"
    except Exception as e:
        return "SKIP", 0, "NPY_Error", str(e)

def fingerprint_json(path):
    try:
        with open(path) as f:
            d = json.load(f)
        if d.get("schema", "").startswith("s2s_"):
            return "CERTIFIED", 0, "S2S_JSON", \
                   f"Already certified — action={d.get('action')} tier={d.get('imu_tier')}"
        return "GENERIC", 0, "JSON_Unknown", "JSON without S2S schema"
    except:
        return "SKIP", 0, "JSON_Error", "Cannot parse JSON"

def fingerprint_dat(path):
    # PAMAP2 .dat — binary format, need official column spec
    return "SKIP", 0, "PAMAP2_DAT", \
           "PAMAP2 binary — use s2s_dataset JSON (already certified)"

def route_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':   return fingerprint_csv(path)
    if ext == '.mat':   return fingerprint_mat(path)
    if ext == '.npy':   return fingerprint_npy(path)
    if ext == '.json':  return fingerprint_json(path)
    if ext == '.dat':   return fingerprint_dat(path)
    return "GENERIC", 0, "Unknown_ext", ext

# ── Scan and report ───────────────────────────────────────────────────────────
def scan_directory(root, max_per_type=3):
    """Scan root, fingerprint one representative file per unique (dataset, head) pair."""
    all_files = []
    for ext in ['*.csv', '*.mat', '*.npy', '*.json', '*.dat']:
        all_files.extend(glob.glob(os.path.join(root, '**', ext), recursive=True))

    seen_types = {}
    results    = []
    skip_count = 0

    for path in sorted(all_files):
        head, hz, dataset, note = route_file(path)
        key = (dataset, head)

        if head == "SKIP":
            skip_count += 1
            continue

        if key not in seen_types:
            seen_types[key] = 0
        if seen_types[key] >= max_per_type:
            continue
        seen_types[key] += 1

        results.append({
            "path":    path,
            "file":    os.path.relpath(path, root),
            "head":    head,
            "hz":      hz,
            "dataset": dataset,
            "note":    note,
        })

    return results, skip_count, len(all_files)

# ── Main ──────────────────────────────────────────────────────────────────────
ROOTS = [
    ("NinaproDB5",    os.path.expanduser("~/ninapro_db5")),
    ("EMG_Amputee",   os.path.expanduser("~/S2S_Project/EMG_Amputee")),
    ("EMG_Gestures",  os.path.expanduser("~/S2S_Project/EMG_Gestures")),
    ("NinaproDB1",    os.path.expanduser("~/S2S_Project/NinaproDB1")),
    ("S2S_data",      os.path.expanduser("~/S2S/data")),
    ("S2S_dataset",   os.path.expanduser("~/S2S/s2s_dataset")),
]

print(f"\n{W}{'═'*62}")
print(f"  S2S AUTO ROUTER v2 — Full Dataset Coverage")
print(f"{'═'*62}{X}")

all_results  = []
summary      = {}

for label, root in ROOTS:
    if not os.path.exists(root):
        continue
    results, skipped, total = scan_directory(root, max_per_type=2)
    if not results and skipped == 0:
        continue

    print(f"\n{W}{label}{X} {D}({root}){X}")
    print(f"  {total} files found, {skipped} skipped (binary/orientation)")

    seen = set()
    for r in results:
        key = r['dataset']
        if key in seen: continue
        seen.add(key)
        hc = G if r['head'] not in ("SKIP","GENERIC","CERTIFIED") else Y
        print(f"  {hc}→ {r['head']:<12}{X} {C}{r['dataset']:<22}{X} Hz={r['hz']} | {D}{r['note']}{X}")
        summary[r['dataset']] = r['head']

    all_results.extend(results)

print(f"\n{W}{'═'*62}")
print(f"  ROUTING SUMMARY — All datasets on this machine")
print(f"{'═'*62}{X}")
head_groups = {}
for ds, head in summary.items():
    head_groups.setdefault(head, []).append(ds)
for head, datasets in sorted(head_groups.items()):
    hc = G if head not in ("SKIP","GENERIC","CERTIFIED") else Y
    print(f"  {hc}{head:<14}{X} ← {', '.join(datasets)}")

print(f"\n  {len(all_results)} files routable")
print(f"  {len([r for r in all_results if r['head']=='GESTURE'])} GESTURE")
print(f"  {len([r for r in all_results if r['head']=='ACTIVITY'])} ACTIVITY")
print(f"  {len([r for r in all_results if r['head']=='CARDIAC'])} CARDIAC")
print(f"  {len([r for r in all_results if r['head']=='CERTIFIED'])} CERTIFIED (skip — already done)")
print(f"{'═'*62}\n")

with open(os.path.expanduser("~/S2S/experiments/routing_report.json"), "w") as f:
    json.dump({"summary": summary, "sample_routes": all_results[:20]}, f, indent=2)
print(f"  Report saved → experiments/routing_report.json\n")
