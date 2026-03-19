#!/usr/bin/env python3
"""
collect_action.py — S2S v1.3 | Labeled Action Record Builder

USAGE — with Sensor Logger annotations:
  python3 collect_action.py \
    --accel AccelerometerUncalibrated.csv \
    --gyro  GyroscopeUncalibrated.csv \
    --gravity Gravity.csv \
    --annotation Annotation.csv \
    --person timbo --out dataset/

USAGE — with manual label (whole file):
  python3 collect_action.py \
    --accel AccelerometerUncalibrated.csv \
    --gyro  GyroscopeUncalibrated.csv \
    --gravity Gravity.csv \
    --label reach_forward \
    --person timbo --out dataset/

USAGE — show dataset summary:
  python3 collect_action.py --summary --out dataset/
"""

import sys, os, csv, json, math, argparse, glob
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Loaders ────────────────────────────────────────────────────────
def load_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            d = {'t': float(r['seconds_elapsed']), 'time': int(r['time'])}
            for k in r:
                if k not in ('time', 'seconds_elapsed'):
                    try: d[k] = float(r[k])
                    except: pass
            rows.append(d)
    return rows

def load_annotations(path):
    """Load Sensor Logger Annotation.csv → [{t, time, label}]"""
    rows = []
    if not path or not os.path.exists(path):
        return rows
    size = os.path.getsize(path)
    if size < 5:
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            label = r.get('annotation', '').strip()
            if label:
                rows.append({
                    'time':  int(r['time']),
                    't':     float(r['seconds_elapsed']),
                    'label': label.lower().replace(' ', '_')
                })
    return rows

# ── Split by gyro magnitude into individual reps ───────────────────
def split_into_reps(accel_seg, gyro_seg, grav_seg, label,
                    min_rep_s=1.0, rest_thresh=0.15):
    """Find individual action reps by detecting rest periods between them."""
    n = len(accel_seg)
    if n < 20:
        return [(label, accel_seg, gyro_seg, grav_seg)]

    dur = accel_seg[-1]['t'] - accel_seg[0]['t']
    hz  = n / dur if dur > 0 else 100.0
    min_samples  = int(min_rep_s * hz)
    rest_samples = int(0.4 * hz)

    gmag = [math.sqrt(gyro_seg[i]['x']**2 + gyro_seg[i]['y']**2 + gyro_seg[i]['z']**2)
            for i in range(n)]

    reps = []
    in_action = False
    start_i = 0
    rest_count = 0

    for i in range(n):
        if not in_action:
            if gmag[i] > rest_thresh:
                in_action = True; start_i = i; rest_count = 0
        else:
            if gmag[i] < rest_thresh:
                rest_count += 1
                if rest_count >= rest_samples:
                    end_i = i - rest_count
                    if end_i - start_i >= min_samples:
                        reps.append((label,
                                     accel_seg[start_i:end_i],
                                     gyro_seg[start_i:end_i],
                                     grav_seg[start_i:end_i] if grav_seg else []))
                    in_action = False
            else:
                rest_count = 0

    if in_action and n - start_i >= min_samples:
        reps.append((label, accel_seg[start_i:], gyro_seg[start_i:],
                     grav_seg[start_i:] if grav_seg else []))

    return reps if reps else [(label, accel_seg, gyro_seg, grav_seg)]

# ── Certify one segment and save ──────────────────────────────────
def certify_segment(accel_seg, gyro_seg, grav_seg,
                    label, person_id, device_id, out_dir):
    from s2s_standard_v1_3.s2s_physics_v1_3 import (
        PhysicsEngine, check_rigid_body, check_jerk, check_imu_consistency)
    from s2s_standard_v1_3.s2s_signing_v1_3 import CertSigner
    from s2s_standard_v1_3.s2s_stream_certify_v1_3 import StreamCertifier

    n   = len(accel_seg)
    G   = 9.81
    ts  = [accel_seg[i]['time'] for i in range(n)]
    araw = [[accel_seg[i]['x']*G, accel_seg[i]['y']*G, accel_seg[i]['z']*G] for i in range(n)]
    graw = [[gyro_seg[i]['x'],    gyro_seg[i]['y'],    gyro_seg[i]['z']]    for i in range(n)]

    # Linear accel (gravity removed) if available
    if grav_seg and len(grav_seg) == n and grav_seg[0] is not None:
        lin = [[accel_seg[i]['x']*G - grav_seg[i]['x'],
                accel_seg[i]['y']*G - grav_seg[i]['y'],
                accel_seg[i]['z']*G - grav_seg[i]['z']] for i in range(n)]
    else:
        lin = araw

    imu_raw = {"timestamps_ns": ts, "accel": araw, "gyro": graw}
    imu_lin = {"timestamps_ns": ts, "accel": lin,  "gyro": graw}

    phys       = PhysicsEngine().certify(imu_raw=imu_raw, segment="forearm")
    _, _, d_rb = check_rigid_body(imu_raw)
    _, _, d_jk = check_jerk(imu_lin)
    _, _, d_ic = check_imu_consistency(imu_raw)

    # Stream certifier for IMU tier
    win = min(256, n // 2)
    sc  = StreamCertifier(['ax','ay','az','gx','gy','gz'], window=win, step=win)
    imu_cert = None
    for i in range(n):
        r = sc.push_frame(ts[i], araw[i] + graw[i])
        if r: imu_cert = r

    dur = accel_seg[-1]['t'] - accel_seg[0]['t']

    record = {
        "schema":             "s2s_action_record_v1",
        "action":             label,
        "person_id":          person_id,
        "device_id":          device_id,
        "recorded_at":        datetime.now(timezone.utc).isoformat(),
        "duration_s":         round(dur, 2),
        "n_samples":          n,
        "sample_rate_hz":     round(n / dur, 1) if dur > 0 else 100,
        "imu_tier":           imu_cert['tier'] if imu_cert else 'UNRATED',
        "physics_tier":       phys['tier'],
        "physics_score":      phys['physical_law_score'],
        "physics_laws_passed":phys['laws_passed'],
        "physics_laws_failed":phys['laws_failed'],
        "jerk_p95_ms3":       d_jk.get('p95_jerk_ms3'),
        "imu_coupling_r":     d_rb.get('pearson_r_measured_vs_predicted'),
        "imu_consistency_r":  d_ic.get('pearson_r_var_coupling'),
        "flags":              list(set((phys.get('flags') or []) +
                                       (imu_cert.get('flags', []) if imu_cert else []))),
        "sensor_data":        {"timestamps_ns": ts, "accel_ms2": araw, "gyro_rads": graw},
    }

    signer, _ = CertSigner.generate()
    signed = signer.sign_cert(record)

    os.makedirs(out_dir, exist_ok=True)
    existing = glob.glob(f"{out_dir}/{label}_{person_id}_*.json")
    seq      = len(existing) + 1
    fname    = f"{label}_{person_id}_{seq:03d}.json"
    fpath    = os.path.join(out_dir, fname)

    with open(fpath, 'w') as f:
        json.dump(signed, f, indent=2)

    tier  = phys['tier']
    score = phys['physical_law_score']
    jerk  = d_jk.get('p95_jerk_ms3', '?')
    coup  = d_rb.get('pearson_r_measured_vs_predicted', '?')
    print(f"  ✓ {fname}  tier={tier} score={score}/100  jerk={jerk}m/s³  r={coup}")
    return fpath

# ── Dataset summary ───────────────────────────────────────────────
def dataset_summary(out_dir):
    records = glob.glob(f"{out_dir}/**/*.json", recursive=True) + \
              glob.glob(f"{out_dir}/*.json")
    if not records:
        print("  No records found")
        return
    actions, tiers = {}, {}
    for path in records:
        try:
            with open(path) as f: rec = json.load(f)
            a = rec.get('action', '?')
            t = rec.get('physics_tier', '?')
            actions[a] = actions.get(a, 0) + 1
            tiers[t]   = tiers.get(t, 0) + 1
        except: pass
    print(f"\n{'='*50}")
    print(f"  DATASET:  {out_dir}")
    print(f"  Records:  {len(records)}")
    print(f"\n  Actions:")
    for a, c in sorted(actions.items()):
        print(f"    {a:<22} {c:>3}  {'█'*c}")
    print(f"\n  Tiers:  {tiers}")
    print(f"{'='*50}")

# ── Main ──────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--accel')
    p.add_argument('--gyro')
    p.add_argument('--gravity')
    p.add_argument('--annotation', help='Annotation.csv from Sensor Logger')
    p.add_argument('--label',      help='Manual action label')
    p.add_argument('--person',     default='user1')
    p.add_argument('--device',     default='iphone_11')
    p.add_argument('--out',        default='dataset/')
    p.add_argument('--t-start',    type=float)
    p.add_argument('--t-end',      type=float)
    p.add_argument('--summary',    action='store_true')
    args = p.parse_args()

    if args.summary:
        dataset_summary(args.out); return

    if not args.accel:
        p.print_help(); return

    print(f"\nS2S Action Collector")
    print("=" * 45)
    accel_rows   = load_csv(args.accel)
    gyro_rows    = load_csv(args.gyro)
    grav_rows    = load_csv(args.gravity) if args.gravity else []
    annotations  = load_annotations(args.annotation) if args.annotation else []

    dur = accel_rows[-1]['t']
    print(f"  {len(accel_rows)} samples  {dur:.1f}s  ~{len(accel_rows)/dur:.0f}Hz")

    n = min(len(accel_rows), len(gyro_rows))

    if annotations:
        # ── ANNOTATION MODE ──────────────────────────────────────
        print(f"\n  Annotations ({len(annotations)}):")
        for a in annotations:
            print(f"    t={a['t']:.1f}s → '{a['label']}'")
        print()
        total_reps = 0
        for i, ann in enumerate(annotations):
            t0 = ann['t']
            t1 = annotations[i+1]['t'] if i+1 < len(annotations) else dur
            label = ann['label']
            idx = [j for j in range(n) if t0 <= accel_rows[j]['t'] < t1]
            if len(idx) < 20: continue
            a_seg  = [accel_rows[j] for j in idx]
            g_seg  = [gyro_rows[j]  for j in idx]
            gr_seg = [grav_rows[j] if (grav_rows and j < len(grav_rows) and grav_rows[j] is not None) else None for j in idx]
            reps = split_into_reps(a_seg, g_seg, gr_seg, label)
            print(f"  [{label}] → {len(reps)} reps found:")
            for rep_label, ra, rg, rgr in reps:
                certify_segment(ra, rg, rgr, rep_label, args.person, args.device, args.out)
                total_reps += 1
        print(f"\n  Total certified: {total_reps} reps")

    elif args.label:
        # ── MANUAL LABEL MODE ────────────────────────────────────
        print(f"\n  Certifying: {args.label}")
        t0, t1 = args.t_start, args.t_end
        if t0 or t1:
            t0 = t0 or 0; t1 = t1 or dur
            idx = [j for j in range(n) if t0 <= accel_rows[j]['t'] <= t1]
        else:
            idx = list(range(n))
        a_seg  = [accel_rows[j] for j in idx]
        g_seg  = [gyro_rows[j]  for j in idx]
        gr_seg = [grav_rows[j] if (grav_rows and j < len(grav_rows) and grav_rows[j] is not None) else None for j in idx]
        certify_segment(a_seg, g_seg, gr_seg,
                        args.label, args.person, args.device, args.out)
    else:
        print("ERROR: provide --label or --annotation")
        p.print_help(); return

    dataset_summary(args.out)

if __name__ == '__main__':
    main()
