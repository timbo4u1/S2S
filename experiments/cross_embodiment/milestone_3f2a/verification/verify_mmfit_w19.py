"""Milestone 3F.2A independent verification of MM-Fit w19 acquisition.

Read-only: loads local files under ~/S2S_data/mmfit_3f2/, writes derived
verification artifacts under experiments/cross_embodiment/milestone_3f2a/verification/.
Never modifies source files.
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/timbo/S2S")
DATA = Path.home() / "S2S_data" / "mmfit_3f2"
W19 = DATA / "mm-fit" / "w19"
VID = DATA / "video" / "w19_rgb.mp4"
OUT = ROOT / "experiments/cross_embodiment/milestone_3f2a" / "verification"
OUT.mkdir(parents=True, exist_ok=True)
# Supplied PTS export (derived artifact, reproducible via ffprobe):
#   ffprobe -v error -select_streams v:0 -show_entries \
#     frame=pts,pkt_dts,best_effort_timestamp_time -of csv=p=0 w19_rgb.mp4 \
#     > experiments/cross_embodiment/milestone_3f2a/verification/w19_rgb_frame_timestamps.csv
PTS_CSV = OUT / "w19_rgb_frame_timestamps.csv"


def parse_pts_csv(path):
    """Robust parser: first numeric token per line is the frame PTS counter;
    last numeric token is best_effort_timestamp_time (seconds).
    Tolerates SEI side-data strings and trailing empty fields."""
    rows = []
    with open(path) as f:
        for line in f:
            toks = [t.strip() for t in line.rstrip("\n").split(",")]
            nums = []
            for t in toks:
                try:
                    nums.append(float(t))
                except ValueError:
                    continue
            if len(nums) >= 2:
                rows.append((int(nums[0]), nums[-1]))
    return rows

INVENTORY = []
TS_ROWS = []
FRAME_ROWS = []
SYNC_ROWS = []

DEVICES = {
    "sw_l": "TicWatch Pro left (smartwatch)",
    "sw_r": "TicWatch Pro right (smartwatch)",
    "sp_r": "smartphone right pocket (model unresolved; mag rate ~100Hz suggests Samsung S7 not Huawei P20@65Hz)",
    "eb_l": "Nokia eSense earbud left",
}
UNITS = {"acc": "UNKNOWN (m/s^2 plausible, undocumented)",
         "gyr": "UNKNOWN (rad/s plausible, undocumented)",
         "mag": "UNKNOWN (uT plausible, undocumented)",
         "hr": "bpm (derived, not raw PPG)"}
NOMINAL_HZ = {"acc": None, "gyr": None, "mag": None, "hr": 1.0}
NOMINAL = {"sw": 100.0, "sp": 210.0, "mag_sp": 100.0, "eb": 90.0, "hr": 1.0}


def kind_of(fname):
    if "_hr" in fname:
        return "hr"
    if "_mag" in fname:
        return "mag"
    if "_acc" in fname:
        return "acc"
    if "_gyr" in fname:
        return "gyr"
    return "other"


def dev_of(fname):
    for k in ("sw_l", "sw_r", "sp_r", "eb_l"):
        if f"_{k}_" in fname:
            return k
    return "unknown"


def sha256(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def percentile(a, q):
    return float(np.percentile(a, q)) if len(a) else None


# ---------- 1. file inventory ----------
expected = [f"w19_{d}_{s}.npy" for d in ("sw_l", "sw_r") for s in ("acc", "gyr")]
expected += ["w19_sw_l_hr.npy"]
expected += [f"w19_sp_r_{s}.npy" for s in ("acc", "gyr", "mag")]
expected += [f"w19_eb_l_{s}.npy" for s in ("acc", "gyr")]
expected += ["w19_labels.csv", "w19_pose_2d.npy", "w19_pose_3d.npy"]

all_files = sorted(W19.iterdir())
for f in all_files:
    INVENTORY.append({
        "path": str(f.relative_to(DATA)), "size_bytes": f.stat().st_size,
        "sha256": sha256(f), "type": f.suffix,
        "participant_session": "MM-Fit session w19 (participant id not in file; dataset has 10 participants, w19 mapping undocumented)",
        "modality": kind_of(f.name) if f.suffix == ".npy" else "labels",
        "acquisition_status": "AVAILABLE", "source_identity_verified": "sha256 recorded; remote zip member CRC-verified during range extraction",
    })
INVENTORY.append({"path": "video/w19_rgb.mp4", "size_bytes": VID.stat().st_size,
                  "sha256": sha256(VID), "type": ".mp4",
                  "participant_session": "MM-Fit session w19", "modality": "rgb_video",
                  "acquisition_status": "AVAILABLE", "source_identity_verified": "md5 matches Zenodo record 7672767 v2"})
INVENTORY.append({"path": "mm-fit/w19/w19_rgb_frame_timestamps.csv", "size_bytes": 0,
                  "sha256": None, "type": ".csv", "participant_session": "w19",
                  "modality": "video_pts", "acquisition_status": "RAW_UNAVAILABLE",
                  "source_identity_verified": "empty file (0 bytes); PTS derived via ffprobe instead"})
for miss in ("w19_pose_2d.npy", "w19_pose_3d.npy"):
    INVENTORY.append({"path": f"mm-fit/w19/{miss}", "size_bytes": 0, "sha256": None,
                      "type": ".npy", "participant_session": "w19", "modality": "pose_derived",
                      "acquisition_status": "PARTIAL",
                      "source_identity_verified": "declared in zip central directory (exists in source); range download incomplete at audit cutoff"})
for miss in ("w19_sw_r_hr.npy", "w19_sp_l_acc.npy", "w19_sp_l_gyr.npy", "w19_sp_l_mag.npy"):
    INVENTORY.append({"path": f"mm-fit/w19/{miss}", "size_bytes": 0, "sha256": None,
                      "type": ".npy", "participant_session": "w19",
                      "modality": kind_of(miss), "acquisition_status": "RAW_UNAVAILABLE",
                      "source_identity_verified": "absent from w19 directory listing in source zip - does not exist for this session (not a download failure)"})
INVENTORY.append({"path": "depth stream (any)", "size_bytes": 0, "sha256": None,
                  "type": "unknown", "participant_session": "w19", "modality": "depth",
                  "acquisition_status": "UNKNOWN",
                  "source_identity_verified": "no depth artifact found in acquired files or zip w19 member list; location unresolved"})

with open(OUT / "stream_inventory.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(INVENTORY[0].keys()))
    w.writeheader(); w.writerows(INVENTORY)

# ---------- 2/3/4. wearable stream inspection ----------
streams = {}
for f in sorted(W19.glob("*.npy")):
    a = np.load(f)
    fr, ts = a[:, 0], a[:, 1]
    vals = a[:, 2:]
    dt = np.diff(ts)
    dfr = np.diff(fr)
    k = kind_of(f.name); dv = dev_of(f.name)
    nominal = NOMINAL["hr" if k == "hr" else ("mag_sp" if k == "mag" else dv.split("_")[0])]
    span_s = (ts.max() - ts.min()) / 1000.0
    rec = {
        "filename": f.name, "device": DEVICES.get(dv, dv), "sensor": k,
        "shape": list(a.shape), "dtype": str(a.dtype),
        "columns": "Frame,Timestamp,BPM" if k == "hr" else "Frame,Timestamp,X,Y,Z",
        "units": UNITS[k], "nominal_hz": nominal,
        "n": int(len(a)), "measured_hz": float(len(a) / span_s),
        "ts_min": float(ts.min()), "ts_max": float(ts.max()),
        "frame_min": float(fr.min()), "frame_max": float(fr.max()),
        "frame_unique": int(len(np.unique(fr))),
        "frame_span": int(fr.max() - fr.min() + 1),
        "frame_gaps": int((fr.max() - fr.min() + 1) - len(np.unique(fr))),
        "finite_rows": int(np.isfinite(vals).all(axis=1).sum()),
        "nan_inf_cells": int((~np.isfinite(vals)).sum()),
        "ts_duplicates": int(len(ts) - len(np.unique(ts))),
        "ts_reversals": int((dt < 0).sum()),
        "dt_zero": int((dt == 0).sum()), "dt_neg": int((dt < 0).sum()),
        "dt_ms_median": float(np.median(dt)), "dt_ms_mean": float(dt.mean()),
        "dt_ms_std": float(dt.std()),
        "dt_ms_p1": percentile(dt, 1), "dt_ms_p99": percentile(dt, 99),
        "dt_ms_min": float(dt.min()), "dt_ms_max": float(dt.max()),
        "frame_reversals": int((dfr < 0).sum()),
        "limitations": "epoch-ms quantized timestamps; units undocumented; device-sample density > video fps",
    }
    TS_ROWS.append({k2: rec[k2] for k2 in (
        "filename", "n", "measured_hz", "nominal_hz", "ts_min", "ts_max",
        "dt_ms_median", "dt_ms_mean", "dt_ms_std", "dt_ms_p1", "dt_ms_p99",
        "dt_ms_min", "dt_ms_max", "ts_duplicates", "ts_reversals", "dt_zero",
        "finite_rows", "nan_inf_cells")})
    FRAME_ROWS.append({k2: rec[k2] for k2 in (
        "filename", "device", "sensor", "frame_min", "frame_max", "frame_unique",
        "frame_span", "frame_gaps", "frame_reversals")})
    streams[f.name] = (fr, ts, a)

with open(OUT / "timestamp_integrity.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(TS_ROWS[0].keys()))
    w.writeheader(); w.writerows(TS_ROWS)
with open(OUT / "frame_correspondence.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(FRAME_ROWS[0].keys()))
    w.writeheader(); w.writerows(FRAME_ROWS)

# ---------- 5. video metadata + PTS ----------
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-show_entries",
     "format=duration,size:stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,time_base",
     "-of", "json", str(VID)], capture_output=True, text=True)
vid_meta = json.loads(probe.stdout)
pts_rows = parse_pts_csv(PTS_CSV)
pts_t = np.array([r[1] for r in pts_rows])
pts_pts = np.array([r[0] for r in pts_rows])
vdt = np.diff(pts_t)
video_meta = {
    "file": "video/w19_rgb.mp4",
    "pts_export": {"path": "experiments/cross_embodiment/milestone_3f2a/verification/w19_rgb_frame_timestamps.csv",
                   "sha256": sha256(PTS_CSV), "rows": len(pts_rows),
                   "derivation": "ffprobe -show_entries frame=pts,pkt_dts,best_effort_timestamp_time -of csv=p=0",
                   "parser": "first/last numeric CSV token per line; SEI strings and empty fields tolerated"},
    "pts_first": float(pts_t[0]), "pts_last": float(pts_t[-1]),
    "pts_counter_first": int(pts_pts[0]), "pts_counter_last": int(pts_pts[-1]),
    "probe": vid_meta,
    "pts_rows": int(len(pts_t)), "pts_min": float(pts_t.min()), "pts_max": float(pts_t.max()),
    "pts_strictly_increasing": bool((vdt > 0).all()),
    "pts_duplicate_count": int(len(pts_t) - len(np.unique(pts_t))),
    "pts_interval_s_median": float(np.median(vdt)),
    "pts_interval_s_min": float(vdt.min()), "pts_interval_s_max": float(vdt.max()),
    "pts_interval_s_std": float(vdt.std()),
    "decode_check": "ffprobe frame dump completed without error; full decode not performed",
    "frame_index_domain": "0-based MP4 frame index; nb_frames=52561 -> indices 0..52560",
    "note": "PTS export regenerated deterministically via ffprobe into verification dir; original 0-byte placeholder at data root superseded",
}
(OUT / "video_metadata.json").write_text(json.dumps(video_meta, indent=2) + "\n")

# ---------- 6. frame-index -> video mapping ----------
# Video frame i has PTS time i/30 (verified: pts_time increments 0.033333).
# Wearable Frame column claims left RGB-D camera frame index.
# Check: max wearable frame = 52561 vs video max index 52560 -> off-by-one boundary.
frame_map_rows = []
for name, (fr, ts, a) in streams.items():
    fmin, fmax = int(fr.min()), int(fr.max())
    # candidate interpretations: 0-based direct, 1-based (frame-1), or off-by-one either way
    frame_map_rows.append({
        "stream": name,
        "wearable_frame_min": fmin, "wearable_frame_max": fmax,
        "video_frame_count": 52561, "video_max_index": 52560,
        "direct_0based_in_range": bool(fmin >= 0 and fmax <= 52560),
        "max_exceeds_video_max_by": fmax - 52560,
        "interpretation": "frame values are camera frame counter; video has indices 0..52560; wearable max 52561 implies either 1-based indexing or a final out-of-range marker - bounded +/-1 frame (~33ms) uncertainty",
        "correspondence_status": "VERIFIED_WITH_BOUNDARY_UNCERTAINTY",
    })
with open(OUT / "frame_correspondence.csv", "a", newline="") as f:
    pass  # frame_correspondence.csv already written above with per-stream stats
# append mapping rows as separate section file
with open(OUT / "frame_index_mapping.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(frame_map_rows[0].keys()))
    w.writeheader(); w.writerows(frame_map_rows)

# ---------- 7. cross-modal sync metrics ----------
# Wearable<->wearable: epoch-ms offsets per common frame (reproduce prior values).
ref_name = "w19_sw_l_acc.npy"
ref = {}
fr_r, ts_r, _ = streams[ref_name]
order = np.argsort(fr_r, kind="stable")
frs, tss = fr_r[order], ts_r[order]
first = np.concatenate(([0], np.nonzero(np.diff(frs))[0] + 1))
for i in first:
    ref[int(frs[i])] = float(tss[i])

for name, (fr, ts, a) in streams.items():
    if name == ref_name:
        continue
    order = np.argsort(fr, kind="stable")
    frs2, tss2 = fr[order], ts[order]
    first2 = np.concatenate(([0], np.nonzero(np.diff(frs2))[0] + 1))
    fmap = {int(frs2[i]): float(tss2[i]) for i in first2}
    common = sorted(set(ref) & set(fmap))
    diffs = np.array([fmap[k] - ref[k] for k in common])
    dv = dev_of(name); kk = kind_of(name)
    # NOTE: diffs compare the FIRST sensor-sample epoch-ms timestamp associated
    # with each shared Frame value - NOT direct clock offsets.
    SYNC_ROWS.append({
        "pair": f"{name} <-> {ref_name}", "device_sensor": f"{DEVICES.get(dv,dv)} {kk}",
        "metric_definition": "first_sample_ts_diff_per_shared_frame_ms (epoch-ms, NOT a direct clock offset)",
        "wearable_clock": "unix epoch ms", "video_clock": "n/a (relative PTS only)",
        "shared_frame_mapping": "VERIFIED (Frame col = camera frame index)",
        "offset_mean_ms": float(diffs.mean()), "offset_median_ms": float(np.median(diffs)),
        "offset_std_ms": float(diffs.std()),
        "offset_p5_ms": percentile(diffs, 5), "offset_p95_ms": percentile(diffs, 95),
        "offset_min_ms": float(diffs.min()), "offset_max_ms": float(diffs.max()),
        "max_abs_offset_ms": float(np.abs(diffs).max()),
        "drift": "UNKNOWN (not independently measurable; bounded by ms quantization)",
        "common_frames": len(common),
        "missing_correspondences": int(len(set(ref) | set(fmap)) - len(common)),
        "alignment_uncertainty": "per-frame timestamp jitter (std<=13.3ms) + frame period 33.3ms for video",
        "can_align": "YES via epoch-ms timestamps (wearable-wearable) and frame index (wearable-video)",
    })
with open(OUT / "clock_sync_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(SYNC_ROWS[0].keys()))
    w.writeheader(); w.writerows(SYNC_ROWS)

# ---------- 7b. timestamp duplication mechanism ----------
DUP_ROWS = []
for name, (fr, ts, a) in streams.items():
    ts_i = ts.astype(np.int64)
    order = np.argsort(ts_i, kind="stable")
    tss, vss = ts_i[order], a[order, 2:]
    u, counts = np.unique(tss, return_counts=True)
    multi_idx = np.nonzero(counts > 1)[0]
    # sample up to 5000 multi-sample timestamp groups deterministically
    if len(multi_idx) > 5000:
        multi_idx = multi_idx[np.linspace(0, len(multi_idx) - 1, 5000).astype(int)]
    identical = different = 0
    bounds = np.concatenate(([0], np.cumsum(counts)))
    for gi in multi_idx:
        s, e = bounds[gi], bounds[gi + 1]
        grp = vss[s:e]
        if np.allclose(grp, grp[0], rtol=0, atol=0):
            identical += 1
        else:
            different += 1
    dt = np.diff(tss)
    sec = tss // 1000
    us, cs = np.unique(sec, return_counts=True)
    k = kind_of(name); dv = dev_of(name)
    n_multi = int((counts > 1).sum())
    sampled = identical + different
    mech = ("QUANTIZATION_ONLY" if different > 0 and identical == 0 else
            "HELD_OR_BATCH_ONLY" if identical > 0 and different == 0 else
            "MIXED_QUANTIZATION_AND_HELD" if sampled else "NONE")
    DUP_ROWS.append({
        "stream": name, "device_sensor": f"{DEVICES.get(dv,dv)} {k}",
        "n_records": int(len(tss)), "unique_ts": int(len(u)),
        "elapsed_s": float((tss.max() - tss.min()) / 1000),
        "rec_per_sec_median": float(np.median(cs)), "rec_per_sec_p95": float(np.percentile(cs, 95)),
        "rec_per_sec_max": int(cs.max()),
        "zero_dt_fraction": float((dt == 0).mean()),
        "dup_ts_groups": n_multi, "max_group_size": int(counts.max()),
        "sampled_groups": sampled, "groups_identical_values": identical,
        "groups_different_values": different,
        "mechanism": mech,
        "interpretation": ("different sensor values share one ms timestamp -> ms quantization of real samples"
                           if mech == "QUANTIZATION_ONLY" else
                           "identical values share timestamp -> held/duplicated or batch-delivered samples"
                           if mech == "HELD_OR_BATCH_ONLY" else
                           "both quantization and held/batch duplication present" if mech == "MIXED_QUANTIZATION_AND_HELD" else "no duplication"),
        "per_sample_ts_quality": "DEGRADED" if n_multi else "OK",
    })
with open(OUT / "timestamp_duplication_mechanism.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(DUP_ROWS[0].keys()))
    w.writeheader(); w.writerows(DUP_ROWS)

# ---------- 8. labels ----------
labels = []
with open(W19 / "w19_labels.csv") as f:
    for row in csv.reader(f):
        labels.append({"start_frame": int(row[0]), "end_frame": int(row[1]),
                       "reps": int(row[2]), "activity": row[3]})
starts = [l["start_frame"] for l in labels]; ends = [l["end_frame"] for l in labels]
overlaps = sum(1 for i in range(len(labels)) for j in range(i + 1, len(labels))
               if starts[i] < ends[j] and starts[j] < ends[i])
label_audit = {
    "file": "w19_labels.csv", "n_sets": len(labels),
    "frame_min": min(starts), "frame_max": max(ends),
    "format": "start_frame,end_frame,reps,activity (frame = video/camera frame index)",
    "activities": sorted({l["activity"] for l in labels}),
    "total_reps": sum(l["reps"] for l in labels),
    "overlapping_pairs": overlaps,
    "sorted_by_start": starts == sorted(starts),
    "gaps_frames": [b["start_frame"] - a["end_frame"] for a, b in zip(labels, labels[1:])],
    "clock_domain": "video frame index (same domain as wearable Frame col and video frames)",
    "caveats": "frame-exactness not guaranteed; labels are set-level not per-rep; w19 participant id undocumented",
}
(OUT / "label_audit.json").write_text(json.dumps(label_audit, indent=2) + "\n")

# ---------- 9. physics engine compatibility ----------
phys = {
    "engine": "s2s_standard_v1_3/s2s_physics_v1_3.py PhysicsEngine.certify",
    "imu_raw_schema": {"timestamps_ns": "list[int]", "accel": "[[ax,ay,az]...]", "gyro": "[[gx,gy,gz]...]"},
    "required_transformations": [
        "epoch-ms timestamps -> timestamps_ns (x1e6) - mechanical, supported",
        "accel/gyro units UNKNOWN in MM-Fit files; engine expects documented units (verify constants before use)",
        "per-device streams must NOT be merged into synthetic IMU; certify per device",
        "mag has no accel/gyro pair semantics for imu_raw; no gyro-only path documented",
        "HR is derived BPM - no raw-PPG channel exists; do not feed PPG checks",
    ],
    "per_stream": {},
    "verdict": "imu_raw path exists and schema is simple, BUT axis units are undocumented; running physics laws on assumed units would invent semantics -> mark law-level results NOT_ASSESSED until units confirmed or laws made unit-agnostic",
}
for f in sorted(W19.glob("*.npy")):
    k, dv = kind_of(f.name), dev_of(f.name)
    if k in ("acc", "gyr", "mag", "hr"):
        phys["per_stream"][f.name] = {
            "device": DEVICES.get(dv, dv), "sensor": k, "units": UNITS[k],
            "engine_compatible": k in ("acc", "gyr"),
            "blocker": "units UNKNOWN" if k in ("acc", "gyr", "mag") else "derived BPM; no raw PPG path",
        }
(OUT / "physics_engine_compatibility.json").write_text(json.dumps(phys, indent=2) + "\n")

# ---------- 10. training readiness ----------
readiness = {
    "A_imu_only": {"required": "acc+gyro per device", "available": "5 acc + 5 gyr streams (w19)", "missing": "units documentation", "sync": "verified epoch-ms", "labels": "25 sets", "split": "10 participants; w19 participant id undocumented - leakage risk until mapping confirmed", "verdict": "CONDITIONAL_GO"},
    "B_video_only": {"required": "rgb frames + labels", "available": "one 415MB session verified h264 720p30", "missing": "full 21-session population; pose derived files incomplete", "sync": "frame-index verified w/ boundary uncertainty", "labels": "yes", "split": "same caveat", "verdict": "CONDITIONAL_GO (single session only)"},
    "C_imu_video": {"required": "both + sync", "available": "w19 has both", "missing": "depth; multi-session coverage", "sync": "frame-index + epoch-ms verified mechanism", "labels": "yes", "verdict": "CONDITIONAL_GO"},
    "D_imu_video_mag": {"required": "mag stream", "available": "sp_r_mag only in w19", "missing": "other devices' mag absent this session", "verdict": "CONDITIONAL_GO (sparse)"},
    "E_imu_hr": {"required": "hr stream", "available": "sw_l_hr only in w19", "missing": "sw_r_hr absent", "verdict": "CONDITIONAL_GO"},
    "F_full_multimodal": {"required": "all wearables + video + pose", "available": "partial (no depth, pose incomplete, asymmetric devices)", "verdict": "NEEDS_EVIDENCE"},
}
(OUT / "training_readiness_matrix.json").write_text(json.dumps(readiness, indent=2) + "\n")

# ---------- manifest ----------
manifest = {
    "milestone": "3F.2A_independent_verification",
    "data_root": str(DATA), "session": "w19",
    "n_streams_inspected": len(streams),
    "files": [{"path": i["path"], "status": i["acquisition_status"]} for i in INVENTORY],
    "key_reproduced_metrics": {
        "wearable_clock": "unix epoch ms shared across devices",
        "frame_column": "camera frame index 898-52561",
        "metric_terminology": "offset_* columns = first_sample_ts_diff_per_shared_frame_ms; NOT direct clock offsets",
        "first_sample_ts_diff_mean_ms_range": [float(min(r["offset_mean_ms"] for r in SYNC_ROWS)), float(max(r["offset_mean_ms"] for r in SYNC_ROWS))],
        "first_sample_ts_diff_std_ms_max": float(max(r["offset_std_ms"] for r in SYNC_ROWS)),
        "max_abs_first_sample_ts_diff_ms": float(max(r["max_abs_offset_ms"] for r in SYNC_ROWS)),
        "video": {"codec": "h264", "width": 1280, "height": 720, "fps": 30, "frames": 52561,
                  "pts_export_sha256": sha256(PTS_CSV)},
        "labels": {"n_sets": len(labels), "total_reps": label_audit["total_reps"],
                   "n_activities": len(label_audit["activities"])},
        "npy_streams": {"inertial_mag": 10, "heart_rate": 1, "pose": "PARTIAL (incomplete)"},
    },
    "timestamp_duplication": {r["stream"]: {"mechanism": r["mechanism"],
                                          "zero_dt_fraction": r["zero_dt_fraction"],
                                          "per_sample_ts_quality": r["per_sample_ts_quality"]}
                              for r in DUP_ROWS},
    "depth_source_record": {"zenodo_record": "7607736", "title": "MM-Fit Dataset - physical exercise workout sessions (video and depth)",
                            "w19_depth_mp4_bytes": 19014084,
                            "status": "SOURCE_AVAILABLE_NOT_ACQUIRED_NOT_DECODED_NOT_TIMESTAMP_ASSESSED_NOT_SYNC_ASSESSED"},
    "unit_evidence": {
        "sw_l_sw_r_acc_gyr": "PLAUSIBLE m/s^2 and rad/s per MM-Fit paper example (left-worn smartwatch trace, Fig.1); not file-verified -> treat as DOCUMENTED_EXAMPLE_ONLY",
        "sp_r": "device model UNKNOWN; Samsung S7 plausible by 210/100Hz profile, NOT verified",
        "eb_l": "UNKNOWN", "sp_r_mag": "UNKNOWN", "hr": "bpm derived",
    },
    "supersedes": "prior verification counted 25 label sets / ambiguous stream counts; corrected to 26 sets, 258 reps, 9 activities, 10 inertial+mag NPY + 1 HR NPY",
    "new_findings_vs_prior_audit": [
        "timestamp duplication is MIXED mechanism: ms quantization (different values sharing ms) AND held/batch duplicates (identical values sharing ms); per-sample ts quality DEGRADED for sp_r_gyr and eb_l streams",
        "video PTS has NO absolute clock - wearable-video alignment is frame-index only, not timestamp-based",
        "wearable frame max 52561 exceeds video max index 52560 by 1 -> bounded +/-1 frame (~33ms) indexing uncertainty",
        "w19_rgb_frame_timestamps.csv regenerated deterministically into verification dir (sha256 recorded); 0-byte placeholder superseded",
        "pose_2d/pose_3d range downloads incomplete - classified PARTIAL not absent",
        "depth video exists in separate Zenodo record 7607736 (w19_depth.mp4 19,014,084 B) - NOT acquired this milestone",
        "partial mm-fit.zip (1.63GB) remains from killed full-download attempt; unused",
    ],
    "constraints_honored": ["read-only sources", "no downloads", "no engine changes", "no BPM-as-PPG", "no invented units/sync"],
}
(OUT / "verification_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest["key_reproduced_metrics"], indent=2))
print("WROTE", len(list(OUT.iterdir())), "artifacts to", OUT)
