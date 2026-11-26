"""Run the MM-Fit w19 multimodal assessment and write outputs/w19/*."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.cross_embodiment.milestone_3f2b import (
    frame_association, mmfit_adapter as ma, video_assessment, wearable_assessment)

DATA_ROOT = ma.DEFAULT_DATA_ROOT
PTS_CSV = (Path(__file__).resolve().parent.parent / "milestone_3f2a" / "verification"
           / "w19_rgb_frame_timestamps.csv")
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "w19"


def _dump(name, payload):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def label_audit(labels):
    starts = [l["start_frame"] for l in labels]
    ends = [l["end_frame"] for l in labels]
    overlaps = sum(1 for i in range(len(labels)) for j in range(i + 1, len(labels))
                   if starts[i] < ends[j] and starts[j] < ends[i])
    return {
        "n_sets": len(labels), "total_repetitions": sum(l["repetitions"] for l in labels),
        "activities": sorted({l["activity"] for l in labels}),
        "n_activities": len({l["activity"] for l in labels}),
        "overlapping_pairs": overlaps,
        "sorted_by_start": starts == sorted(starts),
        "frame_min": min(starts), "frame_max": max(ends),
        "clock_domain": "camera_frame_index (same domain as wearable Frame col)",
        "boundary_uncertainty_frames": 1,
        "caveats": ["set-level annotations, not per-rep", "frame-exactness not guaranteed",
                    "not aligned to absolute wearable timestamps"],
    }


def main():
    streams = ma.load_wearable_streams(DATA_ROOT)
    stats = {n: ma.stream_stats(r) for n, r in streams.items()}
    labels = ma.load_labels(DATA_ROOT)
    la = label_audit(labels)

    video_path = DATA_ROOT / "video" / "w19_rgb.mp4"
    probe = video_assessment.probe_video(video_path)
    pts_rows = ma.load_video_pts(PTS_CSV)
    pts_sha = sha256_file(PTS_CSV)
    vchecks, vpayload = video_assessment.video_checks(
        video_path, PTS_CSV, probe=probe, pts_rows=pts_rows, pts_sha256=pts_sha)

    fa_payload = frame_association.frame_association(
        streams, labels, video_frames=vpayload["stats"]["rows"])
    # Aggregate frame-association finding cites ALL contributing evidence.
    fa_extra_refs = [
        ma.evidence_ref("mm-fit/w19/w19_labels.csv",
                        sha256_file(DATA_ROOT / "mm-fit/w19/w19_labels.csv")),
        ma.evidence_ref("video/w19_rgb.mp4", vpayload["sha256"]),
        ma.evidence_ref("experiments/cross_embodiment/milestone_3f2a/verification/"
                        "w19_rgb_frame_timestamps.csv", vpayload["pts_export_sha256"]),
    ]
    fa_checks = frame_association.frame_checks(fa_payload, streams, fa_extra_refs)

    wchecks = wearable_assessment.wearable_checks(streams, stats)

    # Labels check
    lchecks = [ma.check(
        "MMFIT_LABEL_INTEGRITY", "COMMON", "EPISODE",
        "PASS" if (la["n_sets"] == 26 and la["overlapping_pairs"] == 0) else "WARNING",
        "LABELS_CONSISTENT" if la["overlapping_pairs"] == 0 else "LABEL_OVERLAP",
        f"{la['n_sets']} exercise sets, {la['total_repetitions']} reps, "
        f"{la['n_activities']} activities, {la['overlapping_pairs']} overlaps; "
        "frame-index domain shared with wearable Frame column.",
        observed={"n_sets": la["n_sets"], "total_repetitions": la["total_repetitions"],
                  "n_activities": la["n_activities"],
                  "overlapping_pairs": la["overlapping_pairs"]},
        refs=[ma.evidence_ref("mm-fit/w19/w19_labels.csv",
                              sha256_file(DATA_ROOT / "mm-fit/w19/w19_labels.csv"))],
        limitations=["frame-index domain; +/-1 frame boundary uncertainty",
                     "participant id for w19 undocumented"],
        subject={"session_id": "w19", "sample_id": "w19_labels.csv"})]

    # Missing modalities honesty check
    mchecks = [ma.check(
        "MMFIT_MODALITY_ABSENCE", "COMMON", "EPISODE", "NOT_ASSESSED",
        "MODALITIES_NOT_ACQUIRED",
        "Not present in this bounded acquisition: depth video (source exists), "
        "pose_2d/pose_3d (PARTIAL), sw_r_hr, sp_l_*, EMG, raw PPG. "
        "Other MM-Fit sessions not assessed; no full-population claim.",
        observed={"absent": ["depth_video", "pose_2d", "pose_3d", "sw_r_hr",
                             "sp_l_acc", "sp_l_gyr", "sp_l_mag", "emg", "raw_ppg"]},
        limitations=["bounded w19 acquisition only"],
        not_assessed="NOT_ACQUIRED_IN_BOUNDED_SCOPE",
        subject={"session_id": "w19"})]

    checks = wchecks + vchecks + fa_checks + lchecks + mchecks

    # modality availability (schema enum: AVAILABLE/PARTIAL/RAW_UNAVAILABLE)
    modality_availability = {}
    for s in streams.values():
        key = f"{s['device_key']}_{s['sensor_kind']}"
        modality_availability[key] = "AVAILABLE"
    modality_availability.update({
        "rgb_video": "AVAILABLE", "depth_video": "RAW_UNAVAILABLE",
        "pose_2d": "PARTIAL", "pose_3d": "PARTIAL",
        "labels": "AVAILABLE", "emg": "RAW_UNAVAILABLE", "raw_ppg": "RAW_UNAVAILABLE",
        "sw_r_hr": "RAW_UNAVAILABLE", "sp_l_sensors": "RAW_UNAVAILABLE",
    })

    provenance = {
        "dataset": "MM-Fit", "doi": "10.1145/3432701", "license": "CC-BY-4.0",
        "zenodo_record": "7672767", "depth_source_record": "7607736",
        "acquisition": "bounded range-member extraction; see milestone_3f2a manifest",
        "prior_verification": "MILESTONE_3F2A_INDEPENDENT_VERIFICATION.md + 3F2A1 reconciliation",
    }
    present = sorted(set(
        ["imu_accel", "imu_gyro", "magnetometer", "heart_rate_derived", "rgb_video",
         "labels"] + [f"{s['device_key']}_{s['sensor_kind']}" for s in streams.values()]))
    record = ma.build_dataset_record(streams, labels, present, provenance)
    report = ma.assemble_report(record, checks, modality_availability, {
        "bounded_scope": "session w19 only; full MM-Fit population NOT acquired",
        "no_universal_score": True,
        "physics_engine": "v1.8 unchanged; law-level eval NOT_ASSESSED pending unit verification",
        "frame_boundary_uncertainty": fa_payload["boundary_uncertainty_frames"],
    })

    _dump("wearable_stream_inventory.json",
          {"session": "w19", "streams": [
              {k: v for k, v in s.items() if k not in ("array", "candidate_nominal_hz_hint")} | {
                  "observed_record_rate_hz": stats[s["filename"]]["observed_record_rate_hz"],
                  "observed_distinct_ts_rate_hz": stats[s["filename"]]["observed_distinct_ts_rate_hz"],
                  "verified_nominal_hz": stats[s["filename"]]["verified_nominal_hz"],
                  "candidate_reference_hz": stats[s["filename"]]["candidate_reference_hz"],
                  "dup_observed_pattern": stats[s["filename"]]["dup_observed_pattern"],
                  "dup_causal_interpretation": stats[s["filename"]]["dup_causal_interpretation"],
                  "per_sample_ts_quality": stats[s["filename"]]["per_sample_ts_quality"]}
              for s in streams.values()]})
    _dump("wearable_timestamp_assessment.json",
          __import__("experiments.cross_embodiment.milestone_3f2b.wearable_assessment",
                     fromlist=["timestamp_payload"]).timestamp_payload(stats))
    _dump("rgb_video_assessment.json", vpayload)
    _dump("frame_association.json", fa_payload)
    _dump("label_assessment.json", la)
    _dump("modality_readiness.json", modality_readiness(streams, stats, vpayload, fa_payload, la))
    _dump("assessment_report.json", report.to_dict())
    print(f"WROTE {OUT_DIR}")
    print(json.dumps(report.compute_summary(), indent=2))


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def modality_readiness(streams, stats, vpayload, fa_payload, la):
    r = {}
    for s in streams.values():
        st = stats[s["filename"]]
        k = s["sensor_kind"]
        if k == "hr":
            r[s["filename"]] = "READY_FOR_TEST (derived BPM; not PPG)"
        elif "UNKNOWN" in st["units"]:
            r[s["filename"]] = "CONDITIONAL (schema OK; units unverified -> laws NOT_ASSESSED)"
        else:
            r[s["filename"]] = "READY_FOR_TEST"
    r["rgb_video_integrity"] = "READY_FOR_TEST (independent checks implemented)"
    r["depth_video"] = "RAW_UNAVAILABLE (source: zenodo 7607736, not acquired)"
    r["frame_level_wearable_video_association"] = "CONDITIONAL (+/-1 frame uncertainty)"
    r["sample_level_wearable_video_sync"] = "NOT_ASSESSED (no absolute video clock; degraded per-sample ts)"
    r["multimodal_model_training"] = "NOT_ASSESSED until 3F.3 experiments"
    r["labels"] = f"READY_FOR_TEST ({la['n_sets']} sets, {la['total_repetitions']} reps)"
    return r


if __name__ == "__main__":
    main()
