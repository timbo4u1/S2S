"""Per-stream wearable integrity + timestamp findings for MM-Fit w19."""
from typing import Any, Dict, List

import numpy as np

from experiments.cross_embodiment.milestone_3f2b import mmfit_adapter as ma


def wearable_checks(streams: Dict[str, Dict[str, Any]],
                    stats: Dict[str, Dict[str, Any]]) -> List[Any]:
    """One integrity + one timestamp-quality check per stream; honest NOT_ASSESSED
    for unit-dependent physics and degraded per-sample timing."""
    checks = []
    for name, rec in streams.items():
        st = stats[name]
        subj = {"session_id": "w19", "sample_id": name}
        refs = [ma.evidence_ref(rec["source_path"], rec["sha256"])]
        kind = rec["sensor_kind"]

        # Stream integrity: finite rows, monotonic frames, expected shape.
        finite_ok = st["nan_inf_cells"] == 0
        frame_ok = st["frame_reversals"] == 0
        status = "PASS" if (finite_ok and frame_ok) else "WARNING"
        checks.append(ma.check(
            "MMFIT_STREAM_INTEGRITY", "COMMON", "EPISODE", status,
            "STREAM_FINITE_ORDERED" if status == "PASS" else "STREAM_ANOMALY",
            f"{rec['device_label']} {rec['sensor']}: {st['n_records']} rows, "
            f"{st['finite_rows']} finite, frame reversals={st['frame_reversals']}, "
            f"frame gaps={st['frame_gaps']} of span {int(st['frame_max']-st['frame_min']+1)}.",
            observed={"n_records": st["n_records"], "finite_rows": st["finite_rows"],
                      "frame_gaps": st["frame_gaps"], "shape": st["shape"]},
            refs=refs, limitations=["frame gaps present; per-frame coverage incomplete"],
            subject=subj))

        # Timestamp quality: epoch-ms shared, but degraded per-sample timing.
        if st["ts_reversals"] == 0 and st["per_sample_ts_quality"] == "OK":
            tstatus, treason = "PASS", "TIMESTAMPS_CLEAN"
        elif st["ts_reversals"] == 0:
            tstatus, treason = "WARNING", "TIMESTAMP_QUANTIZATION_OR_HELD"
        else:
            tstatus, treason = "FAIL", "TIMESTAMP_REVERSALS"
        checks.append(ma.check(
            "MMFIT_TIMESTAMP_QUALITY", "COMMON", "EPISODE", tstatus,
            treason,
            f"{rec['device_label']} {rec['sensor']}: epoch-ms timestamps; "
            f"{st['ts_duplicates']} duplicate ts ({st['zero_dt_fraction']:.1%} zero-dt), "
            f"observed_pattern={st['dup_observed_pattern']} "
            f"(cause: {st['dup_causal_interpretation']}), "
            f"quality={st['per_sample_ts_quality']}. No resampling applied.",
            observed={"ts_duplicates": st["ts_duplicates"],
                      "zero_dt_fraction": st["zero_dt_fraction"],
                      "dup_observed_pattern": st["dup_observed_pattern"],
                      "dup_causal_interpretation": st["dup_causal_interpretation"],
                      "dup_confidence": st["dup_confidence"],
                      "observed_record_rate_hz": st["observed_record_rate_hz"],
                      "observed_distinct_ts_rate_hz": st["observed_distinct_ts_rate_hz"],
                      "verified_nominal_hz": st["verified_nominal_hz"],
                      "candidate_reference_hz": st["candidate_reference_hz"],
                      "unique_ts": st["ts_unique"],
                      "per_sample_ts_quality": st["per_sample_ts_quality"]},
            unit="ms", refs=refs,
            limitations=["ms-quantized timestamps; duplicate-group causal mechanism inferred not proven",
                         "record-rate estimate is not proof of physical sensor rate"],
            clock_domain="wearable_epoch_ms", subject=subj))

        # Unit/semantics: UNKNOWN unless documentary example evidence.
        units_ok = "UNKNOWN" not in st["units"]
        if kind == "hr":
            ust = "PASS"; ureason = "DERIVED_BPM_NOT_PPG"
            uexp = ("Derived heart-rate stream (bpm). Explicitly NOT raw PPG; "
                    "must not enter PPG-specific PhysicsEngine checks.")
        elif units_ok:
            ust = "PASS"; ureason = "UNITS_DOCUMENTED"
            uexp = f"Units: {st['units']}."
        else:
            ust = "NOT_ASSESSED"; ureason = "UNITS_UNKNOWN"
            uexp = (f"Units unresolved ({st['units']}). Documentary example evidence "
                    "exists for smartwatch acc/gyro only; physical-law evaluation NOT_ASSESSED.")
        checks.append(ma.check(
            "MMFIT_UNIT_SEMANTICS", "HUMAN", "EPISODE", ust, ureason, uexp,
            observed={"units": st["units"], "device": rec["device_label"],
                      "device_unit_evidence": rec["unit_evidence"],
                      "observed_record_rate_hz": st["observed_record_rate_hz"],
                      "candidate_reference_hz": st["candidate_reference_hz"]},
            refs=refs,
            limitations=["coordinate convention undocumented", "no calibration data acquired"],
            not_assessed=None if ust != "NOT_ASSESSED" else "UNVERIFIED_UNITS",
            subject=subj))

        # PhysicsEngine compatibility: schema OK, laws gated on units.
        if kind in ("acc", "gyr"):
            schema_ok = True
        else:
            schema_ok = False
        checks.append(ma.check(
            "MMFIT_PHYSICS_SCHEMA_COMPAT", "HUMAN", "EPISODE",
            "PASS" if schema_ok else "NOT_ASSESSED",
            "SCHEMA_COMPATIBLE" if schema_ok else "NO_ENGINE_PATH",
            ("imu_raw schema {timestamps_ns, accel, gyro} structurally compatible; "
             "epoch-ms -> ns is mechanical. Per-device only; no merging."
             if schema_ok else
             "Magnetometer/HR have no documented imu_raw path; HR is derived BPM, "
             "never raw PPG."),
            observed={"engine": "PhysicsEngine.certify v1.8", "schema": "imu_raw",
                      "device": rec["device_label"], "sensor": rec["sensor"]},
            refs=refs,
            not_assessed=None if schema_ok else "NO_DOCUMENTED_ENGINE_PATH",
            subject=subj))
        checks.append(ma.check(
            "MMFIT_PHYSICS_LAWS", "HUMAN", "EPISODE", "NOT_ASSESSED",
            "UNVERIFIED_UNITS" if kind in ("acc", "gyr", "mag") else "DERIVED_SIGNAL_NO_LAW",
            "Physical-law evaluation withheld: axis units unverified for this stream; "
            "existing v1.8 thresholds/laws unchanged.",
            observed={"blocked_by": "units UNKNOWN" if "UNKNOWN" in st["units"] else "derived semantics"},
            refs=refs,
            limitations=["no law-level certification claimed"],
            not_assessed="UNVERIFIED_UNITS_OR_DERIVED_SEMANTICS",
            subject=subj))
    return checks


def inventory_payload(streams, stats):
    return {"session": "w19", "streams": [
        {"filename": s["filename"], "device": s["device_label"],
         "device_key": s["device_key"], "device_identity_verified": s["device_key"] != "sp_r",
         "sensor": s["sensor"], "sensor_kind": s["sensor_kind"],
         "shape": s["shape"], "dtype": s["dtype"], "columns": s["columns"],
         "units": s["units"],
         "observed_record_rate_hz": stats[s["filename"]]["observed_record_rate_hz"],
         "observed_distinct_ts_rate_hz": stats[s["filename"]]["observed_distinct_ts_rate_hz"],
         "verified_nominal_hz": stats[s["filename"]]["verified_nominal_hz"],
         "candidate_reference_hz": stats[s["filename"]]["candidate_reference_hz"],
         "clock_domain": s["clock_domain"], "frame_domain": s["frame_domain"],
         "participant_id": s["participant_id"], "session_id": s["session_id"],
         "source_path": s["source_path"], "sha256": s["sha256"],
         "merged_with_other_devices": False}
        for s in streams.values()]}


def timestamp_payload(stats):
    return {"clock_domain": "unix_epoch_ms",
            "notes": ["all wearable Timestamp columns share epoch-ms representation",
                      "duplication is mixed quantization/held-batch; per-sample timing DEGRADED",
                      "HR stream comparatively clean", "no negative reversals observed",
                      "no resampling or interpolation applied"],
            "streams": {name: {k: st[k] for k in (
                "ts_min", "ts_max", "ts_unique", "ts_duplicates", "ts_reversals",
                "zero_dt_fraction", "dt_ms_median", "dt_ms_mean", "dt_ms_p1",
                "dt_ms_p99", "dt_ms_min", "dt_ms_max",
                "dup_observed_pattern", "dup_causal_interpretation", "dup_confidence",
                "dup_groups_identical_values", "dup_groups_different_values",
                "observed_record_rate_hz", "observed_distinct_ts_rate_hz",
                "verified_nominal_hz", "candidate_reference_hz",
                "per_sample_ts_quality", "frame_min", "frame_max", "frame_unique",
                "frame_gaps", "frame_reversals")}
                for name, st in stats.items()}}
