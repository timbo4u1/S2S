"""MM-Fit w19 multimodal evidence adapter.

Loads the bounded MM-Fit acquisition (default ~/S2S_data/mmfit_3f2/,
overridable with S2S_MMFIT_DATA_ROOT) and produces a shared-schema
AssessmentReport plus modality-specific evidence artifacts.

Honesty constraints baked in:
- Per-device streams are never merged into a synthetic IMU.
- Units stay UNKNOWN unless documentary evidence exists.
- Derived BPM is never treated as raw PPG; no EMG exists.
- Video PTS is relative; frame-index association carries +/-1 frame uncertainty.
- Depth is locally RAW_UNAVAILABLE (source record exists, not acquired).
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from experiments.cross_embodiment import quality_schema as qs

ADAPTER_NAME = "mmfit_adapter"
ADAPTER_VERSION = "3f.2b.2"
SESSION_ID = "w19"
DATASET_ID = "mmfit"
DATA_ROOT_ENV_VAR = "S2S_MMFIT_DATA_ROOT"
DEFAULT_DATA_ROOT = Path.home() / "S2S_data" / "mmfit_3f2"

DEVICE_INFO = {
    "sw_l": {"label": "TicWatch Pro left", "kind": "smartwatch",
             "unit_evidence": "DOCUMENTED_EXAMPLE_ONLY (paper shows m/s^2, rad/s for a left-worn smartwatch)"},
    "sw_r": {"label": "TicWatch Pro right", "kind": "smartwatch",
             "unit_evidence": "DOCUMENTED_EXAMPLE_ONLY"},
    "sp_r": {"label": "smartphone right pocket", "kind": "phone",
             "unit_evidence": "UNKNOWN; Samsung S7 plausible by 210/100Hz profile, NOT verified"},
    "eb_l": {"label": "Nokia eSense earbud left", "kind": "earbud",
             "unit_evidence": "UNKNOWN"},
}
SENSOR_INFO = {
    "acc": {"signal": "accelerometer", "columns": ["Frame", "Timestamp", "X", "Y", "Z"],
            "shape_cols": 5, "units": "UNKNOWN (m/s^2 plausible for sw_*)"},
    "gyr": {"signal": "gyroscope", "columns": ["Frame", "Timestamp", "X", "Y", "Z"],
            "shape_cols": 5, "units": "UNKNOWN (rad/s plausible for sw_*)"},
    "mag": {"signal": "magnetometer", "columns": ["Frame", "Timestamp", "X", "Y", "Z"],
            "shape_cols": 5, "units": "UNKNOWN"},
    "hr": {"signal": "heart_rate_derived_bpm", "columns": ["Frame", "Timestamp", "Heart-rate"],
           "shape_cols": 3, "units": "bpm (derived; NOT raw PPG)"},
}
NOMINAL_HZ = {"sw": 100.0, "sp": 210.0, "sp_mag": 100.0, "eb": 90.0, "hr": 1.0}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _kind(fname: str) -> str:
    if "_hr" in fname:
        return "hr"
    if "_mag" in fname:
        return "mag"
    if "_acc" in fname:
        return "acc"
    if "_gyr" in fname:
        return "gyr"
    raise ValueError(f"unrecognized stream filename: {fname}")


def _device(fname: str) -> str:
    for k in ("sw_l", "sw_r", "sp_r", "eb_l"):
        if f"_{k}_" in fname:
            return k
    raise ValueError(f"unrecognized device prefix: {fname}")


def _nominal_hz(device: str, kind: str) -> Optional[float]:
    if kind == "hr":
        return NOMINAL_HZ["hr"]
    if kind == "mag":
        return NOMINAL_HZ["sp_mag"] if device.startswith("sp") else None
    return NOMINAL_HZ.get(device.split("_")[0])


def resolve_data_root(data_root: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the MM-Fit acquisition root.

    Explicit arguments win; otherwise S2S_MMFIT_DATA_ROOT is used when set,
    falling back to the local default. User paths are expanded.
    """
    if data_root is None:
        env_root = os.environ.get(DATA_ROOT_ENV_VAR)
        data_root = Path(env_root) if env_root else DEFAULT_DATA_ROOT
    return Path(data_root).expanduser()


def _session_dir(data_root: Optional[Union[str, Path]], session: str) -> Tuple[Path, Path]:
    root = resolve_data_root(data_root)
    wdir = root / "mm-fit" / session
    if not wdir.is_dir():
        raise FileNotFoundError(
            f"MM-Fit {session} data directory not found: {wdir} "
            f"(set {DATA_ROOT_ENV_VAR} to the acquisition root)")
    return root, wdir


def load_wearable_streams(data_root: Optional[Union[str, Path]] = None,
                           session: str = SESSION_ID) -> Dict[str, Dict[str, Any]]:
    """Load every acquired wearable .npy stream independently (no merging)."""
    root, wdir = _session_dir(data_root, session)
    streams: Dict[str, Dict[str, Any]] = {}
    for f in sorted(wdir.glob("*.npy")):
        kind, dev = _kind(f.name), _device(f.name)
        arr = np.load(f)
        expected_cols = SENSOR_INFO[kind]["shape_cols"]
        if arr.ndim != 2 or arr.shape[1] != expected_cols:
            raise ValueError(f"{f.name}: expected (N,{expected_cols}), got {arr.shape}")
        streams[f.name] = {
            "filename": f.name,
            "device_key": dev,
            "device_label": DEVICE_INFO[dev]["label"],
            "device_kind": DEVICE_INFO[dev]["kind"],
            "unit_evidence": DEVICE_INFO[dev]["unit_evidence"],
            "sensor": SENSOR_INFO[kind]["signal"],
            "sensor_kind": kind,
            "columns": SENSOR_INFO[kind]["columns"],
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "units": SENSOR_INFO[kind]["units"],
            "candidate_nominal_hz_hint": _nominal_hz(dev, kind),
            "clock_domain": "unix_epoch_ms",
            "frame_domain": "left_rgbd_camera_frame_index",
            "participant_id": "UNKNOWN (w19 participant mapping undocumented)",
            "session_id": session,
            "array": arr,
            "source_path": str(f.relative_to(root)),
            "sha256": _sha256(f),
        }
    return streams


def load_labels(data_root: Optional[Union[str, Path]] = None,
                session: str = SESSION_ID) -> List[Dict[str, Any]]:
    import csv
    _, wdir = _session_dir(data_root, session)
    path = wdir / f"{session}_labels.csv"
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            rows.append({"start_frame": int(r[0]), "end_frame": int(r[1]),
                         "repetitions": int(r[2]), "activity": r[3]})
    return rows


def load_video_pts(pts_csv: Path) -> List[Tuple[int, float]]:
    """Robust PTS parse: first/last numeric CSV token per line."""
    rows = []
    with open(pts_csv) as f:
        for line in f:
            nums = []
            for tok in line.rstrip("\n").split(","):
                try:
                    nums.append(float(tok.strip()))
                except ValueError:
                    continue
            if len(nums) >= 2:
                rows.append((int(nums[0]), nums[-1]))
    return rows


INT64_MIN_F = float(-2**63)      # exactly representable
INT64_MAX_EXCL_F = float(2**63)  # 2**63-1 rounds to 2**63 in float64; use exclusive bound


def _validate_index_column(a: np.ndarray, col: int, name: str,
                           require_int64: bool = False) -> np.ndarray:
    """Validate finiteness and integer-valuedness BEFORE any int casting.
    With require_int64 also enforce signed-int64 representability using
    float-exact bounds (no precision error at the boundary)."""
    v = a[:, col]
    if not np.isfinite(v).all():
        raise ValueError(f"{name}: column {col} contains NaN/Inf; refusing int conversion")
    if not np.all(v == np.floor(v)):
        raise ValueError(f"{name}: column {col} has non-integer values")
    if require_int64 and (v.min() < INT64_MIN_F or v.max() >= INT64_MAX_EXCL_F):
        raise ValueError(f"{name}: column {col} outside signed int64 range")
    return v


def stream_stats(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Per-stream integrity statistics (no resampling, no invented timing)."""
    a = rec["array"]
    fname = rec["filename"]
    fr = _validate_index_column(a, 0, fname)
    ts_f = _validate_index_column(a, 1, fname, require_int64=True)
    ts = ts_f.astype(np.int64)  # safe: validated finite + integer-valued
    vals = a[:, 2:]
    dt = np.diff(ts)
    order = np.argsort(ts, kind="stable")
    tss = ts[order]
    u, counts = np.unique(tss, return_counts=True)
    multi = counts[counts > 1]
    # classify duplication mechanism on a deterministic sample of dup groups
    identical = different = 0
    if len(multi):
        bounds = np.concatenate(([0], np.cumsum(counts)))
        idxs = np.nonzero(counts > 1)[0]
        if len(idxs) > 5000:
            idxs = idxs[np.linspace(0, len(idxs) - 1, 5000).astype(int)]
        vss = a[order, 2:]
        for gi in idxs:
            grp = vss[bounds[gi]:bounds[gi + 1]]
            if np.allclose(grp, grp[0], rtol=0, atol=0):
                identical += 1
            else:
                different += 1
    sampled = identical + different
    # OBSERVED pattern only - causal mechanism is an interpretation, not proven.
    if sampled == 0:
        pattern = "NO_DUPLICATE_GROUPS"
    elif identical == 0:
        pattern = "ALL_DUP_GROUPS_DIFFER_IN_VALUES"
    elif different == 0:
        pattern = "ALL_DUP_GROUPS_IDENTICAL_VALUES"
    else:
        pattern = "MIXED_DUP_GROUP_VALUE_PATTERNS"
    causal = {
        "ALL_DUP_GROUPS_DIFFER_IN_VALUES":
            "consistent with ms quantization of distinct samples; not proven",
        "ALL_DUP_GROUPS_IDENTICAL_VALUES":
            "consistent with held/duplicated or batch-delivered samples; not proven",
        "MIXED_DUP_GROUP_VALUE_PATTERNS":
            "both patterns present; causal mechanism unresolved",
        "NO_DUPLICATE_GROUPS": "no duplication observed",
    }[pattern]
    span_s = (float(ts.max()) - float(ts.min())) / 1000.0  # float math avoids int64 overflow
    n = len(a)
    ts_unique = len(u)
    # nominal rate: only documentary/dataset-level claims; never used for pass/fail
    nom = rec.get("candidate_nominal_hz_hint")
    cand = {"value_hz": nom, "source": "MM-Fit dataset documentation (device-class nominal)",
            "verified_for_this_device": False,
            "confidence": "LOW for sp_r (hardware UNKNOWN); MODERATE for documented device classes"}
    return {
        "filename": rec["filename"], "device": rec["device_label"],
        "sensor": rec["sensor"], "shape": rec["shape"], "dtype": rec["dtype"],
        "units": rec["units"], "candidate_nominal_hz_hint": rec.get("candidate_nominal_hz_hint"),
        "n_records": int(n),
        "observed_record_rate_hz": float(n / span_s) if span_s > 0 else None,
        "observed_distinct_ts_rate_hz": float(ts_unique / span_s) if span_s > 0 else None,
        "record_rate_note": "records/elapsed-span estimate; NOT proof of physical sensor rate when timestamps are duplicated/quantized",
        "verified_nominal_hz": None,
        "candidate_reference_hz": cand,
        "measured_hz": float(n / span_s) if span_s > 0 else None,  # deprecated alias
        "ts_min": float(ts.min()), "ts_max": float(ts.max()),
        "ts_clock_domain": "unix_epoch_ms",
        "ts_unique": int(len(u)), "ts_duplicates": int(len(ts) - len(u)),
        "ts_reversals": int((dt < 0).sum()),
        "zero_dt_fraction": float((dt == 0).mean()),
        "dt_ms_median": float(np.median(dt)), "dt_ms_mean": float(dt.mean()),
        "dt_ms_p1": float(np.percentile(dt, 1)), "dt_ms_p99": float(np.percentile(dt, 99)),
        "dt_ms_min": float(dt.min()), "dt_ms_max": float(dt.max()),
        "dup_observed_pattern": pattern,
        "dup_causal_interpretation": causal,
        "dup_mechanism": pattern,  # backward-compat alias (observed pattern only)
        "dup_confidence": "LOW (pattern measured; cause inferred, not proven)",
        "dup_groups_identical_values": identical,
        "dup_groups_different_values": different,
        "frame_min": float(fr.min()), "frame_max": float(fr.max()),
        "frame_unique": int(len(np.unique(fr))),
        "frame_gaps": int((fr.max() - fr.min() + 1) - len(np.unique(fr))),
        "frame_reversals": int((np.diff(fr) < 0).sum()),
        "finite_rows": int(np.isfinite(vals).all(axis=1).sum()),
        "nan_inf_cells": int((~np.isfinite(vals)).sum()),
        "per_sample_ts_quality": "DEGRADED" if len(multi) else "OK",
        "sha256": rec["sha256"], "source_path": rec["source_path"],
    }


def build_dataset_record(streams: Dict[str, Dict[str, Any]],
                         labels: List[Dict[str, Any]],
                         modalities_present: List[str],
                         provenance: Dict[str, Any]) -> qs.DatasetRecord:
    return qs.DatasetRecord(
        record_id="mmfit_w19",
        dataset_id=DATASET_ID,
        embodiment="HUMAN",
        source_type="MMFIT_NPY_CSV_MP4",
        sample_count=sum(int(s["shape"][0]) for s in streams.values()),
        episode_id=None,
        session_id=SESSION_ID,
        modalities_present=modalities_present,
        modalities_expected=["imu_accel", "imu_gyro", "magnetometer", "heart_rate_derived",
                             "rgb_video", "depth_video", "pose_2d", "pose_3d", "labels"],
        provenance={**provenance,
                    "sample_count_convention": "sum of record counts across acquired modality streams; NOT unique synchronized timestamps"},
        clock_domains={
            "wearable_epoch_ms": {"unit": "unix_epoch_milliseconds",
                                  "scope": "all wearable Timestamp columns",
                                  "absolute": True},
            "video_relative_pts": {"unit": "seconds_relative",
                                   "scope": "MP4 PTS only; NOT an absolute clock",
                                   "absolute": False},
            "camera_frame_index": {"unit": "frame_index",
                                   "scope": "wearable Frame col + labels + video frames",
                                   "boundary_uncertainty_frames": 1},
        },
    )


def evidence_ref(path: str, sha: Optional[str] = None, **extra) -> Dict[str, Any]:
    ref = {"type": "file", "path": path}
    if sha:
        ref["sha256"] = sha
    ref.update(extra)
    return ref


def check(check_id: str, domain: str, scope: str, status: str, reason: str,
          explanation: str, observed: Any = None, unit: Optional[str] = None,
          refs: Optional[List[Dict[str, Any]]] = None,
          limitations: Optional[List[str]] = None,
          not_assessed: Optional[str] = None,
          clock_domain: Optional[str] = None,
          subject: Optional[Dict[str, Any]] = None) -> qs.CheckResult:
    return qs.CheckResult(
        check_id=check_id, check_version=ADAPTER_VERSION, domain=domain,
        scope=scope, status=status, reason_code=reason, explanation=explanation,
        threshold_source="DEVICE_METADATA" if status != "NOT_ASSESSED" else "NOT_APPLICABLE",
        observed_value=observed, unit=unit,
        evidence_refs=tuple(refs or []),
        limitations=tuple(limitations or []),
        not_assessed_reason=not_assessed, clock_domain=clock_domain,
        subject=subject or {})


def assemble_report(record: qs.DatasetRecord, checks: List[qs.CheckResult],
                    modality_availability: Dict[str, str],
                    metadata: Dict[str, Any]) -> qs.AssessmentReport:
    return qs.AssessmentReport(
        report_id="mmfit_w19_assessment",
        record=record,
        profile_name="mmfit_multimodal_evidence",
        profile_version=ADAPTER_VERSION,
        adapter_name=ADAPTER_NAME,
        adapter_version=ADAPTER_VERSION,
        acquisition_availability="PARTIAL",
        checks=tuple(checks),
        modality_availability=modality_availability,
        metadata=metadata,
    )
