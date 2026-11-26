"""Frame-level wearable/video association for MM-Fit w19.

Wearable Frame column = left RGB-D camera frame index (dataset convention).
Video frame indices are 0..52560; wearable max is 52561 -> +/-1 frame
(~33 ms) boundary uncertainty. No sample-exact cross-modal joins.
"""
from typing import Any, Dict, List, Optional

import numpy as np

from experiments.cross_embodiment.milestone_3f2b import mmfit_adapter as ma

VIDEO_MAX_INDEX = 52560
FRAME_BOUNDARY_UNCERTAINTY = 1  # frames (~33 ms at 30fps)


def frame_association(streams: Dict[str, Dict[str, Any]],
                      labels: List[Dict[str, Any]],
                      video_frames: int) -> Dict[str, Any]:
    per_stream = []
    for name, rec in streams.items():
        a = rec["array"]
        fr = a[:, 0]
        fmin, fmax = int(fr.min()), int(fr.max())
        per_stream.append({
            "stream": name, "device": rec["device_label"], "sensor": rec["sensor"],
            "frame_min": fmin, "frame_max": fmax,
            "frame_unique": int(len(np.unique(fr))),
            "video_index_in_range_0based": bool(0 <= fmin and fmax <= VIDEO_MAX_INDEX),
            "max_minus_video_max": fmax - VIDEO_MAX_INDEX,
        })
    label_frames = [l["start_frame"] for l in labels] + [l["end_frame"] for l in labels]
    all_fr = [s["frame_min"] for s in per_stream] + [s["frame_max"] for s in per_stream]
    payload = {
        "method": "wearable Frame column == left RGB-D camera frame index (dataset convention); "
                  "video frame index is MP4 decode order 0..52560",
        "frame_index_domain": "camera_frame_index",
        "boundary_uncertainty_frames": FRAME_BOUNDARY_UNCERTAINTY,
        "boundary_uncertainty_ms_approx": round(FRAME_BOUNDARY_UNCERTAINTY * 1000 / 30, 1),
        "zero_vs_one_based": "AMBIGUOUS: wearable max 52561 vs video max index 52560; "
                             "labels max 52560 consistent with 0-based video indexing; "
                             "no offset applied",
        "video_has_absolute_clock": False,
        "drift": "NOT_MEASURED",
        "sample_exact_sync_claimed": False,
        "per_stream": per_stream,
        "label_frame_min": min(label_frames), "label_frame_max": max(label_frames),
        "frame_span_covered": [min(all_fr), max(all_fr)],
    }
    return payload


def frame_checks(payload: Dict[str, Any],
                 streams: Dict[str, Dict[str, Any]],
                 extra_refs: Optional[List[Dict[str, Any]]] = None) -> List[Any]:
    # Aggregate finding must cite ALL contributing evidence: every wearable
    # stream, labels CSV, MP4 video, and PTS export (relative paths + sha256).
    refs = [ma.evidence_ref(s["source_path"], s["sha256"]) for s in streams.values()]
    refs += list(extra_refs or [])
    checks = [ma.check(
        "MMFIT_FRAME_VIDEO_ASSOCIATION", "COMMON", "EPISODE", "WARNING",
        "FRAME_INDEX_ASSOCIATION_WITH_BOUNDARY_UNCERTAINTY",
        "Wearable Frame column associates to camera frame index; video indices "
        "0-52560 vs wearable max 52561 -> +/-1 frame (~33ms) boundary ambiguity "
        "(zero- vs one-based unresolved). No offset applied; no sample-exact join.",
        observed={"boundary_uncertainty_frames": payload["boundary_uncertainty_frames"],
                  "boundary_uncertainty_ms_approx": payload["boundary_uncertainty_ms_approx"],
                  "zero_vs_one_based": payload["zero_vs_one_based"],
                  "drift": payload["drift"],
                  "sample_exact_sync_claimed": False,
                  "n_wearable_stream_refs": len(streams),
                  "n_total_evidence_refs": len(refs)},
        refs=refs,
        limitations=["frame-index association only; video PTS is relative",
                     "first/last boundary mapping unresolved",
                     "drift NOT_MEASURED"],
        subject={"session_id": "w19"})]

    # Depth acquisition state.
    checks.append(ma.check(
        "MMFIT_DEPTH_ACQUISITION", "COMMON", "EPISODE", "NOT_ASSESSED",
        "RAW_UNAVAILABLE_LOCALLY",
        "Depth video not locally acquired. Source exists: Zenodo record 7607736 "
        "w19_depth.mp4 (19,014,084 bytes declared). Not decoded, not timestamp-assessed, "
        "not sync-assessed. RGB assessed independently.",
        observed={"source_record": "zenodo:7607736", "declared_bytes": 19014084,
                  "local_state": "RAW_UNAVAILABLE"},
        refs=[{"type": "source_record", "record": "zenodo:7607736",
               "file": "w19_depth.mp4"}],
        limitations=["depth modality cannot be assessed until acquired"],
        not_assessed="DEPTH_NOT_ACQUIRED",
        subject={"session_id": "w19", "sample_id": "w19_depth.mp4"}))
    return checks
