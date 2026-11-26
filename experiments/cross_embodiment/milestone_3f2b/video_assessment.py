"""Independent RGB video metadata/PTS integrity assessment for MM-Fit w19.

Video PTS is relative (starts at 0) and is NOT an absolute wearable clock.
PhysicsEngine is never applied to video.
"""
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.cross_embodiment.milestone_3f2b import mmfit_adapter as ma

EXPECTED = {"codec": "h264", "width": 1280, "height": 720, "fps": 30.0,
            "frames": 52561}
PTS_EXPORT_REL = ("experiments/cross_embodiment/milestone_3f2a/verification/"
                  "w19_rgb_frame_timestamps.csv")


def parse_frame_rate(value: Any) -> Optional[float]:
    """Parse an ffprobe rational frame-rate string like '30/1' via Fraction.
    Returns None for empty/invalid input."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return float(Fraction(value.strip()))
    except (ValueError, ZeroDivisionError):
        return None


def probe_video(video_path: Path) -> Dict[str, Any]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,size:stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames,time_base",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


MIN_PTS_ROWS = 3  # need >=3 points for an interval distribution


def video_stats(pts_rows: List[Tuple[int, float]]) -> Dict[str, Any]:
    if not pts_rows:
        return {"rows": 0, "insufficient": True, "empty": True}
    if len(pts_rows) < MIN_PTS_ROWS:
        return {"rows": len(pts_rows), "insufficient": True, "empty": False}
    pts_t = np.array([r[1] for r in pts_rows])
    pts_c = np.array([r[0] for r in pts_rows])
    d = np.diff(pts_t)
    return {
        "rows": len(pts_rows), "empty": False,
        "pts_first": float(pts_t[0]), "pts_last": float(pts_t[-1]),
        "pts_counter_first": int(pts_c[0]), "pts_counter_last": int(pts_c[-1]),
        "strictly_increasing": bool((d > 0).all()),
        "duplicate_pts": int(len(pts_t) - len(np.unique(pts_t))),
        "reversals": int((d < 0).sum()),
        "interval_s_median": float(np.median(d)),
        "interval_s_min": float(d.min()), "interval_s_max": float(d.max()),
        "interval_s_std": float(d.std()),
        "interval_s_p1": float(np.percentile(d, 1)),
        "interval_s_p99": float(np.percentile(d, 99)),
        "derived_fps_from_pts": float(1.0 / np.median(d)) if np.median(d) > 0 else None,
        "insufficient": False,
    }


def video_checks(video_path: Path, pts_csv: Path,
                 probe: Optional[Dict[str, Any]] = None,
                 pts_rows: Optional[List[Tuple[int, float]]] = None,
                 pts_sha256: Optional[str] = None,
                 video_sha256: Optional[str] = None) -> tuple:
    """Returns (checks, payload)."""
    probe = probe or probe_video(video_path)
    pts_rows = pts_rows if pts_rows is not None else ma.load_video_pts(pts_csv)
    vsha = video_sha256 if video_sha256 is not None else sha256_file(video_path)
    st = video_stats(pts_rows)
    stream = (probe.get("streams") or [{}])[0]
    fmt = probe.get("format", {})

    codec_ok = stream.get("codec_name") == EXPECTED["codec"]
    res_ok = (stream.get("width") == EXPECTED["width"]
              and stream.get("height") == EXPECTED["height"])
    frames_ok = int(stream.get("nb_frames") or 0) == EXPECTED["frames"]
    pts_insufficient = bool(st.get("insufficient"))
    pts_monotonic = (not pts_insufficient) and st["strictly_increasing"] and st["reversals"] == 0
    pts_count_ok = (not pts_insufficient) and st["rows"] == EXPECTED["frames"]
    fps_meta = parse_frame_rate(stream.get("r_frame_rate"))
    fps_ok = fps_meta is not None and abs(fps_meta - EXPECTED["fps"]) < 1e-6
    # Cross-check reported FPS against PTS-derived rate (tolerance 0.5 fps)
    pts_fps = st.get("derived_fps_from_pts")
    pts_consistent = (pts_fps is not None and fps_meta is not None
                      and abs(pts_fps - fps_meta) < 0.5)

    problems = []
    if not codec_ok:
        problems.append("codec")
    if not res_ok:
        problems.append("resolution")
    if not frames_ok:
        problems.append("frame_count")
    if fps_meta is None:
        problems.append("frame_rate_unparseable")
    elif not fps_ok:
        problems.append("frame_rate_mismatch")
    if pts_insufficient:
        problems.append("pts_insufficient_rows")
    else:
        if not pts_monotonic:
            problems.append("pts_nonmonotonic")
        if not pts_count_ok:
            problems.append("pts_count_mismatch")
        if not pts_consistent:
            problems.append("fps_pts_inconsistent")

    integrity = "PASS" if not problems else "WARNING"
    checks = [ma.check(
        "MMFIT_RGB_VIDEO_METADATA_PTS_INTEGRITY", "COMMON", "EPISODE", integrity,
        "METADATA_PTS_CONSISTENT" if integrity == "PASS" else "METADATA_PTS_ANOMALY",
        f"w19_rgb.mp4 metadata+PTS: {stream.get('codec_name')} "
        f"{stream.get('width')}x{stream.get('height')} reported_fps={stream.get('r_frame_rate')} "
        f"({fps_meta}), pts_derived_fps={pts_fps and round(pts_fps,3)}, "
        f"frames={st.get('rows')}, PTS monotonic={pts_monotonic}. "
        f"Problems: {problems or 'none'}. Pixel-level decode NOT exhaustively verified.",
        observed={"codec": stream.get("codec_name"), "width": stream.get("width"),
                  "height": stream.get("height"), "reported_r_frame_rate": stream.get("r_frame_rate"),
                  "reported_fps_parsed": fps_meta, "expected_fps": EXPECTED["fps"],
                  "pts_derived_fps": pts_fps, "pts_fps_consistent": pts_consistent,
                  "frames_meta": stream.get("nb_frames"), "frames_pts": st.get("rows"),
                  "duration_s": float(fmt.get("duration", 0)),
                  "pts_duplicate_count": st.get("duplicate_pts"),
                  "problems": problems},
        refs=[ma.evidence_ref("video/w19_rgb.mp4", vsha),
              ma.evidence_ref(PTS_EXPORT_REL, pts_sha256)],
        limitations=["pixel-level decode not exhaustively verified",
                     "metadata/PTS consistency only; no frame-content checks",
                     "no depth stream in this artifact"],
        subject={"session_id": "w19", "sample_id": "w19_rgb.mp4"})]

    checks.append(ma.check(
        "MMFIT_VIDEO_CLOCK_DOMAIN", "COMMON", "EPISODE", "NOT_ASSESSED",
        "RELATIVE_PTS_NO_ABSOLUTE_CLOCK",
        "Video PTS is relative (starts at 0.0s) and carries no absolute wall-clock "
        "mapping. It must not be assigned the wearable epoch-ms clock. "
        "Wearable-video association is via camera frame index only.",
        observed={"pts_first_s": st.get("pts_first"), "pts_last_s": st.get("pts_last"),
                  "time_base": stream.get("time_base")},
        refs=[ma.evidence_ref(PTS_EXPORT_REL, pts_sha256)],
        limitations=["no absolute video clock exists in acquired artifacts"],
        not_assessed="NO_ABSOLUTE_VIDEO_CLOCK",
        clock_domain=None,
        subject={"session_id": "w19", "sample_id": "w19_rgb.mp4"}))

    payload = {
        "file": "video/w19_rgb.mp4", "sha256": vsha,
        "pts_export": PTS_EXPORT_REL, "pts_export_sha256": pts_sha256,
        "probe": probe, "stats": st,
        "clock_domain": "relative_pts_seconds",
        "absolute_clock": False,
        "depth_status": "RAW_UNAVAILABLE_LOCALLY (source: Zenodo 7607736 w19_depth.mp4 19014084B, not acquired)",
    }
    return checks, payload


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
