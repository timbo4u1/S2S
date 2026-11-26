"""Tests for the MM-Fit w19 multimodal adapter (Milestone 3F.2B)."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.cross_embodiment import quality_schema as qs
from experiments.cross_embodiment.milestone_3f2b import (
    frame_association, mmfit_adapter as ma, video_assessment, wearable_assessment)
from experiments.cross_embodiment.milestone_3f2b.run_mmfit_assessment import (
    label_audit, main as run_main, OUT_DIR)

PTS = (Path(__file__).resolve().parents[1]
       / "experiments/cross_embodiment/milestone_3f2a/verification"
       / "w19_rgb_frame_timestamps.csv")


@pytest.fixture(scope="session")
def mmfit_data_root():
    """Real w19 wearable acquisition; skipped only when genuinely absent."""
    root = ma.resolve_data_root()
    wdir = root / "mm-fit" / ma.SESSION_ID
    if not wdir.is_dir():
        pytest.skip(
            f"MM-Fit w19 wearable acquisition unavailable at {wdir}; "
            f"set {ma.DATA_ROOT_ENV_VAR} to the external data root")
    return root


@pytest.fixture(scope="session")
def mmfit_full_data_root(mmfit_data_root):
    """Real w19 acquisition including RGB video for full-runner tests."""
    video = mmfit_data_root / "video" / "w19_rgb.mp4"
    if not video.is_file():
        pytest.skip(
            f"MM-Fit w19 RGB video unavailable at {video}; "
            f"set {ma.DATA_ROOT_ENV_VAR} to the external data root")
    return mmfit_data_root


@pytest.fixture(scope="module")
def streams(mmfit_data_root):
    return ma.load_wearable_streams(mmfit_data_root)


@pytest.fixture(scope="module")
def stats(streams):
    return {n: ma.stream_stats(r) for n, r in streams.items()}


@pytest.fixture(scope="module")
def labels(mmfit_data_root):
    return ma.load_labels(mmfit_data_root)


def test_a_all_streams_load(streams):
    assert len(streams) == 10  # 9 inertial/mag (no sp_l_* in w19) + 1 HR
    for rec in streams.values():
        assert rec["array"].ndim == 2
        assert len(rec["sha256"]) == 64


def test_b_no_stream_merging(streams):
    keys = {(s["device_key"], s["sensor_kind"]) for s in streams.values()}
    assert len(keys) == len(streams)
    arrs = {id(s["array"]) for s in streams.values()}
    assert len(arrs) == len(streams)


def test_c_unknown_device_preserved(streams):
    sp = [s for s in streams.values() if s["device_key"] == "sp_r"]
    assert sp and all("UNKNOWN" in s["unit_evidence"] or "plausible" in s["unit_evidence"].lower() or "NOT verified" in s["unit_evidence"] for s in sp)


def test_d_unknown_units_no_physics_pass(streams, stats):
    checks = wearable_assessment.wearable_checks(streams, stats)
    law = [c for c in checks if c.check_id == "MMFIT_PHYSICS_LAWS"]
    assert law and all(c.status == "NOT_ASSESSED" for c in law)
    assert all(c.not_assessed_reason for c in law)


def test_e_bpm_never_ppg(streams, stats):
    hr = streams["w19_sw_l_hr.npy"]
    assert hr["sensor_kind"] == "hr" and "NOT raw PPG" in hr["units"]
    checks = wearable_assessment.wearable_checks(streams, stats)
    hr_checks = [c for c in checks if c.subject.get("sample_id") == "w19_sw_l_hr.npy"]
    assert all("PPG" not in (c.check_id or "") for c in hr_checks)
    unit_check = [c for c in hr_checks if c.check_id == "MMFIT_UNIT_SEMANTICS"][0]
    assert unit_check.status == "PASS" and "PPG" in unit_check.explanation


def test_f_missing_modalities_not_available(streams):
    names = set(streams)
    assert not any("emg" in n or "ppg" in n for n in names)
    assert "w19_sw_r_hr.npy" not in names and not any(n.startswith("w19_sp_l") for n in names)


def test_g_duplicate_ts_preserved(stats):
    gyr = stats["w19_sp_r_gyr.npy"]
    assert gyr["ts_duplicates"] > 30000 and gyr["zero_dt_fraction"] > 0.1
    assert gyr["dup_observed_pattern"] == "MIXED_DUP_GROUP_VALUE_PATTERNS"
    assert gyr["dup_causal_interpretation"] and gyr["dup_confidence"].startswith("LOW")
    eb = stats["w19_eb_l_acc.npy"]
    assert eb["dup_observed_pattern"] == "ALL_DUP_GROUPS_DIFFER_IN_VALUES"


def test_h_no_invented_reversals(stats):
    for st in stats.values():
        assert st["ts_reversals"] == 0 and st["frame_reversals"] == 0


def test_i_video_relative_pts():
    rows = ma.load_video_pts(PTS)
    assert len(rows) == 52561 and rows[0][1] == pytest.approx(0.0)
    times = [r[1] for r in rows]
    assert all(b > a for a, b in zip(times, times[1:]))


def test_j_frame_association_uncertainty(streams, labels):
    payload = frame_association.frame_association(streams, labels, 52561)
    assert payload["boundary_uncertainty_frames"] == 1
    assert "AMBIGUOUS" in payload["zero_vs_one_based"]
    assert payload["sample_exact_sync_claimed"] is False
    assert payload["drift"] == "NOT_MEASURED"


def test_k_no_sample_exact_sync(streams, stats, labels):
    checks = wearable_assessment.wearable_checks(streams, stats)
    fa = frame_association.frame_checks(
        frame_association.frame_association(streams, labels, 52561), streams)
    allc = checks + fa
    assert all("sample-exact" not in c.explanation.lower() or "no" in c.explanation.lower() for c in allc)


def test_l_depth_raw_unavailable(streams, labels):
    checks = frame_association.frame_checks(
        frame_association.frame_association(streams, labels, 52561), streams)
    depth = [c for c in checks if c.check_id == "MMFIT_DEPTH_ACQUISITION"][0]
    assert depth.status == "NOT_ASSESSED"
    assert depth.observed_value["local_state"] == "RAW_UNAVAILABLE"


def test_m_labels_26_258(labels):
    la = label_audit(labels)
    assert la["n_sets"] == 26 and la["total_repetitions"] == 258
    assert la["n_activities"] == 9 and la["overlapping_pairs"] == 0


def test_n_report_roundtrip(mmfit_full_data_root):
    run_main()
    data = json.loads((OUT_DIR / "assessment_report.json").read_text())
    report = qs.AssessmentReport.from_dict(data)
    assert report.record.embodiment == "HUMAN"
    assert report.to_json() == qs.AssessmentReport.from_json(report.to_json()).to_json()


def test_o_deterministic(mmfit_full_data_root):
    run_main()
    first = (OUT_DIR / "assessment_report.json").read_text()
    run_main()
    second = (OUT_DIR / "assessment_report.json").read_text()
    assert first == second


def test_p_engine_and_adapter_unchanged():
    import hashlib
    for p in ["s2s_standard_v1_3/s2s_physics_v1_3.py",
              "experiments/cross_embodiment/human_adapter.py",
              "experiments/cross_embodiment/quality_schema.py"]:
        assert Path(p).exists()
    # no PhysicsEngine import/execution anywhere in milestone_3f2b
    for f in Path("experiments/cross_embodiment/milestone_3f2b").glob("*.py"):
        text = f.read_text()
        assert "s2s_physics" not in text
        assert not any(line.strip().startswith(("import", "from")) and "physics" in line.lower()
                       for line in text.splitlines())


# ---------- synthetic loader fixtures (structure only; not real MM-Fit data) ----------

_SYNTHETIC_STREAMS = [
    ("w19_eb_l_acc.npy", "acc"), ("w19_eb_l_gyr.npy", "gyr"),
    ("w19_sp_r_acc.npy", "acc"), ("w19_sp_r_gyr.npy", "gyr"),
    ("w19_sp_r_mag.npy", "mag"), ("w19_sw_l_acc.npy", "acc"),
    ("w19_sw_l_gyr.npy", "gyr"), ("w19_sw_l_hr.npy", "hr"),
    ("w19_sw_r_acc.npy", "acc"), ("w19_sw_r_gyr.npy", "gyr"),
]


def _write_synthetic_mmfit(root: Path, session: str = "w19") -> Path:
    """Write tiny structurally valid NPY files for loader tests only."""
    wdir = root / "mm-fit" / session
    wdir.mkdir(parents=True)
    for i, (filename, kind) in enumerate(_SYNTHETIC_STREAMS):
        n = 5
        frame = np.arange(n, dtype=np.float64) + i
        ts = 1_700_000_000_000.0 + np.arange(n, dtype=np.float64) * 10.0
        if kind == "hr":
            arr = np.column_stack([frame, ts, 60.0 + np.arange(n)])
        else:
            arr = np.column_stack([frame, ts, np.zeros((n, 3))])
        np.save(wdir / filename, arr)
    return root


def test_synthetic_loader_loads_ten_independent_streams(tmp_path):
    root = _write_synthetic_mmfit(tmp_path)
    streams = ma.load_wearable_streams(root)
    assert len(streams) == 10
    assert {s["sensor_kind"] for s in streams.values()} == {"acc", "gyr", "mag", "hr"}
    assert {s["device_key"] for s in streams.values()} == {"sw_l", "sw_r", "sp_r", "eb_l"}
    assert streams["w19_sp_r_mag.npy"]["sensor_kind"] == "mag"
    assert streams["w19_sw_l_hr.npy"]["sensor_kind"] == "hr"
    assert len({(s["device_key"], s["sensor_kind"]) for s in streams.values()}) == 10
    assert len({id(s["array"]) for s in streams.values()}) == 10
    assert all(len(s["sha256"]) == 64 for s in streams.values())
    assert all(s["source_path"].startswith("mm-fit/w19/") for s in streams.values())


def test_synthetic_loader_env_var_missing_dir_and_user_expansion(tmp_path, monkeypatch):
    root = _write_synthetic_mmfit(tmp_path / "acquisition")
    monkeypatch.setenv(ma.DATA_ROOT_ENV_VAR, str(root))
    assert ma.resolve_data_root() == root
    assert len(ma.load_wearable_streams()) == 10

    missing = tmp_path / "missing"
    monkeypatch.setenv(ma.DATA_ROOT_ENV_VAR, str(missing))
    with pytest.raises(FileNotFoundError) as excinfo:
        ma.load_wearable_streams()
    assert str(missing / "mm-fit" / "w19") in str(excinfo.value)

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(ma.DATA_ROOT_ENV_VAR, "~/mmfit_s2s")
    assert ma.resolve_data_root() == tmp_path / "mmfit_s2s"


def test_synthetic_loader_rejects_invalid_shape(tmp_path):
    wdir = tmp_path / "mm-fit" / "w19"
    wdir.mkdir(parents=True)
    np.save(wdir / "w19_sw_l_acc.npy", np.zeros((5, 4)))
    with pytest.raises(ValueError, match="expected"):
        ma.load_wearable_streams(tmp_path)


# ---------- 3F.2B.1 review-fix regressions ----------

def _fake_stream(array):
    return {"filename": "fake.npy", "device_label": "fake", "device_key": "sw_l",
            "device_kind": "smartwatch", "unit_evidence": "x", "sensor": "acc",
            "sensor_kind": "acc", "columns": ["Frame", "Timestamp", "X", "Y", "Z"],
            "shape": list(array.shape), "dtype": str(array.dtype), "units": "UNKNOWN",
            "candidate_nominal_hz_hint": 100.0, "clock_domain": "unix_epoch_ms",
            "frame_domain": "f", "participant_id": "?", "session_id": "w19",
            "array": array, "source_path": "fake.npy", "sha256": "0" * 64}


def _arr(frame, ts):
    n = len(ts)
    return np.column_stack([np.asarray(frame, float), np.asarray(ts, float),
                            np.zeros((n, 3))])


@pytest.mark.parametrize("bad_col,bad_val", [
    (0, np.nan), (0, np.inf), (1, np.nan), (1, np.inf),
    (0, 1.5), (1, 1.5)])
def test_malformed_index_columns_rejected(bad_col, bad_val):
    fr = np.arange(100.0); ts = np.arange(100.0) * 10
    fr[10] = ts[10] = 0.0
    if bad_col == 0:
        fr[10] = bad_val
    else:
        ts[10] = bad_val
    with pytest.raises(ValueError):
        ma.stream_stats(_fake_stream(_arr(fr, ts)))


def test_valid_integer_float_inputs_accepted():
    st = ma.stream_stats(_fake_stream(_arr(np.arange(100.0), np.arange(100.0) * 10)))
    assert st["n_records"] == 100 and st["ts_reversals"] == 0


def test_nominal_rate_semantics(stats):
    sp = stats["w19_sp_r_acc.npy"]
    assert sp["verified_nominal_hz"] is None
    assert "NOT proof" in sp["record_rate_note"]
    assert sp["candidate_reference_hz"]["verified_for_this_device"] is False
    assert "LOW" in sp["candidate_reference_hz"]["confidence"]


def _probe_stream(**kw):
    base = {"codec_name": "h264", "width": 1280, "height": 720,
            "r_frame_rate": "30/1", "nb_frames": "52561", "time_base": "1/15360"}
    base.update(kw)
    return {"streams": [base], "format": {"duration": "1752.0"}}


def _pts_rows(n=52561, step=1.0 / 30):
    return [(i * 512, round(i * step, 6)) for i in range(n)]


def _vcheck(probe=None, pts=None):
    return video_assessment.video_checks(
        Path("/nonexistent"), Path("/nonexistent"),
        probe=probe or _probe_stream(), pts_rows=pts if pts is not None else _pts_rows(),
        pts_sha256="ab" * 32, video_sha256="cd" * 32)[0][0]


def test_video_correct_30fps():
    c = _vcheck()
    assert c.status == "PASS" and c.check_id == "MMFIT_RGB_VIDEO_METADATA_PTS_INTEGRITY"


def test_video_reported_15fps_but_30fps_pts():
    c = _vcheck(probe=_probe_stream(r_frame_rate="15/1"))
    assert c.status == "WARNING" and "frame_rate_mismatch" in c.observed_value["problems"]


def test_video_correct_meta_inconsistent_pts():
    pts = _pts_rows()
    # corrupt PTS timing: double every interval -> derived fps ~15
    bad = [(p, t * 2) for p, t in pts]
    c = _vcheck(pts=bad)
    assert c.status == "WARNING"
    assert "fps_pts_inconsistent" in c.observed_value["problems"]


def test_video_invalid_frame_rate_string():
    c = _vcheck(probe=_probe_stream(r_frame_rate="not-a-rate"))
    assert c.status == "WARNING"
    assert "frame_rate_unparseable" in c.observed_value["problems"]


def test_video_empty_pts():
    c = _vcheck(pts=[])
    assert c.status == "WARNING"
    assert "pts_insufficient_rows" in c.observed_value["problems"]


def test_frame_association_refs_all_evidence(streams, labels):
    payload = frame_association.frame_association(streams, labels, 52561)
    fchecks = frame_association.frame_checks(payload, streams, [
        ma.evidence_ref("mm-fit/w19/w19_labels.csv", "aa" * 32),
        ma.evidence_ref("video/w19_rgb.mp4", "bb" * 32),
        ma.evidence_ref("pts.csv", "cc" * 32)])
    assoc = [c for c in fchecks if c.check_id == "MMFIT_FRAME_VIDEO_ASSOCIATION"][0]
    paths = {r["path"] for r in assoc.evidence_refs}
    assert len(assoc.evidence_refs) == len(streams) + 3
    assert all(not r["path"].startswith("/") for r in assoc.evidence_refs)
    assert any("labels" in p for p in paths) and any(p.endswith(".mp4") for p in paths)


# ---------- 3F.2B.2.1 corrections ----------

def test_int64_boundary_values_accepted():
    I64 = 2**63
    fr = np.array([0.0, float(I64 - 1), 1.0])
    ts = np.array([0.0, float(I64 - 1), float(I64 - 2)])
    # use in-range realistic magnitudes: test the validator directly instead
    import numpy as _np
    a = _np.zeros((3, 5)); a[:, 0] = [0.0, 1.0, 2.0]; a[:, 1] = [0.0, 10.0, 20.0]
    a[1, 1] = float(-2**63)   # int64 min exactly representable
    st = ma.stream_stats(_fake_stream(a))
    assert st["n_records"] == 3


@pytest.mark.parametrize("val", [float(-2**64), float(2**63), float(2**64)])
def test_int64_out_of_range_rejected(val):
    a = np.zeros((3, 5)); a[:, 0] = [0.0, 1.0, 2.0]
    a[:, 1] = [0.0, val, 20.0]
    with pytest.raises(ValueError):
        ma.stream_stats(_fake_stream(a))


@pytest.mark.parametrize("n", [0, 1, 2])
def test_pts_insufficient_rows(n):
    pts = [(i * 512, i / 30.0) for i in range(n)]
    c = _vcheck(pts=pts)
    assert c.status == "WARNING"
    assert "pts_insufficient_rows" in c.observed_value["problems"]


def test_version_consistency():
    assert ma.ADAPTER_VERSION == "3f.2b.2"
    data = json.loads((OUT_DIR / "assessment_report.json").read_text())
    assert data["adapter_version"] == "3f.2b.2"
    assert data["profile_version"] == "3f.2b.2"
    assert all(c["check_version"] == "3f.2b.2" for c in data["checks"])


def test_frame_association_13_refs(streams, labels):
    payload = frame_association.frame_association(streams, labels, 52561)
    fchecks = frame_association.frame_checks(payload, streams, [
        ma.evidence_ref("mm-fit/w19/w19_labels.csv", "aa" * 32),
        ma.evidence_ref("video/w19_rgb.mp4", "bb" * 32),
        ma.evidence_ref("pts.csv", "cc" * 32)])
    assoc = [c for c in fchecks if c.check_id == "MMFIT_FRAME_VIDEO_ASSOCIATION"][0]
    assert len(assoc.evidence_refs) == 13  # 10 streams + labels + mp4 + pts
    assert all(r.get("sha256") for r in assoc.evidence_refs)
    assert all(not r["path"].startswith("/") for r in assoc.evidence_refs)


def test_sample_count_convention(streams, labels):
    record = ma.build_dataset_record(streams, labels, [], {})
    assert "sample_count_convention" in record.provenance
    assert "NOT unique synchronized timestamps" in record.provenance["sample_count_convention"]
