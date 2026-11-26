import dataclasses
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "cross_embodiment"))

import quality_schema as qs


def make_record(embodiment="HUMAN", modalities=None, expected=None, clocks=None, episode_id="episode-7"):
    return qs.DatasetRecord(
        record_id=f"{embodiment.lower()}-record-1",
        dataset_id=f"{embodiment.lower()}-dataset",
        episode_id=episode_id,
        session_id="session-3",
        embodiment=embodiment,
        source_type="test_source",
        modalities_present=modalities or ["imu"],
        modalities_expected=expected,
        sample_count=128,
        original_index_range=[0, 127],
        provenance={"source_file": "synthetic.bin", "sha256": "abc123", "adapter": "test-adapter", "adapter_version": "0.1"},
        clock_domains=clocks or {},
    )


def make_check(status="PASS", domain="COMMON", scope="EPISODE", clock_domain=None, evidence_refs=None, not_assessed_reason=None, check_id="numeric_finite", subject=None, observed_value=0.0):
    return qs.CheckResult(
        check_id=check_id,
        check_version="1.0.0",
        domain=domain,
        scope=scope,
        status=status,
        reason_code="OK" if status == "PASS" else status,
        explanation="synthetic check",
        observed_value=observed_value,
        unit="dimensionless",
        expected_bounds={"min": 0.0, "max": 1.0},
        threshold_source="EXPLORATORY_EMPIRICAL",
        evidence_refs=evidence_refs or [],
        limitations=["synthetic fixture"],
        not_assessed_reason=not_assessed_reason,
        clock_domain=clock_domain,
        subject=subject or {"episode_id": "episode-7"},
    )


def make_report(embodiment="HUMAN", checks=None, availability="AVAILABLE", modality_availability=None, clocks=None, summary=None, record_episode_id="episode-7"):
    report = qs.AssessmentReport(
        report_id=f"{embodiment.lower()}-report-1",
        record=make_record(embodiment, clocks=clocks, episode_id=record_episode_id),
        profile_name="synthetic_profile",
        profile_version="0.1.0",
        adapter_name="test_adapter",
        adapter_version="0.1.0",
        acquisition_availability=availability,
        checks=checks if checks is not None else [make_check()],
        modality_availability=modality_availability or {"imu": "AVAILABLE"},
        metadata={"created_at": "2026-01-01T00:00:00Z", "software_revision": "test"},
    )
    if summary is None:
        return report
    data = report.to_dict()
    data["summary"] = summary
    return qs.AssessmentReport.from_dict(data)


def transform_record(source_report_id="human-report-1"):
    return qs.TransformationRecord(
        source_report_id=source_report_id,
        target_robot_id="robot-7",
        target_model_version="model-v1",
        transformation_algorithm="retarget_synth",
        transformation_version="0.1.0",
        source_refs=[{"report_id": source_report_id, "episode_id": "episode-7", "sample_indices": [0, 1, 2]}],
        coordinate_frames={"source": "human_pelvis", "target": "robot_base"},
        calibration_refs=[{"calibration_id": "cal-1", "uri": "cal://synthetic"}],
        mapping_spec={"joints": {"wrist": "joint_6"}},
        time_mapping={"method": "identity", "source_clock": "motion_capture", "target_clock": "robot_state"},
        exclusions=[{"reason": "manual", "interval": [10, 12]}],
        output_content_hash="def456",
        target_report_id="robot-report-1",
    )


class CrossEmbodimentSchemaTests(unittest.TestCase):
    def assert_roundtrip(self, obj):
        text = obj.to_json()
        restored = type(obj).from_json(text)
        self.assertEqual(restored.to_dict(), obj.to_dict())
        return restored

    def test_dataset_record_roundtrip(self):
        rec = make_record("ROBOT", modalities=["joint_state", "action", "camera"], expected=["joint_state", "action", "camera", "force"], clocks={"robot_state_ns": {"unit": "ns"}, "camera_capture_ns": {"unit": "ns"}})
        restored = self.assert_roundtrip(rec)
        self.assertEqual(restored.embodiment, "ROBOT")
        self.assertEqual(restored.clock_domains["robot_state_ns"]["unit"], "ns")

    def test_check_result_roundtrip(self):
        chk = make_check(scope="SAMPLE", subject={"episode_id": "episode-7", "sample_id": "s-42"}, evidence_refs=[{"dataset_path": "x.h5", "episode_id": "episode-7", "sample_index": 42, "content_hash": "h"}])
        restored = self.assert_roundtrip(chk)
        self.assertEqual(restored.evidence_refs[0]["sample_index"], 42)

    def test_transformation_record_roundtrip(self):
        tr = transform_record()
        restored = self.assert_roundtrip(tr)
        self.assertEqual(restored.source_report_id, "human-report-1")
        self.assertEqual(restored.target_robot_id, "robot-7")
        self.assertEqual(restored.time_mapping["target_clock"], "robot_state")

    def test_human_report_serialization(self):
        report = make_report("HUMAN", checks=[make_check(domain="HUMAN", scope="EPISODE")], modality_availability={"imu": "AVAILABLE", "emg": "PARTIAL", "video": "RAW_UNAVAILABLE"})
        restored = self.assert_roundtrip(report)
        self.assertEqual(restored.record.embodiment, "HUMAN")
        self.assertEqual(restored.acquisition_availability, "AVAILABLE")
        self.assertEqual(restored.modality_availability["video"], "RAW_UNAVAILABLE")
        self.assertEqual(restored.summary["status_counts"]["PASS"], 1)

    def test_robot_report_serialization(self):
        clocks = {"robot_state_ns": {"unit": "ns"}, "camera_capture_ns": {"unit": "ns"}}
        checks = [
            make_check(domain="ROBOT", scope="EPISODE", clock_domain="robot_state_ns", evidence_refs=[{"clock_domain": "robot_state_ns", "sample_index": 10}]),
            make_check(check_id="camera_sequence", domain="ROBOT", scope="SEGMENT", clock_domain="camera_capture_ns", subject={"episode_id": "episode-7", "segment_id": "seg-1"}, evidence_refs=[{"clock_domain": "camera_capture_ns", "segment_id": "seg-1"}]),
        ]
        report = make_report("ROBOT", checks=checks, clocks=clocks, modality_availability={"joint_state": "AVAILABLE", "action": "AVAILABLE", "camera": "PARTIAL"})
        restored = self.assert_roundtrip(report)
        self.assertEqual(restored.checks[0].clock_domain, "robot_state_ns")
        self.assertEqual(restored.checks[1].clock_domain, "camera_capture_ns")
        self.assertEqual(restored.checks[0].evidence_refs[0]["clock_domain"], "robot_state_ns")
        self.assertEqual(restored.checks[1].evidence_refs[0]["clock_domain"], "camera_capture_ns")

    def test_transformed_report_serialization(self):
        report = make_report("TRANSFORMED", checks=[make_check(domain="TRANSFORMATION", scope="EPISODE")], modality_availability={"retargeted_traj": "AVAILABLE"})
        restored = self.assert_roundtrip(report)
        self.assertEqual(restored.record.embodiment, "TRANSFORMED")
        self.assertEqual(restored.checks[0].domain, "TRANSFORMATION")

    def test_missing_modalities_do_not_fail_unrelated_checks(self):
        report = make_report("HUMAN", checks=[make_check(domain="COMMON", scope="DATASET"), make_check(check_id="video_decode", domain="COMMON", scope="EPISODE", status="NOT_ASSESSED", not_assessed_reason="video modality not supplied")], modality_availability={"imu": "AVAILABLE", "video": "RAW_UNAVAILABLE"})
        restored = self.assert_roundtrip(report)
        statuses = {c.check_id: c.status for c in restored.checks}
        self.assertEqual(statuses["numeric_finite"], "PASS")
        self.assertEqual(statuses["video_decode"], "NOT_ASSESSED")
        self.assertEqual(restored.modality_availability["video"], "RAW_UNAVAILABLE")

    def test_raw_unavailable_distinct_from_quality_rejected(self):
        self.assertNotIn("QUALITY_REJECTED", qs.ACQUISITION_STATES)
        self.assertNotIn("QUALITY_REJECTED", qs.CHECK_STATUSES)
        report = make_report("ROBOT", availability="RAW_UNAVAILABLE", modality_availability={"hdf5": "RAW_UNAVAILABLE", "rlds": "AVAILABLE"})
        restored = self.assert_roundtrip(report)
        self.assertEqual(restored.acquisition_availability, "RAW_UNAVAILABLE")
        self.assertEqual(restored.modality_availability["hdf5"], "RAW_UNAVAILABLE")
        with self.assertRaises(qs.SchemaValidationError):
            make_check(status="QUALITY_REJECTED")

    def test_not_assessed_does_not_become_pass(self):
        chk = make_check(status="NOT_ASSESSED", not_assessed_reason="required modality absent")
        report = make_report("HUMAN", checks=[chk])
        restored = self.assert_roundtrip(report)
        self.assertEqual(restored.checks[0].status, "NOT_ASSESSED")
        self.assertEqual(restored.checks[0].not_assessed_reason, "required modality absent")
        self.assertEqual(restored.summary["status_counts"].get("PASS", 0), 0)
        with self.assertRaises(qs.SchemaValidationError):
            make_check(status="NOT_ASSESSED")

    def test_clock_domains_remain_distinct(self):
        clocks = {"robot_state_ns": {"unit": "ns", "source": "controller"}, "camera_capture_ns": {"unit": "ns", "source": "camera"}}
        report = make_report("ROBOT", checks=[make_check(domain="ROBOT", clock_domain="robot_state_ns"), make_check(check_id="camera", domain="ROBOT", clock_domain="camera_capture_ns")], clocks=clocks)
        restored = self.assert_roundtrip(report)
        self.assertEqual(restored.checks[0].clock_domain, "robot_state_ns")
        self.assertEqual(restored.checks[1].clock_domain, "camera_capture_ns")
        with self.assertRaises(qs.SchemaValidationError):
            make_report("ROBOT", checks=[make_check(domain="ROBOT", clock_domain="undeclared_clock")], clocks=clocks)
        with self.assertRaises(qs.SchemaValidationError):
            make_report("ROBOT", checks=[make_check(domain="ROBOT", evidence_refs=[{"clock_domain": "undeclared_clock"}])], clocks=clocks)

    def test_sample_and_episode_evidence_references_survive(self):
        chk = make_check(scope="SAMPLE", subject={"episode_id": "episode-7", "segment_id": "seg-1", "window_id": "win-1", "sample_id": "s-42"}, evidence_refs=[{"dataset_path": "droid.h5", "episode_id": "episode-7", "sample_index": 42, "content_hash": "sha256:x", "clock_domain": "robot_state_ns"}])
        report = make_report("ROBOT", checks=[chk], clocks={"robot_state_ns": {"unit": "ns"}})
        restored = self.assert_roundtrip(report)
        c = restored.checks[0]
        self.assertEqual(c.subject["sample_id"], "s-42")
        self.assertEqual(c.evidence_refs[0]["episode_id"], "episode-7")
        self.assertEqual(c.evidence_refs[0]["sample_index"], 42)
        self.assertEqual(c.evidence_refs[0]["content_hash"], "sha256:x")
        self.assertEqual(restored.segment_findings(), [])
        self.assertEqual([x.check_id for x in restored.sample_findings()], ["numeric_finite"])

    def test_invalid_status_and_malformed_report_rejection(self):
        with self.assertRaises(qs.SchemaValidationError):
            make_check(status="UNKNOWN")
        with self.assertRaises(qs.SchemaValidationError):
            qs.CheckResult.from_dict(make_check().to_dict() | {"extra": 1})
        with self.assertRaises(qs.SchemaValidationError):
            make_report("HUMAN", availability="QUALITY_REJECTED")
        bad = make_report("HUMAN").to_dict(); bad["quality_score"] = 0.9
        with self.assertRaises(qs.SchemaValidationError):
            qs.AssessmentReport.from_dict(bad)
        with self.assertRaises(qs.SchemaValidationError):
            qs.DatasetRecord(record_id="x", dataset_id="y", embodiment="OTHER", source_type="z", sample_count=1)
        with self.assertRaises(qs.SchemaValidationError):
            qs.CheckResult.from_json('{"check_id":"x"}')

    def test_json_module_serialization(self):
        report = make_report("TRANSFORMED", checks=[make_check(domain="TRANSFORMATION")])
        data = json.loads(report.to_json())
        self.assertEqual(data["record"]["embodiment"], "TRANSFORMED")
        self.assertEqual(data["checks"][0]["status"], "PASS")
        self.assertNotIn("quality_score", data)
        self.assertNotIn("tier", data)

    def test_summary_forged_stale_and_direct_mutation_safe(self):
        report = make_report("HUMAN")
        correct = report.compute_summary()
        self.assertEqual(make_report("HUMAN", summary=correct).summary, correct)
        with self.assertRaises(qs.SchemaValidationError):
            make_report("HUMAN", summary={"status_counts": {"PASS": 99}, "domain_counts": {}, "scope_counts": {}})
        data = report.to_dict()
        data["summary"]["status_counts"]["PASS"] = 42
        with self.assertRaises(qs.SchemaValidationError):
            qs.AssessmentReport.from_dict(data)
        data = report.to_dict()
        data["summary"] = report.compute_summary()
        self.assertEqual(qs.AssessmentReport.from_dict(data).summary, report.compute_summary())
        with self.assertRaises(AttributeError):
            report.checks.append(make_check())
        leaked = report.summary
        leaked["status_counts"]["PASS"] = 999
        self.assertEqual(report.summary["status_counts"]["PASS"], 1)
        self.assertEqual(report.to_dict()["summary"], report.compute_summary())
        with self.assertRaises(dataclasses.FrozenInstanceError):
            report.summary = {}

    def test_reference_entries_must_be_dicts(self):
        with self.assertRaises(qs.SchemaValidationError):
            make_check(evidence_refs=["bad"])
        base = {"source_report_id": "s", "target_robot_id": "r", "transformation_algorithm": "a", "transformation_version": "1"}
        for field_name in ("source_refs", "calibration_refs", "exclusions"):
            with self.assertRaises(qs.SchemaValidationError):
                qs.TransformationRecord(**dict(base, **{field_name: [1]}))

    def test_serialization_rejects_nested_mutation_and_nonfinite(self):
        chk = make_check()
        chk.subject["bad"] = float("nan")
        with self.assertRaises(qs.SchemaValidationError):
            chk.to_json()
        chk2 = make_check(scope="SAMPLE", subject={"episode_id": "episode-7", "sample_id": "s-1"}, evidence_refs=[{"episode_id": "episode-7", "sample_index": 1}])
        chk2.evidence_refs[0]["bad"] = float("inf")
        with self.assertRaises(qs.SchemaValidationError):
            chk2.to_json()

    def test_observed_value_tuple_nested_roundtrip(self):
        chk = qs.CheckResult(
            check_id="nested", check_version="1.0.0", domain="COMMON", scope="EPISODE",
            status="PASS", reason_code="OK", explanation="nested", threshold_source="NOT_APPLICABLE",
            observed_value=(1, (2, 3), {"x": (4, 5)}), evidence_refs=(), limitations=(),
            subject={"episode_id": "episode-7"},
        )
        restored = qs.CheckResult.from_json(chk.to_json())
        self.assertEqual(restored.observed_value, [1, [2, 3], {"x": [4, 5]}])
        self.assertEqual(restored.to_dict(), chk.to_dict())

    def test_embodiment_domain_compatibility(self):
        with self.assertRaises(qs.SchemaValidationError):
            make_report("HUMAN", checks=[make_check(domain="ROBOT")])
        with self.assertRaises(qs.SchemaValidationError):
            make_report("ROBOT", checks=[make_check(domain="HUMAN")])
        with self.assertRaises(qs.SchemaValidationError):
            make_report("TRANSFORMED", checks=[make_check(domain="HUMAN")])
        make_report("TRANSFORMED", checks=[make_check(domain="ROBOT")])

    def test_finding_scope_identifiers_required(self):
        with self.assertRaises(qs.SchemaValidationError):
            make_check(scope="SEGMENT", subject={"episode_id": "episode-7"})
        with self.assertRaises(qs.SchemaValidationError):
            make_check(scope="WINDOW", subject={"episode_id": "episode-7", "segment_id": "seg-1"})
        with self.assertRaises(qs.SchemaValidationError):
            make_check(scope="SAMPLE", subject={"episode_id": "episode-7"})
        make_check(scope="WINDOW", subject={"episode_id": "episode-7", "segment_id": "seg-1", "window_id": "win-1"})
        make_check(scope="SAMPLE", subject={"episode_id": "episode-7", "sample_index": 3})
        with self.assertRaises(qs.SchemaValidationError):
            make_check(scope="SAMPLE", subject={"episode_id": "episode-7"}, evidence_refs=[{"sample_index": 3}])
        with self.assertRaises(qs.SchemaValidationError):
            make_check(scope="WINDOW", subject={"episode_id": "episode-7", "segment_id": "seg-1", "window_id": "win-1"}, evidence_refs=[{"episode_id": "episode-8"}])

    def test_invalid_identifier_values_rejected(self):
        for bad_subject in (
            {"episode_id": "", "segment_id": "seg-1", "window_id": "win-1"},
            {"episode_id": "episode-7", "segment_id": "seg-1", "window_id": "win-1", "sample_index": True},
            {"episode_id": "episode-7", "segment_id": "seg-1", "window_id": "win-1", "sample_index": -1},
            {"episode_id": 1.5, "segment_id": "seg-1", "window_id": "win-1"},
        ):
            with self.assertRaises(qs.SchemaValidationError):
                make_check(scope="WINDOW", subject=bad_subject)
        chk = make_check(scope="SAMPLE", subject={"episode_id": "episode-7", "sample_id": "s-1"})
        chk.subject["sample_id"] = ""
        with self.assertRaises(qs.SchemaValidationError):
            chk.to_json()

    def test_nested_mutation_regressions_rejected(self):
        chk = make_check(scope="SAMPLE", subject={"episode_id": "episode-7", "sample_id": "s-1"}, evidence_refs=[{"episode_id": "episode-7", "sample_index": 1}])
        chk.subject.pop("sample_id")
        with self.assertRaises(qs.SchemaValidationError):
            chk.to_json()

        report = make_report("HUMAN", checks=[make_check(domain="HUMAN")])
        report.record.embodiment = "ROBOT"
        with self.assertRaises(qs.SchemaValidationError):
            report.to_json()

        clocks = {"robot_state_ns": {"unit": "ns"}}
        clock_report = make_report("ROBOT", checks=[make_check(domain="ROBOT", clock_domain="robot_state_ns")], clocks=clocks)
        clock_report.record.clock_domains.pop("robot_state_ns")
        with self.assertRaises(qs.SchemaValidationError):
            clock_report.to_json()

        mod_report = make_report("HUMAN")
        mod_report.modality_availability["imu"] = "BAD_STATE"
        with self.assertRaises(qs.SchemaValidationError):
            mod_report.to_json()

    def test_parent_child_episode_identity_consistency(self):
        consistent = make_report("HUMAN", checks=[make_check(domain="HUMAN", scope="EPISODE", subject={"episode_id": "episode-7"}, evidence_refs=[{"episode_id": "episode-7", "sample_index": 1}])])
        self.assertEqual(self.assert_roundtrip(consistent).checks[0].subject["episode_id"], "episode-7")
        with self.assertRaises(qs.SchemaValidationError):
            make_report("HUMAN", checks=[make_check(domain="HUMAN", subject={"episode_id": "episode-8"})])
        with self.assertRaises(qs.SchemaValidationError):
            make_report("HUMAN", checks=[make_check(domain="HUMAN", subject={"episode_id": "episode-7"}, evidence_refs=[{"episode_id": "episode-8"}])])
        dataset_level = make_report("HUMAN", record_episode_id=None, checks=[make_check(domain="HUMAN", scope="SAMPLE", subject={"episode_id": "episode-9", "sample_id": "s-1"}, evidence_refs=[{"episode_id": "episode-9", "sample_index": 1}])])
        self.assertEqual(self.assert_roundtrip(dataset_level).checks[0].subject["episode_id"], "episode-9")
        with self.assertRaises(qs.SchemaValidationError):
            make_record("HUMAN", episode_id="   ")
        with self.assertRaises(qs.SchemaValidationError):
            make_check(scope="WINDOW", subject={"episode_id": "   ", "segment_id": "seg-1", "window_id": "win-1"})
        with self.assertRaises(qs.SchemaValidationError):
            make_check(scope="EPISODE", subject={"episode_id": "episode-7"}, evidence_refs=[{"episode_id": "   "}])


if __name__ == "__main__":
    unittest.main()
