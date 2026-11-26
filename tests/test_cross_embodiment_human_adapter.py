import copy
import inspect
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "cross_embodiment"))

import quality_schema as qs
import human_adapter as ha
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "experiments" / "cross_embodiment" / "fixtures"

LAW_KEYS = [
    "newton_second_law", "inter_window_continuity", "resonance_frequency",
    "rigid_body_kinematics", "jerk_bounds", "imu_internal_consistency",
    "cross_axis_cohesion", "pointwise_jerk", "spectral_flatness",
    "temporal_autocorrelation", "sensor_freeze", "powerline_interference",
    "intra_window_splice", "innovation_kurtosis", "ballistocardiography",
    "joule_heating",
]


def make_record(**overrides):
    kwargs = dict(
        record_id="human-record-1",
        dataset_id="human-dataset",
        episode_id="episode-7",
        session_id="session-1",
        embodiment="HUMAN",
        source_type="HIL_BIOLOGICAL",
        modalities_present=["imu", "emg"],
        modalities_expected=["imu", "emg", "ppg", "thermal"],
        sample_count=256,
        original_index_range=[0, 255],
        provenance={"source_file": "human_session.bin", "adapter": "test"},
        clock_domains={"imu_timestamps_ns": {"unit": "ns"}},
    )
    kwargs.update(overrides)
    return qs.DatasetRecord(**kwargs)


def make_context(**overrides):
    kwargs = dict(
        record=make_record(),
        report_id="human-report-1",
        acquisition_availability="AVAILABLE",
        modality_availability={"imu": "AVAILABLE", "emg": "PARTIAL", "ppg": "RAW_UNAVAILABLE", "thermal": "RAW_UNAVAILABLE"},
    )
    kwargs.update(overrides)
    return kwargs


def make_cert_result():
    statuses = {law: "NOT_ASSESSED" for law in LAW_KEYS}
    statuses["newton_second_law"] = "PASS"
    statuses["resonance_frequency"] = "FAIL"
    details = {}
    for law, status in statuses.items():
        detail = {"law": law, "status": status, "note": f"detail for {law}"}
        if status == "PASS":
            detail["result"] = "OK"
            detail["confidence"] = 80
        elif status == "FAIL":
            detail["violation"] = "PHYSICS_VIOLATION"
            detail["confidence"] = 90
        else:
            detail["skip"] = "missing required inputs"
            detail["required"] = ["imu_raw", "emg_raw"]
        details[law] = detail
    assessed = [k for k in LAW_KEYS if statuses[k] != "NOT_ASSESSED"]
    passed = [k for k in LAW_KEYS if statuses[k] == "PASS"]
    failed = [k for k in LAW_KEYS if statuses[k] == "FAIL"]
    not_assessed = [k for k in LAW_KEYS if statuses[k] == "NOT_ASSESSED"]
    return {
        "status": "FAIL",
        "tier": "REJECTED",
        "tier_scope": "ASSESSED_CHECKS_ONLY",
        "assessment_status": "PARTIAL",
        "sensor_type": "PHYSICS",
        "physical_law_score": 40,
        "score_basis": "PASS-law confidence summed over assessed laws; FAIL contributes 0",
        "scoring_version": 2,
        "laws_checked": list(LAW_KEYS),
        "n_laws_checked": len(LAW_KEYS),
        "laws_assessed": assessed,
        "n_laws_assessed": len(assessed),
        "laws_not_assessed": not_assessed,
        "n_laws_not_assessed": len(not_assessed),
        "law_statuses": dict(statuses),
        "laws_passed": passed,
        "laws_failed": failed,
        "body_segment": "forearm",
        "source_type": "HIL_BIOLOGICAL",
        "law_details": details,
        "flags": ["PHYSICS_VIOLATION:resonance_frequency"],
        "physical_constants": {"jerk_max_ms3": 500.0},
        "device_id": "dev-1",
        "session_id": "session-1",
        "tool": "s2s_physics_v1_9",
        "numpy_enabled": True,
        "issued_at_ns": 1700000000000000000,
    }


class HumanAdapterTests(unittest.TestCase):
    def test_real_engine_all_not_assessed_maps_all_16_laws(self):
        result = PhysicsEngine().certify()
        before = copy.deepcopy(result)
        ctx = make_context(
            record=make_record(
                modalities_present=[],
                modalities_expected=["imu", "emg", "ppg", "thermal"],
                sample_count=0,
                original_index_range=None,
            ),
            acquisition_availability="RAW_UNAVAILABLE",
            modality_availability={"imu": "RAW_UNAVAILABLE", "emg": "RAW_UNAVAILABLE", "ppg": "RAW_UNAVAILABLE", "thermal": "RAW_UNAVAILABLE"},
        )
        report = ha.human_certify_result_to_assessment(result, **ctx)
        self.assertEqual(result, before)
        self.assertEqual(len(report.checks), 16)
        self.assertEqual({c.check_id for c in report.checks}, set(LAW_KEYS))
        self.assertTrue(all(c.domain == "HUMAN" for c in report.checks))
        self.assertTrue(all(c.status == "NOT_ASSESSED" for c in report.checks))
        self.assertTrue(all(c.not_assessed_reason for c in report.checks))
        self.assertEqual(report.metadata["native_human_certification"], result)
        self.assertEqual(report.metadata["native_human_certification"]["tier"], "UNVERIFIED")
        self.assertEqual(report.metadata["native_human_certification"]["physical_law_score"], 0)

    def test_real_first_window_info_reason(self):
        fixture = FIXTURE_DIR / "real_first_window_certify_input.json"
        self.assertTrue(fixture.exists(), f"missing fixture {fixture}")
        imu_raw = json.loads(fixture.read_text())
        result = PhysicsEngine().certify(
            imu_raw=imu_raw,
            segment="forearm",
            device_id="audit-dev",
            session_id="audit-session",
        )
        self.assertEqual(result["law_statuses"]["inter_window_continuity"], "NOT_ASSESSED")
        self.assertEqual(result["law_details"]["inter_window_continuity"]["info"], "FIRST_WINDOW_INITIALIZED")
        before = copy.deepcopy(result)
        n = len(imu_raw["accel"])
        ctx = make_context(
            record=make_record(
                session_id="audit-session",
                modalities_present=["imu"],
                modalities_expected=["imu", "emg", "ppg", "thermal"],
                sample_count=n,
                original_index_range=[0, n - 1],
            ),
            acquisition_availability="PARTIAL",
            modality_availability={"imu": "AVAILABLE", "emg": "RAW_UNAVAILABLE", "ppg": "RAW_UNAVAILABLE", "thermal": "RAW_UNAVAILABLE"},
        )
        report = ha.human_certify_result_to_assessment(result, **ctx)
        self.assertEqual(result, before)
        self.assertEqual(len(report.checks), 16)
        by_id = {c.check_id: c for c in report.checks}
        self.assertEqual(set(by_id), set(LAW_KEYS))
        for law, status in result["law_statuses"].items():
            self.assertEqual(by_id[law].status, status)
        check = by_id["inter_window_continuity"]
        self.assertEqual(check.status, "NOT_ASSESSED")
        self.assertEqual(check.not_assessed_reason, "FIRST_WINDOW_INITIALIZED")
        self.assertEqual(check.reason_code, "FIRST_WINDOW_INITIALIZED")
        self.assertEqual(report.metadata["native_human_certification"], result)
        restored = qs.AssessmentReport.from_json(report.to_json())
        self.assertEqual(restored.to_dict(), report.to_dict())
        self.assertEqual(restored.metadata["native_human_certification"], result)

    def test_mixed_status_parity_and_native_preservation(self):
        result = make_cert_result()
        before = copy.deepcopy(result)
        report = ha.human_certify_result_to_assessment(result, **make_context())
        self.assertEqual(result, before)
        self.assertEqual(len(report.checks), 16)
        by_id = {c.check_id: c for c in report.checks}
        for law, status in result["law_statuses"].items():
            self.assertEqual(by_id[law].status, status)
        self.assertEqual(by_id["newton_second_law"].status, "PASS")
        self.assertEqual(by_id["resonance_frequency"].status, "FAIL")
        self.assertEqual(report.summary["status_counts"], {"PASS": 1, "FAIL": 1, "NOT_ASSESSED": 14})
        native = report.metadata["native_human_certification"]
        self.assertEqual(native["tier"], "REJECTED")
        self.assertEqual(native["physical_law_score"], 40)
        self.assertEqual(native["laws_passed"], result["laws_passed"])
        self.assertEqual(native["laws_failed"], result["laws_failed"])
        self.assertEqual(native["laws_not_assessed"], result["laws_not_assessed"])
        self.assertEqual(native["law_statuses"], result["law_statuses"])
        self.assertEqual(native["law_details"], result["law_details"])
        self.assertEqual(native["flags"], result["flags"])
        self.assertEqual(native["physical_constants"], result["physical_constants"])
        self.assertEqual(native["tool"], result["tool"])
        self.assertEqual(native["issued_at_ns"], result["issued_at_ns"])
        restored = qs.AssessmentReport.from_json(report.to_json())
        self.assertEqual(restored.to_dict(), report.to_dict())
        self.assertEqual(restored.metadata["native_human_certification"], result)

    def test_context_and_window_locator_preserved(self):
        result = make_cert_result()
        report = ha.human_certify_result_to_assessment(
            result,
            **make_context(
                scope="WINDOW",
                subject={"segment_id": "seg-1", "window_id": "win-1"},
                evidence_refs=[{"dataset_path": "human.bin", "sample_index": 0}],
            ),
        )
        check = report.checks[0]
        self.assertEqual(check.scope, "WINDOW")
        self.assertEqual(check.subject["episode_id"], "episode-7")
        self.assertEqual(check.subject["segment_id"], "seg-1")
        self.assertEqual(check.subject["window_id"], "win-1")
        self.assertEqual(check.subject["body_segment"], "forearm")
        self.assertEqual(check.subject["device_id"], "dev-1")
        self.assertEqual(check.subject["session_id"], "session-1")
        self.assertEqual(report.acquisition_availability, "AVAILABLE")
        self.assertEqual(report.modality_availability["ppg"], "RAW_UNAVAILABLE")
        self.assertIn({"dataset_path": "human.bin", "sample_index": 0}, [dict(r) for r in check.evidence_refs[:-1]])
        self.assertEqual(report.record.clock_domains["imu_timestamps_ns"]["unit"], "ns")

    def test_adapter_is_read_only_and_does_not_invoke_engine(self):
        self.assertFalse(hasattr(ha, "PhysicsEngine"))
        self.assertNotIn("PhysicsEngine(", inspect.getsource(ha))
        result = make_cert_result()
        before = copy.deepcopy(result)
        ha.human_certify_result_to_assessment(result, **make_context())
        self.assertEqual(result, before)

    def test_not_assessed_info_edge_cases(self):
        base = make_cert_result()

        no_reason = copy.deepcopy(base)
        no_reason["law_details"]["inter_window_continuity"] = {
            "law": "Inter_Window_Continuity",
            "status": "NOT_ASSESSED",
            "note": "no explanation fields",
        }
        with self.assertRaises(qs.SchemaValidationError):
            ha.human_certify_result_to_assessment(no_reason, **make_context())

        for bad_info in ("", "   ", 123, None):
            bad = copy.deepcopy(base)
            bad["law_details"]["inter_window_continuity"] = {
                "law": "Inter_Window_Continuity",
                "status": "NOT_ASSESSED",
                "info": bad_info,
            }
            with self.assertRaises(qs.SchemaValidationError, msg=f"info={bad_info!r}"):
                ha.human_certify_result_to_assessment(bad, **make_context())

        pass_fail_info = copy.deepcopy(base)
        pass_fail_info["law_details"]["newton_second_law"]["info"] = "FIRST_WINDOW_INITIALIZED"
        pass_fail_info["law_details"]["resonance_frequency"]["info"] = "FIRST_WINDOW_INITIALIZED"
        report = ha.human_certify_result_to_assessment(pass_fail_info, **make_context())
        by_id = {c.check_id: c for c in report.checks}
        self.assertEqual(by_id["newton_second_law"].status, "PASS")
        self.assertIsNone(by_id["newton_second_law"].not_assessed_reason)
        self.assertEqual(by_id["resonance_frequency"].status, "FAIL")
        self.assertIsNone(by_id["resonance_frequency"].not_assessed_reason)

    def test_negative_cases_rejected(self):
        result = make_cert_result()
        missing = copy.deepcopy(result)
        missing["laws_checked"] = missing["laws_checked"][:-1]
        missing["law_statuses"].pop("joule_heating")
        missing["law_details"].pop("joule_heating")
        missing["n_laws_checked"] = 15
        with self.assertRaises(qs.SchemaValidationError):
            ha.human_certify_result_to_assessment(missing, **make_context())

        unknown = copy.deepcopy(result)
        unknown["laws_checked"] = unknown["laws_checked"] + ["extra_law"]
        unknown["law_statuses"]["extra_law"] = "PASS"
        unknown["law_details"]["extra_law"] = {"status": "PASS", "result": "OK"}
        unknown["n_laws_checked"] = 17
        with self.assertRaises(qs.SchemaValidationError):
            ha.human_certify_result_to_assessment(unknown, **make_context())

        invalid = copy.deepcopy(result)
        invalid["law_statuses"]["newton_second_law"] = "MAYBE"
        with self.assertRaises(qs.SchemaValidationError):
            ha.human_certify_result_to_assessment(invalid, **make_context())

        robot = copy.deepcopy(result)
        robot["source_type"] = "ROBOT_SENSOR_MEASURED"
        with self.assertRaises(qs.SchemaValidationError):
            ha.human_certify_result_to_assessment(robot, **make_context(record=make_record(source_type="ROBOT_SENSOR_MEASURED", embodiment="HUMAN")))

        with self.assertRaises(qs.SchemaValidationError):
            ha.human_certify_result_to_assessment(result, **make_context(record=make_record(embodiment="ROBOT", source_type="HIL_BIOLOGICAL")))
        with self.assertRaises(qs.SchemaValidationError):
            ha.human_certify_result_to_assessment(result, **make_context(record=make_record(session_id="other-session")))
        with self.assertRaises(qs.SchemaValidationError):
            ha.human_certify_result_to_assessment(result, **make_context(acquisition_availability="BAD"))
        with self.assertRaises(qs.SchemaValidationError):
            ha.human_certify_result_to_assessment(result, **make_context(modality_availability={"imu": "BAD"}))


if __name__ == "__main__":
    unittest.main()
