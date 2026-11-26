"""Thin read-only adapter from S2S PhysicsEngine.certify() to cross-embodiment schema.

Milestone 2B research integration only. This module never calls PhysicsEngine,
never recomputes physics, and never modifies the input certification result.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

try:
    from . import quality_schema as qs
except Exception:  # pragma: no cover - used when imported as a top-level module
    import quality_schema as qs

EXPECTED_HUMAN_LAW_KEYS = frozenset({
    "inter_window_continuity",
    "newton_second_law",
    "resonance_frequency",
    "rigid_body_kinematics",
    "jerk_bounds",
    "imu_internal_consistency",
    "cross_axis_cohesion",
    "pointwise_jerk",
    "spectral_flatness",
    "temporal_autocorrelation",
    "sensor_freeze",
    "powerline_interference",
    "intra_window_splice",
    "innovation_kurtosis",
    "ballistocardiography",
    "joule_heating",
})

VALID_LAW_STATUSES = frozenset({"PASS", "FAIL", "NOT_ASSESSED"})
VALID_HUMAN_TIERS = frozenset({"GOLD", "SILVER", "BRONZE", "REJECTED", "UNVERIFIED"})
VALID_ASSESSMENT_STATUSES = frozenset({"ASSESSED", "PARTIAL", "UNVERIFIED"})
VALID_TIER_SCOPES = frozenset({"ASSESSED_CHECKS_ONLY", "FULL_CONFIGURED_CHECKS"})
VALID_ENGINE_STATUS = frozenset({"PASS", "FAIL"})
HUMAN_SOURCE_TYPE = "HIL_BIOLOGICAL"

REQUIRED_RESULT_KEYS = (
    "status", "tier", "tier_scope", "assessment_status", "sensor_type",
    "physical_law_score", "score_basis", "scoring_version",
    "laws_checked", "n_laws_checked", "laws_assessed", "n_laws_assessed",
    "laws_not_assessed", "n_laws_not_assessed", "law_statuses", "laws_passed",
    "laws_failed", "body_segment", "source_type", "law_details", "flags",
    "physical_constants", "device_id", "session_id", "tool", "numpy_enabled",
    "issued_at_ns",
)


def _fail(msg: str) -> None:
    raise qs.SchemaValidationError(msg)


def _require_str_list(value: Any, name: str) -> List[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) and v for v in value):
        _fail(f"{name} must be a list of non-empty strings")
    return list(value)


def _clean_explanation(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


def _explicit_not_assessed_reason(detail: Dict[str, Any]) -> Optional[str]:
    for key in ("skip", "reason"):
        explanation = _clean_explanation(detail.get(key))
        if explanation is not None:
            return explanation
    required = detail.get("required")
    if isinstance(required, list) and required:
        return "missing required inputs: " + ", ".join(str(x) for x in required)
    explanation = _clean_explanation(required)
    if explanation is not None:
        return explanation
    return _clean_explanation(detail.get("info"))


def _validate_certification_result(result: Dict[str, Any]) -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, Any]]]:
    if not isinstance(result, dict):
        _fail("certification result must be a dict")
    missing = [k for k in REQUIRED_RESULT_KEYS if k not in result]
    if missing:
        _fail(f"certification result missing required fields: {missing}")

    if result["source_type"] != HUMAN_SOURCE_TYPE:
        _fail(f"human adapter supports only source_type={HUMAN_SOURCE_TYPE!r}; got {result['source_type']!r}")
    if result["sensor_type"] != "PHYSICS":
        _fail(f"sensor_type must be 'PHYSICS'; got {result['sensor_type']!r}")
    if result["status"] not in VALID_ENGINE_STATUS:
        _fail(f"status must be PASS or FAIL; got {result['status']!r}")
    if result["tier"] not in VALID_HUMAN_TIERS:
        _fail(f"tier must be one of {sorted(VALID_HUMAN_TIERS)}; got {result['tier']!r}")
    if result["assessment_status"] not in VALID_ASSESSMENT_STATUSES:
        _fail(f"assessment_status must be one of {sorted(VALID_ASSESSMENT_STATUSES)}")
    if result["tier_scope"] not in VALID_TIER_SCOPES:
        _fail(f"tier_scope must be one of {sorted(VALID_TIER_SCOPES)}")
    score = result["physical_law_score"]
    if isinstance(score, bool) or not isinstance(score, int) or not (0 <= score <= 100):
        _fail("physical_law_score must be an integer in [0, 100]")
    if not isinstance(result["law_details"], dict):
        _fail("law_details must be an object")
    if not isinstance(result["law_statuses"], dict):
        _fail("law_statuses must be an object")
    if not isinstance(result["flags"], list) or not isinstance(result["physical_constants"], dict):
        _fail("flags must be a list and physical_constants must be an object")
    if not isinstance(result["tool"], str) or not result["tool"]:
        _fail("tool must be a non-empty string")

    laws_checked = _require_str_list(result["laws_checked"], "laws_checked")
    if len(laws_checked) != len(set(laws_checked)):
        _fail("laws_checked contains duplicates")
    if set(laws_checked) != EXPECTED_HUMAN_LAW_KEYS:
        _fail("laws_checked must contain exactly the 16 supported human law keys")
    if result["n_laws_checked"] != len(laws_checked):
        _fail("n_laws_checked does not match laws_checked")

    law_statuses = result["law_statuses"]
    law_details = result["law_details"]
    if set(law_statuses) != set(laws_checked):
        _fail("law_statuses keys must match laws_checked")
    if set(law_details) != set(laws_checked):
        _fail("law_details keys must match laws_checked")

    passed, failed, not_assessed = [], [], []
    for law in laws_checked:
        status = law_statuses[law]
        if status not in VALID_LAW_STATUSES:
            _fail(f"law_statuses[{law!r}] must be PASS, FAIL, or NOT_ASSESSED; got {status!r}")
        detail = law_details[law]
        if not isinstance(detail, dict):
            _fail(f"law_details[{law!r}] must be an object")
        if detail.get("status") != status:
            _fail(f"law_details[{law!r}].status does not match law_statuses[{law!r}]")
        if status == "PASS":
            passed.append(law)
        elif status == "FAIL":
            failed.append(law)
        else:
            not_assessed.append(law)
            if _explicit_not_assessed_reason(detail) is None:
                _fail(f"law {law!r} is NOT_ASSESSED without an explicit reason")

    assessed = [k for k in laws_checked if law_statuses[k] != "NOT_ASSESSED"]
    if set(result["laws_assessed"]) != set(assessed) or result["n_laws_assessed"] != len(assessed):
        _fail("laws_assessed coverage does not match law_statuses")
    if set(result["laws_passed"]) != set(passed) or len(result["laws_passed"]) != len(passed):
        _fail("laws_passed does not match PASS statuses")
    if set(result["laws_failed"]) != set(failed) or len(result["laws_failed"]) != len(failed):
        _fail("laws_failed does not match FAIL statuses")
    if set(result["laws_not_assessed"]) != set(not_assessed) or len(result["laws_not_assessed"]) != len(not_assessed):
        _fail("laws_not_assessed does not match NOT_ASSESSED statuses")
    if result["n_laws_not_assessed"] != len(not_assessed):
        _fail("n_laws_not_assessed does not match NOT_ASSESSED statuses")

    return laws_checked, law_statuses, law_details


def _not_assessed_reason(law: str, detail: Dict[str, Any]) -> Optional[str]:
    reason = _explicit_not_assessed_reason(detail)
    if reason is None:
        _fail(f"law {law!r} is NOT_ASSESSED without an explicit reason")
    return reason


def _subject_for(result: Dict[str, Any], record: qs.DatasetRecord, caller_subject: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    subject: Dict[str, Any] = {}
    for key in ("body_segment", "source_type", "device_id", "session_id"):
        value = result.get(key)
        if value is not None:
            subject[key] = value
    if record.episode_id is not None:
        subject["episode_id"] = record.episode_id
    for key, value in (caller_subject or {}).items():
        if key in subject and subject[key] != value:
            _fail(f"subject.{key} conflicts with certification result")
        subject[key] = value
    if record.episode_id is not None and "episode_id" in subject and subject["episode_id"] != record.episode_id:
        _fail("subject.episode_id conflicts with record.episode_id")
    return subject


def _engine_evidence_ref(result: Dict[str, Any], law: str, status: str) -> Dict[str, Any]:
    return {
        "engine": "s2s_physics_v1_3",
        "tool": result["tool"],
        "law": law,
        "native_status": status,
        "device_id": result.get("device_id"),
        "session_id": result.get("session_id"),
        "body_segment": result.get("body_segment"),
        "source_type": result.get("source_type"),
        "issued_at_ns": result.get("issued_at_ns"),
    }


def human_certify_result_to_assessment(
    result: Dict[str, Any],
    *,
    record: qs.DatasetRecord,
    report_id: str,
    profile_name: str = "s2s_human_physics",
    profile_version: str = "1.9",
    adapter_name: str = "human_certification_adapter",
    adapter_version: str = "0.1.0",
    acquisition_availability: str,
    modality_availability: Dict[str, str],
    scope: str = "EPISODE",
    subject: Optional[Dict[str, Any]] = None,
    evidence_refs: Optional[List[Dict[str, Any]]] = None,
) -> qs.AssessmentReport:
    """Convert one PhysicsEngine.certify() result into an AssessmentReport.

    The adapter is read-only: it does not call PhysicsEngine, rerun laws, or
    mutate ``result``. Caller-supplied ``record``, ``acquisition_availability``,
    ``modality_availability``, ``scope``, ``subject``, and ``evidence_refs``
    provide context that ``certify()`` does not emit.
    """
    laws_checked, law_statuses, law_details = _validate_certification_result(result)

    if not isinstance(record, qs.DatasetRecord):
        _fail("record must be a DatasetRecord")
    if record.embodiment != "HUMAN":
        _fail("human adapter requires record.embodiment='HUMAN'")
    if record.source_type != result["source_type"]:
        _fail("record.source_type does not match certification result source_type")
    if record.session_id is not None and result.get("session_id") is not None and record.session_id != result["session_id"]:
        _fail("record.session_id conflicts with certification result session_id")

    subject_map = _subject_for(result, record, subject)
    extra_refs = list(evidence_refs or [])
    checks = []
    for law in laws_checked:
        status = law_statuses[law]
        detail = law_details[law]
        refs = [dict(r) for r in extra_refs]
        refs.append(_engine_evidence_ref(result, law, status))
        limitations = []
        not_assessed_reason = None
        if status == "NOT_ASSESSED":
            not_assessed_reason = _not_assessed_reason(law, detail)
            limitations.append(not_assessed_reason)
        required = detail.get("required")
        if isinstance(required, list):
            limitations.extend(str(x) for x in required)
        info_reason = _clean_explanation(detail.get("info")) if status == "NOT_ASSESSED" else None
        reason_code = detail.get("result") or detail.get("violation") or detail.get("skip") or detail.get("reason") or info_reason or status
        if not isinstance(reason_code, str) or not reason_code.strip():
            reason_code = status
        explanation = (
            detail.get("interpretation")
            or detail.get("physical_meaning")
            or detail.get("note")
            or (not_assessed_reason if status == "NOT_ASSESSED" else f"Human physics law {law}")
        )
        checks.append(qs.CheckResult(
            check_id=law,
            check_version=result["tool"],
            domain="HUMAN",
            scope=scope,
            status=status,
            reason_code=reason_code,
            explanation=explanation,
            observed_value=detail,
            threshold_source="NOT_APPLICABLE",
            evidence_refs=tuple(refs),
            limitations=tuple(limitations),
            not_assessed_reason=not_assessed_reason,
            subject=subject_map,
        ))

    metadata = {
        "native_human_certification": copy.deepcopy(result),
        "human_adapter": {
            "name": adapter_name,
            "version": adapter_version,
            "schema_version": qs.SCHEMA_VERSION,
        },
    }
    return qs.AssessmentReport(
        report_id=report_id,
        record=record,
        profile_name=profile_name,
        profile_version=profile_version,
        adapter_name=adapter_name,
        adapter_version=adapter_version,
        acquisition_availability=acquisition_availability,
        checks=tuple(checks),
        modality_availability=modality_availability,
        metadata=metadata,
    )


__all__ = [
    "EXPECTED_HUMAN_LAW_KEYS",
    "human_certify_result_to_assessment",
]
