"""Versioned, domain-neutral S2S cross-embodiment quality assessment schema.

Milestone 1 research contract only. Standard library, Python >=3.9.
This module defines evidence/provenance/report structures; it does not implement
human or robot physics, thresholds, retargeting, scoring, or training logic.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = "1.0.0"

EMBODIMENTS = frozenset({"HUMAN", "ROBOT", "TRANSFORMED"})
CHECK_DOMAINS = frozenset({"COMMON", "HUMAN", "ROBOT", "TRANSFORMATION"})
CHECK_STATUSES = frozenset({"PASS", "WARNING", "FAIL", "NOT_ASSESSED"})
ACQUISITION_STATES = frozenset({"AVAILABLE", "PARTIAL", "RAW_UNAVAILABLE"})
SCOPES = frozenset({"DATASET", "EPISODE", "SEGMENT", "WINDOW", "SAMPLE"})
THRESHOLD_SOURCES = frozenset({
    "SPECIFICATION",
    "DEVICE_METADATA",
    "VALIDATED_METHOD",
    "EXPLORATORY_EMPIRICAL",
    "NOT_APPLICABLE",
})
FINDING_SCOPES = frozenset({"SEGMENT", "WINDOW", "SAMPLE"})
FORBIDDEN_REPORT_KEYS = frozenset({
    "quality_score", "overall_score", "score", "tier", "quality_tier", "grade",
})
ALLOWED_CHECK_DOMAINS_BY_EMBODIMENT = {
    "HUMAN": frozenset({"COMMON", "HUMAN"}),
    "ROBOT": frozenset({"COMMON", "ROBOT"}),
    "TRANSFORMED": frozenset({"COMMON", "TRANSFORMATION", "ROBOT"}),
}
LOCATOR_KEYS = frozenset({"episode_id", "segment_id", "window_id", "sample_id", "sample_index"})


class SchemaValidationError(ValueError):
    """Raised when a schema object is malformed or violates the contract."""


def _fail(msg: str) -> None:
    raise SchemaValidationError(msg)


def _req_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be a non-empty string")
    return value


def _opt_str(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _req_str(value, name)


def _str_list(value: Any, name: str, allow_none: bool = False) -> List[str]:
    if value is None and allow_none:
        return []
    if not isinstance(value, (list, tuple)):
        _fail(f"{name} must be a list of strings")
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item:
            _fail(f"{name}[{i}] must be a non-empty string")
        out.append(item)
    return out


def _json_safe(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(f"{path} contains non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str):
                _fail(f"{path} has non-string key {k!r}")
            out[k] = _json_safe(v, f"{path}.{k}")
        return out
    _fail(f"{path} has non-JSON-serializable type {type(value).__name__}")
    return value


def _json_dict(value: Any, name: str, required: bool = False) -> Dict[str, Any]:
    if value is None:
        if required:
            _fail(f"{name} is required")
        return {}
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return _json_safe(value, name)


def _require_ref_dicts(value: Any, name: str) -> List[Dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        _fail(f"{name} must be a list of objects")
    out = []
    for i, item in enumerate(value):
        if not isinstance(item, dict):
            _fail(f"{name}[{i}] must be an object")
        out.append(_json_safe(item, f"{name}[{i}]"))
    return out


def _enum(value: Any, allowed: Iterable[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail(f"{name} must be one of {sorted(allowed)}; got {value!r}")
    return value


def _int_ge(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{name} must be an integer >= {minimum}")
    return value


def _valid_locator_value(value: Any, key: str, path: str) -> None:
    if key == "sample_index":
        _int_ge(value, f"{path}.{key}", 0)
        return
    if isinstance(value, bool):
        _fail(f"{path}.{key} must not be boolean")
    if isinstance(value, int):
        if value < 0:
            _fail(f"{path}.{key} must be >= 0")
        return
    if isinstance(value, str):
        if not value.strip():
            _fail(f"{path}.{key} must be a non-empty string")
        return
    _fail(f"{path}.{key} has unsupported identifier type {type(value).__name__}")


def _validate_scope_identifiers(check: "CheckResult") -> None:
    subject = check.subject if isinstance(check.subject, dict) else {}
    for key in LOCATOR_KEYS:
        if key in subject:
            _valid_locator_value(subject[key], key, "subject")
    for j, ref in enumerate(check.evidence_refs or []):
        if not isinstance(ref, dict):
            continue
        for key in LOCATOR_KEYS.intersection(ref.keys()):
            _valid_locator_value(ref[key], key, f"evidence_refs[{j}]")
            if key in subject and ref[key] != subject[key]:
                _fail(f"evidence_refs[{j}].{key} conflicts with subject.{key}")
    if check.scope in {"DATASET", "EPISODE"}:
        return
    missing = []
    if "episode_id" not in subject:
        missing.append("episode_id")
    if check.scope == "SEGMENT":
        if "segment_id" not in subject:
            missing.append("segment_id")
    elif check.scope == "WINDOW":
        if "segment_id" not in subject:
            missing.append("segment_id")
        if "window_id" not in subject:
            missing.append("window_id")
    elif check.scope == "SAMPLE":
        if "sample_id" not in subject and "sample_index" not in subject:
            missing.append("sample_id or sample_index")
    if missing:
        _fail(f"{check.scope} scope requires complete subject identifiers {missing}")


def _validate_domain_for_embodiment(embodiment: str, domain: str) -> None:
    allowed = ALLOWED_CHECK_DOMAINS_BY_EMBODIMENT.get(embodiment, frozenset())
    if domain not in allowed:
        _fail(f"check domain {domain!r} is not compatible with embodiment {embodiment!r}")


def _unknown_keys(data: Dict[str, Any], allowed: set, cls_name: str) -> None:
    extra = sorted(set(data) - allowed)
    if extra:
        _fail(f"{cls_name} has unknown fields: {extra}")


@dataclass
class DatasetRecord:
    """Identity, embodiment, modality inventory, provenance, and clock domains."""

    record_id: str
    dataset_id: str
    embodiment: str
    source_type: str
    sample_count: int
    schema_version: str = SCHEMA_VERSION
    episode_id: Optional[str] = None
    session_id: Optional[str] = None
    modalities_present: List[str] = field(default_factory=list)
    modalities_expected: Optional[List[str]] = None
    original_index_range: Optional[List[int]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    clock_domains: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.modalities_present = _str_list(self.modalities_present, "modalities_present")
        self.modalities_expected = (
            _str_list(self.modalities_expected, "modalities_expected")
            if self.modalities_expected is not None else None
        )
        if self.original_index_range is not None:
            if not isinstance(self.original_index_range, (list, tuple)) or len(self.original_index_range) != 2:
                _fail("original_index_range must be [start, end]")
            start, end = self.original_index_range
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
                _fail("original_index_range bounds must be integers")
            if start < 0 or end < start:
                _fail("original_index_range must satisfy 0 <= start <= end")
            self.original_index_range = [start, end]
        self.provenance = _json_dict(self.provenance, "provenance")
        self.clock_domains = _json_dict(self.clock_domains, "clock_domains")
        self._validate_semantics()

    def _validate_semantics(self) -> None:
        _req_str(self.record_id, "record_id")
        _req_str(self.dataset_id, "dataset_id")
        _req_str(self.schema_version, "schema_version")
        _opt_str(self.episode_id, "episode_id")
        _opt_str(self.session_id, "session_id")
        if self.episode_id is not None and not self.episode_id.strip():
            _fail("episode_id must not be whitespace-only")
        if self.session_id is not None and not self.session_id.strip():
            _fail("session_id must not be whitespace-only")
        _req_str(self.source_type, "source_type")
        _enum(self.embodiment, EMBODIMENTS, "embodiment")
        _int_ge(self.sample_count, "sample_count", 0)
        _str_list(self.modalities_present, "modalities_present")
        if self.modalities_expected is not None:
            _str_list(self.modalities_expected, "modalities_expected")
        if self.original_index_range is not None:
            if not isinstance(self.original_index_range, (list, tuple)) or len(self.original_index_range) != 2:
                _fail("original_index_range must be [start, end]")
            start, end = self.original_index_range
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
                _fail("original_index_range bounds must be integers")
            if start < 0 or end < start:
                _fail("original_index_range must satisfy 0 <= start <= end")
        _json_dict(self.provenance, "provenance")
        _json_dict(self.clock_domains, "clock_domains")
        if len(set(self.modalities_present)) != len(self.modalities_present):
            _fail("modalities_present contains duplicates")
        if self.modalities_expected is not None and len(set(self.modalities_expected)) != len(self.modalities_expected):
            _fail("modalities_expected contains duplicates")

    def to_dict(self) -> Dict[str, Any]:
        self._validate_semantics()
        return {
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "dataset_id": self.dataset_id,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "embodiment": self.embodiment,
            "source_type": self.source_type,
            "modalities_present": _str_list(self.modalities_present, "modalities_present"),
            "modalities_expected": _str_list(self.modalities_expected, "modalities_expected") if self.modalities_expected is not None else None,
            "sample_count": self.sample_count,
            "original_index_range": list(self.original_index_range) if self.original_index_range is not None else None,
            "provenance": _json_dict(self.provenance, "provenance"),
            "clock_domains": _json_dict(self.clock_domains, "clock_domains"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetRecord":
        if not isinstance(data, dict):
            _fail("DatasetRecord.from_dict requires an object")
        _unknown_keys(data, set(cls.__dataclass_fields__), "DatasetRecord")
        required = ["record_id", "dataset_id", "embodiment", "source_type", "sample_count"]
        missing = [k for k in required if k not in data]
        if missing:
            _fail(f"DatasetRecord missing required fields: {missing}")
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(_json_safe(self.to_dict()), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "DatasetRecord":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class CheckResult:
    """One rule/check outcome with explicit status, provenance, and evidence."""

    check_id: str
    check_version: str
    domain: str
    scope: str
    status: str
    reason_code: str
    explanation: str
    threshold_source: str
    observed_value: Any = None
    unit: Optional[str] = None
    expected_bounds: Optional[Dict[str, Any]] = None
    evidence_refs: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    limitations: Tuple[str, ...] = field(default_factory=tuple)
    not_assessed_reason: Optional[str] = None
    clock_domain: Optional[str] = None
    subject: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "expected_bounds", _json_dict(self.expected_bounds, "expected_bounds") if self.expected_bounds is not None else None)
        object.__setattr__(self, "observed_value", _json_safe(self.observed_value, "observed_value"))
        object.__setattr__(self, "evidence_refs", tuple(_require_ref_dicts(self.evidence_refs, "evidence_refs")))
        object.__setattr__(self, "limitations", tuple(_str_list(self.limitations, "limitations")))
        object.__setattr__(self, "subject", _json_dict(self.subject, "subject"))
        self._validate_semantics()

    def _validate_semantics(self) -> None:
        _req_str(self.check_id, "check_id")
        _req_str(self.check_version, "check_version")
        _enum(self.domain, CHECK_DOMAINS, "domain")
        _enum(self.scope, SCOPES, "scope")
        _enum(self.status, CHECK_STATUSES, "status")
        _req_str(self.reason_code, "reason_code")
        _req_str(self.explanation, "explanation")
        _enum(self.threshold_source, THRESHOLD_SOURCES, "threshold_source")
        _opt_str(self.unit, "unit")
        _opt_str(self.clock_domain, "clock_domain")
        _opt_str(self.not_assessed_reason, "not_assessed_reason")
        if self.status == "NOT_ASSESSED" and not self.not_assessed_reason:
            _fail("NOT_ASSESSED requires not_assessed_reason")
        if self.expected_bounds is not None:
            _json_dict(self.expected_bounds, "expected_bounds")
        _json_safe(self.observed_value, "observed_value")
        _require_ref_dicts(self.evidence_refs, "evidence_refs")
        _str_list(self.limitations, "limitations")
        _json_dict(self.subject, "subject")
        _validate_scope_identifiers(self)

    def to_dict(self) -> Dict[str, Any]:
        self._validate_semantics()
        return {
            "check_id": self.check_id,
            "check_version": self.check_version,
            "domain": self.domain,
            "scope": self.scope,
            "status": self.status,
            "reason_code": self.reason_code,
            "explanation": self.explanation,
            "observed_value": _json_safe(self.observed_value, "observed_value"),
            "unit": self.unit,
            "expected_bounds": _json_dict(self.expected_bounds, "expected_bounds") if self.expected_bounds is not None else None,
            "threshold_source": self.threshold_source,
            "evidence_refs": _require_ref_dicts(self.evidence_refs, "evidence_refs"),
            "limitations": _str_list(self.limitations, "limitations"),
            "not_assessed_reason": self.not_assessed_reason,
            "clock_domain": self.clock_domain,
            "subject": _json_dict(self.subject, "subject"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckResult":
        if not isinstance(data, dict):
            _fail("CheckResult.from_dict requires an object")
        _unknown_keys(data, set(cls.__dataclass_fields__), "CheckResult")
        required = ["check_id", "check_version", "domain", "scope", "status", "reason_code", "explanation", "threshold_source"]
        missing = [k for k in required if k not in data]
        if missing:
            _fail(f"CheckResult missing required fields: {missing}")
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(_json_safe(self.to_dict()), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "CheckResult":
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class AssessmentReport:
    """Domain-neutral assessment report; no unified score or tier."""

    report_id: str
    record: DatasetRecord
    profile_name: str
    profile_version: str
    adapter_name: str
    adapter_version: str
    acquisition_availability: str
    checks: Tuple[CheckResult, ...] = field(default_factory=tuple)
    modality_availability: Dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _req_str(self.report_id, "report_id")
        _req_str(self.profile_name, "profile_name")
        _req_str(self.profile_version, "profile_version")
        _req_str(self.adapter_name, "adapter_name")
        _req_str(self.adapter_version, "adapter_version")
        _req_str(self.schema_version, "schema_version")
        _enum(self.acquisition_availability, ACQUISITION_STATES, "acquisition_availability")
        if not isinstance(self.record, DatasetRecord):
            _fail("record must be a DatasetRecord")
        if not isinstance(self.checks, (list, tuple)):
            _fail("checks must be a list of CheckResult")
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "modality_availability", _json_dict(self.modality_availability, "modality_availability"))
        object.__setattr__(self, "metadata", _json_dict(self.metadata, "metadata"))
        self._validate_semantics()

    @property
    def summary(self) -> Dict[str, Dict[str, int]]:
        return self.compute_summary()

    def compute_summary(self) -> Dict[str, Dict[str, int]]:
        status_counts: Dict[str, int] = {}
        domain_counts: Dict[str, int] = {}
        scope_counts: Dict[str, int] = {}
        for chk in self.checks:
            status_counts[chk.status] = status_counts.get(chk.status, 0) + 1
            domain_counts[chk.domain] = domain_counts.get(chk.domain, 0) + 1
            scope_counts[chk.scope] = scope_counts.get(chk.scope, 0) + 1
        return {"status_counts": status_counts, "domain_counts": domain_counts, "scope_counts": scope_counts}

    def _validate_semantics(self) -> None:
        self.record._validate_semantics()
        modality_availability = _json_dict(self.modality_availability, "modality_availability")
        for mod, state in modality_availability.items():
            _req_str(mod, "modality_availability key")
            _enum(state, ACQUISITION_STATES, f"modality_availability[{mod}]")
        declared_clocks = set((self.record.clock_domains or {}).keys())
        for i, chk in enumerate(self.checks):
            if not isinstance(chk, CheckResult):
                _fail(f"checks[{i}] must be CheckResult")
            chk._validate_semantics()
            parent_episode_id = self.record.episode_id
            if parent_episode_id is not None:
                if isinstance(chk.subject, dict) and "episode_id" in chk.subject and chk.subject["episode_id"] != parent_episode_id:
                    _fail(f"checks[{i}].subject.episode_id conflicts with record.episode_id")
                for j, ref in enumerate(chk.evidence_refs):
                    if isinstance(ref, dict) and "episode_id" in ref and ref["episode_id"] != parent_episode_id:
                        _fail(f"checks[{i}].evidence_refs[{j}].episode_id conflicts with record.episode_id")
            _validate_domain_for_embodiment(self.record.embodiment, chk.domain)
            if chk.clock_domain is not None:
                if not declared_clocks:
                    _fail(f"checks[{i}].clock_domain={chk.clock_domain!r} but record declares no clock_domains")
                if chk.clock_domain not in declared_clocks:
                    _fail(f"checks[{i}].clock_domain={chk.clock_domain!r} not declared in record.clock_domains")
            for j, ref in enumerate(chk.evidence_refs):
                if "clock_domain" in ref:
                    cd = ref["clock_domain"]
                    if not isinstance(cd, str) or not cd:
                        _fail(f"checks[{i}].evidence_refs[{j}].clock_domain must be a non-empty string")
                    if cd not in declared_clocks:
                        _fail(f"checks[{i}].evidence_refs[{j}].clock_domain={cd!r} not declared in record.clock_domains")
        _json_dict(self.metadata, "metadata")

    def findings(self, scope: str) -> List[CheckResult]:
        _enum(scope, SCOPES, "scope")
        return [chk for chk in self.checks if chk.scope == scope]

    def segment_findings(self) -> List[CheckResult]:
        return self.findings("SEGMENT")

    def window_findings(self) -> List[CheckResult]:
        return self.findings("WINDOW")

    def sample_findings(self) -> List[CheckResult]:
        return self.findings("SAMPLE")

    def to_dict(self) -> Dict[str, Any]:
        self._validate_semantics()
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "record": self.record.to_dict(),
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
            "acquisition_availability": self.acquisition_availability,
            "modality_availability": _json_dict(self.modality_availability, "modality_availability"),
            "checks": [chk.to_dict() for chk in self.checks],
            "metadata": _json_dict(self.metadata, "metadata"),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssessmentReport":
        if not isinstance(data, dict):
            _fail("AssessmentReport.from_dict requires an object")
        forbidden = FORBIDDEN_REPORT_KEYS.intersection(data)
        if forbidden:
            _fail(f"AssessmentReport must not contain unified score/tier fields: {sorted(forbidden)}")
        payload = dict(data)
        has_summary = "summary" in payload
        supplied_summary = payload.pop("summary", None)
        _unknown_keys(payload, set(cls.__dataclass_fields__), "AssessmentReport")
        required = ["report_id", "record", "profile_name", "profile_version", "adapter_name", "adapter_version", "acquisition_availability"]
        missing = [k for k in required if k not in payload]
        if missing:
            _fail(f"AssessmentReport missing required fields: {missing}")
        payload["record"] = DatasetRecord.from_dict(payload["record"])
        payload["checks"] = [CheckResult.from_dict(c) for c in payload.get("checks", [])]
        report = cls(**payload)
        if has_summary and _json_dict(supplied_summary, "summary") != report.compute_summary():
            _fail("summary does not match checks; refusing untrusted summary")
        return report

    def to_json(self) -> str:
        return json.dumps(_json_safe(self.to_dict()), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "AssessmentReport":
        return cls.from_dict(json.loads(text))


@dataclass
class TransformationRecord:
    """Auditable human/robot/retargeted trajectory transformation lineage."""

    source_report_id: str
    target_robot_id: str
    transformation_algorithm: str
    transformation_version: str
    schema_version: str = SCHEMA_VERSION
    source_refs: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    target_model_version: Optional[str] = None
    coordinate_frames: Dict[str, Any] = field(default_factory=dict)
    calibration_refs: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    mapping_spec: Dict[str, Any] = field(default_factory=dict)
    time_mapping: Dict[str, Any] = field(default_factory=dict)
    exclusions: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    output_content_hash: Optional[str] = None
    target_report_id: Optional[str] = None

    def __post_init__(self) -> None:
        self.source_refs = tuple(_require_ref_dicts(self.source_refs, "source_refs"))
        self.coordinate_frames = _json_dict(self.coordinate_frames, "coordinate_frames")
        self.calibration_refs = tuple(_require_ref_dicts(self.calibration_refs, "calibration_refs"))
        self.mapping_spec = _json_dict(self.mapping_spec, "mapping_spec")
        self.time_mapping = _json_dict(self.time_mapping, "time_mapping")
        self.exclusions = tuple(_require_ref_dicts(self.exclusions, "exclusions"))
        self._validate_semantics()

    def _validate_semantics(self) -> None:
        _req_str(self.source_report_id, "source_report_id")
        _req_str(self.target_robot_id, "target_robot_id")
        _req_str(self.transformation_algorithm, "transformation_algorithm")
        _req_str(self.transformation_version, "transformation_version")
        _req_str(self.schema_version, "schema_version")
        _opt_str(self.target_model_version, "target_model_version")
        _opt_str(self.output_content_hash, "output_content_hash")
        _opt_str(self.target_report_id, "target_report_id")
        _require_ref_dicts(self.source_refs, "source_refs")
        _json_dict(self.coordinate_frames, "coordinate_frames")
        _require_ref_dicts(self.calibration_refs, "calibration_refs")
        _json_dict(self.mapping_spec, "mapping_spec")
        _json_dict(self.time_mapping, "time_mapping")
        _require_ref_dicts(self.exclusions, "exclusions")

    def to_dict(self) -> Dict[str, Any]:
        self._validate_semantics()
        return {
            "schema_version": self.schema_version,
            "source_report_id": self.source_report_id,
            "target_robot_id": self.target_robot_id,
            "target_model_version": self.target_model_version,
            "transformation_algorithm": self.transformation_algorithm,
            "transformation_version": self.transformation_version,
            "source_refs": _require_ref_dicts(self.source_refs, "source_refs"),
            "coordinate_frames": _json_dict(self.coordinate_frames, "coordinate_frames"),
            "calibration_refs": _require_ref_dicts(self.calibration_refs, "calibration_refs"),
            "mapping_spec": _json_dict(self.mapping_spec, "mapping_spec"),
            "time_mapping": _json_dict(self.time_mapping, "time_mapping"),
            "exclusions": _require_ref_dicts(self.exclusions, "exclusions"),
            "output_content_hash": self.output_content_hash,
            "target_report_id": self.target_report_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationRecord":
        if not isinstance(data, dict):
            _fail("TransformationRecord.from_dict requires an object")
        _unknown_keys(data, set(cls.__dataclass_fields__), "TransformationRecord")
        required = ["source_report_id", "target_robot_id", "transformation_algorithm", "transformation_version"]
        missing = [k for k in required if k not in data]
        if missing:
            _fail(f"TransformationRecord missing required fields: {missing}")
        return cls(**data)

    def to_json(self) -> str:
        return json.dumps(_json_safe(self.to_dict()), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> "TransformationRecord":
        return cls.from_dict(json.loads(text))


__all__ = [
    "SCHEMA_VERSION",
    "EMBODIMENTS",
    "CHECK_DOMAINS",
    "CHECK_STATUSES",
    "ACQUISITION_STATES",
    "SCOPES",
    "THRESHOLD_SOURCES",
    "ALLOWED_CHECK_DOMAINS_BY_EMBODIMENT",
    "LOCATOR_KEYS",
    "SchemaValidationError",
    "DatasetRecord",
    "CheckResult",
    "AssessmentReport",
    "TransformationRecord",
]
