"""
S2S Intent Registry — semantic motion constraints.

Maps natural language intent to physics thresholds.
Allows operators to specify "how" a robot should move
without manually calculating jerk limits.

Usage:
    from s2s_standard_v1_3.intent_registry import get_intent_constraints

    constraints = get_intent_constraints("gentle")
    print(constraints["jerk_limit"])   # 50 m/s³
    print(constraints["description"])  # "Fragile items, surgery"
"""

from typing import Dict, Optional

INTENT_CONSTRAINTS: Dict[str, Dict] = {
    "gentle": {
        "jerk_limit":   50.0,
        "max_vel":       0.2,
        "description":  "Fragile items, surgery, elderly care, pediatric",
        "example":      "Feed the baby, handle glass, surgical procedure",
    },
    "careful": {
        "jerk_limit":   100.0,
        "max_vel":       0.5,
        "description":  "Collaborative robot tasks, near humans",
        "example":      "Hand a tool, pick fragile object, assembly",
    },
    "normal": {
        "jerk_limit":   500.0,
        "max_vel":       2.0,
        "description":  "Standard human motion (Flash & Hogan 1985 baseline)",
        "example":      "Pick and place, walking, reaching",
    },
    "fast": {
        "jerk_limit":   1200.0,
        "max_vel":       5.0,
        "description":  "Rapid industrial tasks",
        "example":      "High-speed sorting, rapid pick and place",
    },
    "ballistic": {
        "jerk_limit":   5000.0,
        "max_vel":      10.0,
        "description":  "Emergency stops, dynamic tasks, sports",
        "example":      "Emergency halt, throwing, high-speed reach",
    },
    # Population-specific (Issue #5)
    "amputee": {
        "jerk_limit":   350.0,
        "max_vel":       1.5,
        "description":  "Amputee prosthetic — reduced jerk tolerance",
        "example":      "Prosthetic hand control, myoelectric gesture",
        "note":         "Calibrated from NinaPro DB5 amputee subset",
    },
    "elderly": {
        "jerk_limit":   150.0,
        "max_vel":       0.8,
        "description":  "Elderly — reduced motor control speed",
        "example":      "Assistive robot, care robot tasks",
    },
    "rehabilitation": {
        "jerk_limit":   80.0,
        "max_vel":       0.3,
        "description":  "Post-injury rehabilitation movement",
        "example":      "Physical therapy, stroke recovery",
    },
}


def get_intent_constraints(intent: str) -> Dict:
    """
    Get physics constraints for a semantic intent.

    Args:
        intent: motion intent string (gentle/careful/normal/fast/ballistic/
                amputee/elderly/rehabilitation)

    Returns:
        dict with jerk_limit, max_vel, description, example
        Falls back to 'normal' if intent not found.
    """
    return INTENT_CONSTRAINTS.get(intent.lower(), INTENT_CONSTRAINTS["normal"])


def list_intents() -> Dict[str, str]:
    """List all available intents with descriptions."""
    return {k: v["description"] for k, v in INTENT_CONSTRAINTS.items()}


def intent_for_jerk(jerk_ms3: float) -> str:
    """
    Suggest an intent name based on observed jerk value.
    Useful for auto-classifying recorded motion.
    """
    if jerk_ms3 <= 50:
        return "gentle"
    elif jerk_ms3 <= 100:
        return "careful"
    elif jerk_ms3 <= 500:
        return "normal"
    elif jerk_ms3 <= 1200:
        return "fast"
    else:
        return "ballistic"
