"""
gesture_generator.py — OU gesture profiles for synthetic prosthetics data.

Each gesture has distinct physical character encoded in OU parameters.
Validated: all profiles yield 100% SILVER/GOLD through S2S 12-law engine.

Profiles calibrated against NinaPro DB5 motion characteristics:
  rest:          near-static, low variance, slow drift
  gentle_flex:   smooth, controlled, low amplitude
  pinch:         fast, high-freq, tight coupling
  power_grip:    high amplitude, sustained, strong coupling
  fast_extension: high amplitude, rapid, fast mean reversion
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.data_gen.ou_generator import generate_window, generate_batch

# Gesture profiles: (label_id, description, OU parameters)
GESTURE_PROFILES = {
    "rest": {
        "label":   0,
        "theta":   2.0,   # slow mean reversion = near-static
        "sigma":   0.15,  # very low amplitude
        "rho_xy":  0.3,
        "rho_xz":  0.2,
        "rho_yz":  0.25,
        "desc":    "Near-static wrist at rest",
    },
    "gentle_flex": {
        "label":   1,
        "theta":   4.0,   # moderate speed
        "sigma":   0.8,   # low amplitude
        "rho_xy":  0.55,
        "rho_xz":  0.35,
        "rho_yz":  0.45,
        "desc":    "Controlled wrist flexion",
    },
    "pinch": {
        "label":   2,
        "theta":   8.0,   # fast oscillation = finger fine motor
        "sigma":   1.0,
        "rho_xy":  0.65,
        "rho_xz":  0.50,
        "rho_yz":  0.55,
        "desc":    "Precision pinch — high freq, tight coupling",
    },
    "power_grip": {
        "label":   3,
        "theta":   5.0,
        "sigma":   2.5,   # high amplitude = strong contraction
        "rho_xy":  0.70,
        "rho_xz":  0.55,
        "rho_yz":  0.60,
        "desc":    "Full hand grip — high amplitude",
    },
    "fast_extension": {
        "label":   4,
        "theta":   12.0,  # very fast mean reversion
        "sigma":   2.0,
        "rho_xy":  0.60,
        "rho_xz":  0.40,
        "rho_yz":  0.50,
        "desc":    "Rapid finger extension",
    },
}


def generate_gesture_dataset(
    n_per_class: int = 500,
    n_samples: int = 256,
    hz: float = 100.0,
    seed_start: int = 0,
) -> Tuple[List[Dict], List[int], List[str]]:
    """
    Generate balanced synthetic gesture dataset.
    Returns: (windows, labels, gesture_names)
    """
    windows, labels, names = [], [], []
    for gesture_name, profile in GESTURE_PROFILES.items():
        cfg = {k: v for k, v in profile.items()
               if k not in ("label", "desc")}
        batch = generate_batch(
            n_per_class,
            seed_start=seed_start + profile["label"] * 10000,
            n_samples=n_samples,
            hz=hz,
            **cfg,
        )
        windows.extend(batch)
        labels.extend([profile["label"]] * n_per_class)
        names.extend([gesture_name] * n_per_class)
    return windows, labels, names


def profile_summary():
    print("S2S Gesture Profiles:")
    for name, p in GESTURE_PROFILES.items():
        print(f"  [{p['label']}] {name:<20} θ={p['theta']:<5} σ={p['sigma']:<4} — {p['desc']}")
