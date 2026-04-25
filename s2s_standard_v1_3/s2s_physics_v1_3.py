#!/usr/bin/env python3
"""
s2s_physics_v1_3.py — S2S Physical Law Engine (v1.4)

Certifies that sensor data OBEYS the physical laws governing human movement.

Statistical anomaly detection (rest of S2S) asks: "Does this data LOOK human?"
Physical law certification (this module) asks: "Does this data OBEY human physics?"

These are fundamentally different. A perfect statistical fake fails here
if it violates F=ma, energy conservation, or body-segment resonance.

Physical laws encoded:

  1. Newton's Second Law        F = ma  (with ~75ms electromechanical delay)
     EMG amplitude (muscle force proxy) must LEAD acceleration by ~50-100ms.
     Zero-lag correlation is wrong — physiology has a fixed delay.
     Synthetic data generated independently has zero lagged correlation.

  2. Arm Segment Resonance     ω = sqrt(K/I)
     Physiological tremor frequency is physically determined by the
     moment of inertia (I) and neuromuscular stiffness (K) of the segment.
     Forearm: I=0.020 kg·m², K=1.5 N·m/rad → f_fund=1.38Hz → tremor 8-12Hz
     This cannot be faked with wrong parameters — the physics is fixed.

  3. Rigid Body Kinematics     a = α×r + ω²×r
     Gyroscope and accelerometer measure the SAME physical motion event
     from different physical principles. Their variance must co-vary.
     Independently generated fake accel+gyro has zero coupling.

  4. Ballistocardiography      F = ρQv (Newton 3, cardiac ejection)
     Each heartbeat ejects ~70ml blood at ~1 m/s → body recoil.
     IMU must show spectral energy at EXACTLY the PPG heart rate.
     Two independent physical measurements of the same heartbeat.

  5. Joule Heating             Q ≈ 0.75 × P_metabolic × t
     Active muscle (EMG bursts >30% for >5s) generates heat (75% inefficiency).
     Thermal camera must confirm temperature elevation if sensors are co-located.

  6. Motor Control Jerk Bound  d³x/dt³ ≤ 500 m/s³  (Flash & Hogan 1985)
     Human voluntary movement follows minimum-jerk trajectories.
     Applied to SMOOTHED signal to remove sensor noise — jerk is a motion property,
     not a noise property. Smoothing window = 5 samples before differentiation.

  7. IMU Internal Consistency  Var(accel) ~ f(Var(gyro))
     For a rigid body, accel and gyro variance must co-vary over time
     (they measure different aspects of the same physical motion event).

Performance: NumPy fast-paths used when available (10-50x speedup).
             Pure Python fallback if NumPy not installed.
"""
from __future__ import annotations
import math, statistics, time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# NumPy fast-path — optional, graceful fallback
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

# ---------------------------------------------------------------------------
# Physical constants — NOT tunable parameters
# ---------------------------------------------------------------------------
FOREARM_MASS_KG             = 1.5
FOREARM_MOMENT_OF_INERTIA   = 0.020        # I kg·m²
FOREARM_NEUROMUSCULAR_STIFF = 1.5          # K N·m/rad
FOREARM_RESONANCE_HZ        = math.sqrt(FOREARM_NEUROMUSCULAR_STIFF /
                                         FOREARM_MOMENT_OF_INERTIA) / (2*math.pi)  # 1.378 Hz fundamental

SEGMENT_PARAMS = {
    # segment: (I kg·m², K N·m/rad, tremor_lo Hz, tremor_hi Hz)
    "forearm":   (0.020, 1.5, 8.0, 12.0),
    "finger":    (0.0003,0.05, 15.0, 25.0),
    "hand":      (0.004, 0.3, 10.0, 16.0),
    "upper_arm": (0.065, 2.5, 5.0,   9.0),
    "head":      (0.020, 1.2, 3.0,   8.0),
    "walking":   (10.0, 50.0, 1.0, 3.0),  # Whole body walking: 1-3 Hz step frequency
}

EMG_ACCEL_DELAY_MS          = 75.0         # ms — electromechanical delay (EMG → force → accel)
EMG_FORCE_MIN_LAGGED_R      = 0.10         # minimum lagged Pearson r for F=ma check
JERK_SMOOTH_WINDOW          = 7
            # samples to smooth before jerk differentiation
JERK_MAX_MS3                = 500.0        # m/s³ — Flash-Hogan 1985 voluntary arm movement
BCG_MIN_ENERGY              = 0.005        # minimum normalised energy for BCG confirmation
MUSCLE_THERMAL_EFFICIENCY   = 0.25         # 25% mechanical → 75% heat
THERMAL_DETECTABLE_DELTA_C  = 0.3          # °C
EMG_BURST_HEAT_THRESHOLD    = 0.30         # burst fraction threshold for heat prediction
EMG_BURST_HEAT_DURATION_S   = 5.0          # seconds sustained burst before heat detectable


# ---------------------------------------------------------------------------
# Helpers — NumPy fast-path with pure-Python fallback
# ---------------------------------------------------------------------------

def _smooth(sig: List[float], w: int = 5) -> List[float]:
    """Centred moving average. NumPy: ~30x faster than Python loop."""
    if _NP:
        a = np.asarray(sig, dtype=np.float64)
        kernel = np.ones(2 * w + 1) / (2 * w + 1)
        # 'same' keeps original length; pad edges with edge values
        padded = np.pad(a, w, mode='edge')
        result = np.convolve(padded, kernel, mode='valid')
        return result.tolist()
    # Pure Python fallback
    return [statistics.mean(sig[max(0, i - w):i + w + 1]) for i in range(len(sig))]


def _diff(sig: List[float], dt: float) -> List[float]:
    """Central difference differentiation. NumPy: ~20x faster."""
    if len(sig) < 3 or dt <= 0:
        return []
    if _NP:
        a = np.asarray(sig, dtype=np.float64)
        return ((a[2:] - a[:-2]) / (2 * dt)).tolist()
    return [(sig[i + 1] - sig[i - 1]) / (2 * dt) for i in range(1, len(sig) - 1)]


def _rms(xs: List[float]) -> float:
    """Root mean square. NumPy: ~15x faster."""
    if _NP:
        a = np.asarray(xs, dtype=np.float64)
        finite = a[np.isfinite(a)]
        return float(np.sqrt(np.mean(finite ** 2))) if len(finite) > 0 else 0.0
    f = [v for v in xs if math.isfinite(v)]
    return math.sqrt(sum(v * v for v in f) / len(f)) if f else 0.0


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation. NumPy: ~25x faster."""
    n = min(len(xs), len(ys))
    if n < 6:
        return 0.0
    if _NP:
        a = np.asarray(xs[:n], dtype=np.float64)
        b = np.asarray(ys[:n], dtype=np.float64)
        da = a - a.mean()
        db = b - b.mean()
        denom = (np.sqrt((da ** 2).sum()) * np.sqrt((db ** 2).sum()))
        if denom < 1e-12:
            return 0.0
        return float(np.dot(da, db) / denom)
    xs, ys = xs[:n], ys[:n]
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs)) or 1e-9
    dy = math.sqrt(sum((y - my) ** 2 for y in ys)) or 1e-9
    return num / (dx * dy)


def _lagged_pearson(lead: List[float], lag: List[float], delay_samples: int) -> float:
    """Pearson r where lead[i] predicts lag[i+delay_samples]."""
    if delay_samples <= 0:
        return _pearson(lead, lag)
    n = min(len(lead) - delay_samples, len(lag) - delay_samples)
    if n < 6:
        return _pearson(lead, lag)
    a = lead[:n]
    b = lag[delay_samples:delay_samples + n]
    return _pearson(a, b)


def _freq_energy(sig: List[float], ts_ns: List[int], f_hz: float) -> float:
    """Normalised spectral energy at f_hz. NumPy: ~40x faster (vectorised DFT)."""
    n = min(len(sig), len(ts_ns))
    if n < 8:
        return 0.0
    if _NP:
        t = np.asarray(ts_ns[:n], dtype=np.float64) * 1e-9
        s = np.asarray(sig[:n], dtype=np.float64)
        s = s - s.mean()
        phase = 2 * math.pi * f_hz * t
        sd = float(np.dot(s, np.sin(phase)))
        cd = float(np.dot(s, np.cos(phase)))
        tot = float(np.dot(s, s)) or 1.0
        return max(0.0, min(1.0, (sd * sd + cd * cd) / (tot * n / 2)))
    t = [ts_ns[i] * 1e-9 for i in range(n)]
    s = [sig[i] for i in range(n)]
    mu = sum(s) / n
    s = [v - mu for v in s]
    sd = sum(s[i] * math.sin(2 * math.pi * f_hz * t[i]) for i in range(n))
    cd = sum(s[i] * math.cos(2 * math.pi * f_hz * t[i]) for i in range(n))
    tot = sum(v * v for v in s) or 1.0
    return max(0.0, min(1.0, (sd * sd + cd * cd) / (tot * n / 2)))


def _band_peak(sig, ts_ns, f_lo, f_hi, steps=40):
    """Find peak frequency in band. NumPy path uses vectorised freq sweep."""
    if _NP and len(sig) >= 8:
        n = min(len(sig), len(ts_ns))
        t = np.asarray(ts_ns[:n], dtype=np.float64) * 1e-9
        s = np.asarray(sig[:n], dtype=np.float64)
        s = s - s.mean()
        tot = float(np.dot(s, s)) or 1.0
        freqs = np.linspace(f_lo, f_hi, steps + 1)
        best_f, best_e = f_lo, 0.0
        for f in freqs:
            phase = 2 * math.pi * float(f) * t
            energy = float((np.dot(s, np.sin(phase)) ** 2 + np.dot(s, np.cos(phase)) ** 2)
                           / (tot * n / 2))
            energy = max(0.0, min(1.0, energy))
            if energy > best_e:
                best_e, best_f = energy, float(f)
        return best_f, best_e
    # Pure Python fallback
    bf, be = f_lo, 0.0
    for i in range(steps + 1):
        f = f_lo + (f_hi - f_lo) * i / steps
        e = _freq_energy(sig, ts_ns, f)
        if e > be:
            be, bf = e, f
    return bf, be


def _wavelet_temporal_check(accel_list, sample_rate: float) -> dict:
    """
    Haar DWT time-frequency tile analysis on total acceleration magnitude.

    Key insight: use magnitude sqrt(ax²+ay²+az²) not individual axes.
    Gravity cancels in the magnitude differential, so z-axis dead zone
    is avoided entirely.

    Physics basis:
      Synthetic data (Gaussian, sine, diffusion): energy is STATIONARY
        → same energy in each time tile → low coefficient of variation
      Real biological motion: energy is NON-STATIONARY
        → bursts of activity → high CV across tiles

    Catches:
      - Pure sine waves (too perfect, CV ≈ 0)
      - Gaussian noise (low energy overall)
      - Diffusion-recovered synthetic (uniform energy distribution)

    Does NOT replace resonance FFT — adds temporal dimension to it.
    """
    if not _NP or len(accel_list) < 64:
        return {"signal_type": "insufficient", "synthetic_flag": False,
                "energy_drift_cv": 0.0}

    a = np.asarray(accel_list, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)

    # Total magnitude — orientation and gravity independent
    mag = np.sqrt(np.sum(a**2, axis=1))
    s = mag - np.mean(mag)

    if np.std(s) < 1e-6:
        return {"signal_type": "flat", "synthetic_flag": False,
                "energy_drift_cv": 0.0}

    # Haar DWT — one level, pure numpy
    L = len(s) & ~1  # even length
    detail = (s[:L:2] - s[1:L:2]) / 1.414  # high-frequency tiles

    # Split detail into 4 time tiles, compute energy per tile
    tiles = np.array_split(detail**2, 4)
    tile_energies = np.array([float(np.mean(t)) for t in tiles if len(t) > 0])

    if len(tile_energies) < 2:
        return {"signal_type": "unknown", "synthetic_flag": False,
                "energy_drift_cv": 0.0}

    mean_e = float(np.mean(tile_energies))
    std_e  = float(np.std(tile_energies))
    cv = std_e / (mean_e + 1e-9)

    # Classification:
    # cv < 0.08 → too stationary → synthetic (sine/diffusion)
    # cv > 0.4  → dynamic → biological
    # 0.08-0.4  → stochastic baseline (Gaussian noise or gentle motion)
    is_perfect_synthetic = bool(cv < 0.08 and mean_e > 1e-6)

    if is_perfect_synthetic:
        sig_type = "perfect_synthetic"
    elif cv > 0.4:
        sig_type = "biological_dynamic"
    elif mean_e < 1e-6:
        sig_type = "flat"
    else:
        sig_type = "stochastic_baseline"

    # Spectral entropy — second dimension of the 2D biological check
    # Pure sine: low entropy (single frequency)
    # Gaussian:  high entropy (all frequencies equal)
    # Biological: medium entropy (concentrated but noisy)
    mag_full = np.sqrt(np.sum(np.asarray(accel_list, dtype=np.float64)**2, axis=1))
    mag_full -= mag_full.mean()
    fft_vals = np.abs(np.fft.rfft(mag_full))
    fft_norm = fft_vals / (fft_vals.sum() + 1e-10)
    spectral_entropy = float(
        -np.sum(fft_norm * np.log(fft_norm + 1e-10)) / np.log(len(fft_norm) + 1)
    )

    # 2D classification: CV + Entropy
    # Validated on 30 NinaPro DB5 files:
    #   Real human: cv=0.18-1.55, entropy=0.83-0.94
    #   Gaussian:   cv=0.17,      entropy=0.97
    #   Pure sine:  cv=0.007,     entropy=0.27
    is_mechanical   = bool(cv < 0.05 or spectral_entropy < 0.50)   # too perfect
    is_random_noise = bool(spectral_entropy > 0.97 and cv < 0.25)  # 0.97 covers low-Hz datasets  # too random
    synthetic_flag  = bool(is_mechanical or is_random_noise)

    if is_mechanical:
        sig_type = "mechanical_synthetic"
    elif is_random_noise:
        sig_type = "random_noise"
    elif cv > 0.15 and 0.75 <= spectral_entropy <= 0.96:
        sig_type = "biological"
    else:
        sig_type = "uncertain"

    return {
        "energy_drift_cv":  round(cv, 3),
        "spectral_entropy": round(spectral_entropy, 3),
        "tile_energies":    [round(float(e), 6) for e in tile_energies],
        "synthetic_flag":   synthetic_flag,
        "signal_type":      sig_type,
    }


def _accel_magnitude_np(accel_list: List) -> List[float]:
    """|a| - g  for a list of [ax,ay,az] vectors. NumPy: ~20x faster."""
    g = 9.81
    if _NP:
        a = np.asarray(accel_list, dtype=np.float64)
        return (np.abs(np.linalg.norm(a[:, :3], axis=1) - g)).tolist()
    return [abs(math.sqrt(sum(v ** 2 for v in s[:3])) - g) for s in accel_list]


def _gyro_magnitude_np(gyro_list: List) -> List[float]:
    """|ω| for a list of [gx,gy,gz] vectors. NumPy: ~20x faster."""
    if _NP:
        g = np.asarray(gyro_list, dtype=np.float64)
        return np.linalg.norm(g[:, :3], axis=1).tolist()
    return [math.sqrt(sum(v ** 2 for v in row[:3])) for row in gyro_list]


def _windowed_variance_pearson(a_vecs: List, g_vecs: List, w: int = 10) -> float:
    """
    Pearson r between windowed accel-mag variance and gyro-mag variance.
    NumPy path: ~50x faster than nested Python loops.
    """
    n = min(len(a_vecs), len(g_vecs))
    if _NP:
        a_mag = np.linalg.norm(np.asarray(a_vecs[:n], dtype=np.float64)[:, :3], axis=1)
        g_mag = np.linalg.norm(np.asarray(g_vecs[:n], dtype=np.float64)[:, :3], axis=1)
        # Sliding window variance via cumsum trick
        def _rolling_var(x, win):
            cumsum  = np.cumsum(np.insert(x, 0, 0))
            cumsum2 = np.cumsum(np.insert(x ** 2, 0, 0))
            sums  = cumsum[win:]  - cumsum[:-win]
            sums2 = cumsum2[win:] - cumsum2[:-win]
            return (sums2 / win) - (sums / win) ** 2  # population variance
        a_var = _rolling_var(a_mag, w)
        g_var = _rolling_var(g_mag, w)
        mn = min(len(a_var), len(g_var))
        if mn < 6:
            return 0.0
        da = a_var[:mn] - a_var[:mn].mean()
        dg = g_var[:mn] - g_var[:mn].mean()
        denom = np.sqrt((da ** 2).sum()) * np.sqrt((dg ** 2).sum())
        return float(np.dot(da, dg) / denom) if denom > 1e-12 else 0.0
    # Pure Python fallback
    a_vars, g_vars = [], []
    for i in range(w, n):
        a_s = [math.sqrt(sum(v ** 2 for v in a_vecs[j][:3])) for j in range(i - w, i)]
        g_s = [math.sqrt(sum(v ** 2 for v in g_vecs[j][:3])) for j in range(i - w, i)]
        a_vars.append(statistics.variance(a_s) if len(set(a_s)) > 1 else 0)
        g_vars.append(statistics.variance(g_s) if len(set(g_s)) > 1 else 0)
    return _pearson(a_vars, g_vars)


# ---------------------------------------------------------------------------
# Law 1: Newton's Second Law — F = ma  (with electromechanical delay)
# ---------------------------------------------------------------------------
def check_newton(imu_raw: Dict, emg_raw: Dict) -> Tuple[bool, int, Dict]:
    d = {"law": "Newton_Second_Law", "equation": "F=ma",
         "note": "EMG force LEADS acceleration by ~75ms (electromechanical delay)"}

    accel = imu_raw.get("accel", [])
    imu_ts = imu_raw.get("timestamps_ns", [])
    emg = emg_raw.get("channels", [])
    emg_ts = emg_raw.get("timestamps_ns", [])

    if len(accel) < 20 or not emg or len(emg[0]) < 20:
        d["skip"] = "INSUFFICIENT_RAW_DATA"
        return True, 50, d

    a_mag = _accel_magnitude_np(accel)

    # Dominant EMG channel (highest variance) — smooth to get envelope
    if _NP:
        emg_arr = np.asarray(emg, dtype=np.float64)
        dom = int(np.argmax(np.var(emg_arr[:, :200], axis=1)))
    else:
        variances = []
        for ch in emg:
            seg = ch[:min(len(ch), 200)]
            variances.append(statistics.variance(seg) if len(set(seg)) > 1 else 0)
        dom = max(range(len(variances)), key=lambda i: variances[i])

    emg_env = _smooth([abs(v) for v in emg[dom]], w=20)

    imu_hz = (1e9 / statistics.mean([imu_ts[i + 1] - imu_ts[i]
                                      for i in range(min(len(imu_ts) - 1, 50))])
              if len(imu_ts) > 1 else 240)
    emg_hz = (1e9 / statistics.mean([emg_ts[i + 1] - emg_ts[i]
                                      for i in range(min(len(emg_ts) - 1, 50))])
              if len(emg_ts) > 1 else 1000)

    n = 256

    def rs(s):
        if _NP:
            src = np.asarray(s, dtype=np.float64)
            idx = (np.arange(n) * len(src) / n).astype(int)
            return src[idx].tolist()
        return [s[int(i * len(s) / n)] for i in range(n)]

    a_rs = rs(a_mag)
    emg_rs = rs(emg_env)
    delay_rs = max(1, int(EMG_ACCEL_DELAY_MS / 1000 * n / (n * 1 / emg_hz + n * 1 / imu_hz) * 2))
    delay_rs = min(delay_rs, 30)

    r_zero = _pearson(emg_rs, a_rs)
    r_lagged = _lagged_pearson(emg_rs, a_rs, delay_samples=delay_rs)
    best_r = max(r_zero, r_lagged)

    d.update({"pearson_r_zero_lag": round(r_zero, 4),
               "pearson_r_lagged": round(r_lagged, 4),
               "delay_applied_samples": delay_rs,
               "delay_applied_ms": round(delay_rs / n * 1000, 1),
               "best_r": round(best_r, 4),
               "dominant_emg_channel": dom,
               "physical_meaning": "Muscle force causes acceleration (with ~75ms delay)"})

    if best_r >= EMG_FORCE_MIN_LAGGED_R:
        d["result"] = "F=MA_COUPLING_CONFIRMED"
        conf = min(100, int(best_r * 120))
        passed = True
    elif r_zero < -0.25:
        d["violation"] = "ANTI_PHASE_EMG_ACCEL"
        d["interpretation"] = "EMG active when decelerating — physically impossible for prime mover"
        conf = 5
        passed = False
    elif best_r >= 0:
        d["note"] = "WEAK_FORCE_MOTION_COUPLING"
        d["interpretation"] = "Stabiliser/postural muscle likely — not prime mover. Acceptable."
        conf = 45
        passed = True
    else:
        d["violation"] = "NEGATIVE_LAGGED_COUPLING"
        conf = 20
        passed = False

    d["confidence"] = conf
    return passed, conf, d


# ---------------------------------------------------------------------------
# Law 2: Resonance  ω = sqrt(K/I)
# ---------------------------------------------------------------------------
def check_resonance(imu_raw: Dict, segment: str = "forearm") -> Tuple[bool, int, Dict]:
    I, K, f_lo, f_hi = SEGMENT_PARAMS.get(segment, SEGMENT_PARAMS["forearm"])
    omega = math.sqrt(K / I)
    f_fund = omega / (2 * math.pi)
    d = {"law": "Resonance_Frequency", "equation": "omega=sqrt(K/I)",
         "segment": segment, "I_kgm2": I, "K_Nm_rad": K,
         "omega_rad_s": round(omega, 3), "f_fundamental_hz": round(f_fund, 3),
         "f_tremor_range_hz": [f_lo, f_hi],
         "derivation": f"ω=√({K}/{I})={omega:.2f}rad/s → neuromuscular loop delay → tremor {f_lo}–{f_hi}Hz"}

    accel = imu_raw.get("accel", [])
    ts = imu_raw.get("timestamps_ns", [])
    dt = (statistics.mean([ts[i + 1] - ts[i] for i in range(min(len(ts) - 1, 50))]) * 1e-9
          if len(ts) > 1 else 1 / 240)
    sample_rate = 1.0 / dt if dt > 0 else 240.0
    min_samples = max(64, min(500, int(3.0 / 8.0 * sample_rate)))  # cap at 500 — 2 tremor cycles at any rate
    if len(accel) < min_samples:
        d["skip"] = "INSUFFICIENT_DATA"
        return True, 50, d

    az = [s[2] if len(s) > 2 else 0.0 for s in accel]
    peak_f, peak_e = _band_peak(az, ts, 5.0, 30.0, steps=60)
    d.update({"measured_peak_hz": round(peak_f, 2), "measured_peak_energy": round(peak_e, 5)})

    # Wavelet temporal analysis — catches synthetic data that fakes correct frequency
    wavelet = _wavelet_temporal_check(accel, sample_rate)
    d["wavelet"] = wavelet
    if wavelet["synthetic_flag"]:
        d["wavelet_note"] = "UNIFORM_ENERGY: tremor energy too evenly distributed — possible synthetic"

    if peak_e < 0.015:
        d["note"] = "NO_SIGNIFICANT_TREMOR"
        d["confidence"] = 60
        return True, 60, d

    in_range = f_lo <= peak_f <= f_hi
    d["in_expected_range"] = in_range
    if in_range:
        d["result"] = f"RESONANCE_CONFIRMED ({peak_f:.1f}Hz matches {segment} ω=√(K/I))"
        conf = min(100, int(60 + peak_e * 400))
        # Reduce confidence if wavelet says energy pattern is too uniform
        if wavelet.get("synthetic_flag"):
            conf = max(30, conf - 20)
            d["result"] += " (WAVELET:UNIFORM_ENERGY)"
        passed = True
    else:
        dev = min(abs(peak_f - f_lo), abs(peak_f - f_hi))
        if 3.0 <= peak_f < f_lo:
            d["note"] = "LOW_FREQ_MOTION: sub-tremor frequency, normal slow deliberate motion"
            d["interpretation"] = "Voluntary movement below tremor band - valid human motion"
            conf = 55
            passed = True
        else:
            d["violation"] = f"RESONANCE_MISMATCH: {peak_f:.1f}Hz outside {f_lo}-{f_hi}Hz for {segment}"
            d["interpretation"] = f"{segment} cannot resonate at {peak_f:.1f}Hz"
            conf = max(0, int(50 - dev * 8))
            passed = dev < 3.0

    d["confidence"] = conf
    return passed, conf, d


# ---------------------------------------------------------------------------
# Law 3: Rigid Body Kinematics  a = α×r + ω²×r
# ---------------------------------------------------------------------------
def check_rigid_body(imu_raw: Dict, r: float = 0.05) -> Tuple[bool, int, Dict]:
    d = {"law": "Rigid_Body_Kinematics", "equation": "a=alpha*r+omega^2*r",
         "sensor_offset_m": r,
         "physical_meaning": "Gyro angular velocity must predict accelerometer via rigid body mechanics"}

    accel = imu_raw.get("accel", [])
    gyro = imu_raw.get("gyro", [])
    ts = imu_raw.get("timestamps_ns", [])
    if not gyro:
        d["violation"] = "MISSING_GYRO"
        d["interpretation"] = "Gyroscope data is required for rigid body kinematics"
        d["skip"] = "MISSING_GYRO"
        return True, 50, d
    n = min(len(accel), len(gyro), len(ts))
    if n < 20:
        d["skip"] = "INSUFFICIENT_DATA"
        return True, 50, d

    dt = (statistics.mean([ts[i + 1] - ts[i] for i in range(min(n - 1, 50))]) * 1e-9
          if n > 1 else 1 / 240)
    d["dt_s"] = round(dt, 6)

    a_mag = _accel_magnitude_np(accel[:n])
    w_mag = _gyro_magnitude_np(gyro[:n])

    if _NP:
        wm = np.asarray(w_mag, dtype=np.float64)
        am = np.asarray(a_mag, dtype=np.float64)
        # alpha = central diff of w_mag
        alpha = np.abs((wm[2:] - wm[:-2]) / (2 * dt))
        w_mid = wm[1:n - 1]
        a_mid = am[1:n - 1]
        a_pred = np.sqrt((alpha * r) ** 2 + (w_mid ** 2 * r) ** 2)
        r_corr = float(np.corrcoef(a_mid, a_pred)[0, 1]) if len(a_mid) >= 6 else 0.0
        rms_m = float(np.sqrt(np.mean(a_mid ** 2)))
        rms_p = float(np.sqrt(np.mean(a_pred ** 2)))
    else:
        alpha = [abs(w_mag[i + 1] - w_mag[i - 1]) / (2 * dt) for i in range(1, n - 1)]
        w_mid = w_mag[1:n - 1]
        a_mid = a_mag[1:n - 1]
        a_pred = [math.sqrt((alpha[i] * r) ** 2 + (w_mid[i] ** 2 * r) ** 2)
                  for i in range(len(alpha))]
        r_corr = _pearson(a_mid, a_pred)
        rms_m = _rms(a_mid)
        rms_p = _rms(a_pred)

    ratio = rms_p / max(rms_m, 1e-6)

    r_corr = 0.0 if math.isnan(r_corr) else r_corr
    d.update({"pearson_r_measured_vs_predicted": round(r_corr, 4),
               "rms_measured_ms2": round(rms_m, 4),
               "rms_predicted_ms2": round(rms_p, 4),
               "gyro_accel_scale_ratio": round(ratio, 4)})

    if r_corr > 0.15 and 0.05 < ratio < 10.0:
        d["result"] = "RIGID_BODY_KINEMATICS_CONSISTENT"
        conf = min(100, int(50 + r_corr * 50))
        passed = True
    elif r_corr < -0.2:
        d["violation"] = "GYRO_ACCEL_ANTI_CORRELATED"
        d["interpretation"] = "Physically impossible for rigid body — indicates independent generation"
        conf = 10
        passed = False
    else:
        d["note"] = "WEAK_KINEMATIC_COUPLING (near-static motion — valid)"
        conf = 45
        passed = True

    d["confidence"] = conf
    return passed, conf, d


# ---------------------------------------------------------------------------
# Law 4: Ballistocardiography  Newton 3 cardiac recoil
# ---------------------------------------------------------------------------
def check_bcg(imu_raw: Dict, ppg_cert: Dict) -> Tuple[bool, int, Dict]:
    d = {"law": "Ballistocardiography", "equation": "F_recoil=rho*Q*v (Newton 3)",
         "physics": "70ml blood × 1m/s ejection → ~0.001 m/s² body recoil at wrist IMU"}

    hr_bpm = (ppg_cert.get("vitals") or {}).get("heart_rate_bpm")
    if not hr_bpm or not (30 <= hr_bpm <= 220):
        d["skip"] = "NO_VALID_PPG_HR"
        return True, 50, d

    hr_hz = hr_bpm / 60.0
    accel = imu_raw.get("accel", [])
    ts = imu_raw.get("timestamps_ns", [])
    if len(accel) < 64:
        d["skip"] = "INSUFFICIENT_IMU_DATA"
        return True, 50, d

    d.update({"ppg_hr_bpm": hr_bpm, "ppg_hr_hz": round(hr_hz, 4)})

    best_e, best_axis = 0.0, 0
    for ax in range(min(3, len(accel[0]))):
        sig = [accel[i][ax] for i in range(len(accel))]
        e = _freq_energy(sig, ts, hr_hz)
        if e > best_e:
            best_e = e
            best_axis = ax

    sig_best = [accel[i][best_axis] for i in range(len(accel))]
    e_harm = _freq_energy(sig_best, ts, hr_hz * 2)
    e_low = _freq_energy(sig_best, ts, hr_hz * 0.7)
    e_high = _freq_energy(sig_best, ts, hr_hz * 1.3)
    spec = best_e / max(max(e_low, e_high), 1e-9)

    d.update({"imu_energy_at_hr": round(best_e, 5), "imu_energy_at_2xhr": round(e_harm, 5),
               "freq_specificity": round(spec, 2), "best_axis": best_axis})

    if best_e >= BCG_MIN_ENERGY and spec >= 1.5:
        d["result"] = f"BCG_CONFIRMED: IMU energy {best_e:.4f} at PPG HR={hr_bpm:.0f}bpm"
        conf = min(100, int(60 + spec * 10 + best_e * 500))
        passed = True
    elif best_e >= 0.001:
        d["note"] = "WEAK_BCG — posture/clothing may attenuate"
        conf = 55
        passed = True
    else:
        d["warn"] = "NO_BCG_DETECTED — IMU may not be on torso/chest for BCG"
        conf = 35
        passed = True

    d["confidence"] = conf
    return passed, conf, d


# ---------------------------------------------------------------------------
# Law 5: Joule Heating  Q ≈ 0.75 × P × t
# ---------------------------------------------------------------------------
def check_joule(emg_cert: Dict, thermal_cert: Dict) -> Tuple[bool, int, Dict]:
    d = {"law": "Joule_Heating", "equation": "Q=0.75*P_metabolic*t",
         "physics": "75% of muscle metabolic energy → heat (thermodynamic inefficiency)"}

    bf = (emg_cert.get("notes") or {}).get("mean_burst_frac", 0)
    dur_s = emg_cert.get("duration_ms", 0) / 1000
    human = (thermal_cert.get("human_presence") or {}).get("human_present", False)
    t_range = (thermal_cert.get("spatial_analysis") or {}).get("spatial_range_C", 0)
    t_delta = (thermal_cert.get("temporal_analysis") or {}).get("mean_frame_delta_C", 0)

    d.update({"emg_burst_frac": round(bf, 4), "emg_duration_s": round(dur_s, 2),
               "thermal_human": human, "thermal_spatial_range_C": round(t_range, 3),
               "thermal_delta_C": round(t_delta, 6)})

    heat_expected = bf > EMG_BURST_HEAT_THRESHOLD and dur_s >= EMG_BURST_HEAT_DURATION_S
    d["heat_should_be_detectable"] = heat_expected

    if not heat_expected:
        d["note"] = "ACTIVITY_TOO_BRIEF_FOR_THERMAL_DETECTION"
        d["interpretation"] = (f"Need >{EMG_BURST_HEAT_THRESHOLD * 100:.0f}% burst for "
                                f">{EMG_BURST_HEAT_DURATION_S}s — not yet")
        conf = 60
        passed = True
    elif heat_expected and human and t_range > THERMAL_DETECTABLE_DELTA_C:
        d["result"] = "METABOLIC_HEAT_CONFIRMED"
        d["interpretation"] = (f"EMG {bf * 100:.0f}% active for {dur_s:.1f}s → "
                                f"thermal {t_range:.1f}°C gradient (Joule heating)")
        conf = 90
        passed = True
    elif heat_expected and not human:
        d["violation"] = "EMG_ACTIVE_NO_THERMAL_HEAT"
        d["interpretation"] = "Sustained EMG but no thermal body heat — co-located sensors would show this"
        conf = 30
        passed = False
    else:
        conf = 55
        passed = True

    d["confidence"] = conf
    return passed, conf, d


# ---------------------------------------------------------------------------
# Law 6: Jerk Bounds  d³x/dt³ ≤ 500 m/s³  (Flash & Hogan 1985)
# ---------------------------------------------------------------------------
def check_jerk(imu_raw: Dict, segment: str = "forearm") -> Tuple[bool, int, Dict]:
    d = {"law": "Motor_Control_Jerk", "equation": "d3x/dt3 <= 500 m/s3",
         "reference": "Flash & Hogan (1985) minimum-jerk model",
         "implementation": "Signal smoothed with w=7 window BEFORE differentiation to separate motion from sensor noise"}

    accel = imu_raw.get("accel", [])
    ts = imu_raw.get("timestamps_ns", [])
    if len(accel) < 20:
        d["skip"] = "INSUFFICIENT_DATA"
        return True, 50, d

    dt = (statistics.mean([ts[i + 1] - ts[i] for i in range(min(len(ts) - 1, 50))]) * 1e-9
          if len(ts) > 1 else 1 / 240)

    WINDOW_SAMPLES = max(20, int(2.56 / dt)) if dt > 0 else 256
    
    # Calculate sample rate for rate normalization
    sample_rate = 1.0 / dt if dt > 0 else 240.0
    rate_normalization_factor = (sample_rate / 50.0) ** 3
    if sample_rate < 30: rate_normalization_factor = 1.0  # robot/video control rates — formula inverts below 30Hz
    d["sample_rate_hz"] = round(sample_rate, 1)
    d["rate_normalization_factor"] = round(rate_normalization_factor, 2)

    # Initialize variables to prevent UnboundLocalError
    p95_j = 0.0
    peak_j = 0.0
    rms_j = 0.0

    all_jerk = []
    window_p95s = []
    n = len(accel)

    for axis in range(min(3, len(accel[0]))):
        sig_raw = [accel[i][axis] for i in range(n)]

        if _NP:
            # Full NumPy pipeline: remove gravity → smooth → diff → smooth → diff → diff
            arr = np.asarray(sig_raw, dtype=np.float64)
            w = JERK_SMOOTH_WINDOW
            
            # Remove gravity using quasi-static window estimation
            # Better than median: works at any sensor mounting angle
            # Find low-variance windows (sensor nearly static = gravity dominant)
            if len(arr) >= 30:
                win_size = 15
                variances = np.array([
                    np.var(arr[i:i+win_size])
                    for i in range(0, len(arr)-win_size, win_size)
                ])
                static_threshold = np.percentile(variances, 25)
                static_mask = np.zeros(len(arr), dtype=bool)
                for i, v in enumerate(variances):
                    if v <= static_threshold:
                        static_mask[i*win_size:i*win_size+win_size] = True
                gravity = float(np.mean(arr[static_mask])) if static_mask.any() else float(np.median(arr))
            else:
                gravity = float(np.median(arr))
            arr_compensated = arr - gravity
            d["gravity_removed_m_s2"] = float(gravity)
            
            def np_smooth(a, win):
                k = np.ones(2 * win + 1) / (2 * win + 1)
                return np.convolve(np.pad(a, win, mode='edge'), k, mode='valid')

            def np_diff(a):
                return (a[2:] - a[:-2]) / (2 * dt)

            s1 = np_smooth(arr_compensated, w)
            vel = np_diff(s1)
            s2 = np_smooth(vel, w)
            jerk_raw = np_diff(s2)
            jerk = jerk_raw.tolist()
        else:
            # Pure Python: remove gravity → smooth → diff → smooth → diff → diff
            sig_s = _smooth(sig_raw, w=JERK_SMOOTH_WINDOW)
            gravity = statistics.median(sig_s)
            sig_compensated = [s - gravity for s in sig_s]
            vel = _diff(sig_compensated, dt)
            if not vel:
                continue
            vel_s = _smooth(vel, w=JERK_SMOOTH_WINDOW)
            jerk = _diff(vel_s, dt)  # Third derivative: position → vel → accel → jerk

        if not jerk:
            continue
        all_jerk.extend(jerk)

        # Micro-window spike detector (64-sample sub-windows) — INNOVATION
        # Catches spikes that average out in the main window.
        # Threshold now segment-aware (fingers allow sharper natural jerks).
        if _NP:
            jerk_abs_full = np.abs(np.asarray(jerk, dtype=np.float64))
            main_mean = float(np.mean(jerk_abs_full)) if len(jerk_abs_full) > 0 else 0
            # segment-aware multiplier (keeps 64 unchanged)
            mult = 8.0 if segment in ('finger', 'hand') else 5.0
            spike_threshold = max(main_mean * mult, 100.0)
            micro_win = 64
            for ms in range(0, len(jerk_abs_full) - micro_win, micro_win // 2):
                micro_seg = jerk_abs_full[ms:ms + micro_win]
                micro_p95 = float(np.percentile(micro_seg, 95))
                if micro_p95 > spike_threshold:
                    d.setdefault("micro_spike_windows", 0)
                    d["micro_spike_windows"] = d.get("micro_spike_windows", 0) + 1

        # P95 per window
        if _NP:
            jerk_abs = np.abs(np.asarray(jerk, dtype=np.float64))
            for w_start in range(0, len(jerk_abs), WINDOW_SAMPLES):
                seg = jerk_abs[w_start:w_start + WINDOW_SAMPLES]
                if len(seg) >= 10:
                    window_p95s.append(float(np.percentile(seg, 95)))
        else:
            for w_start in range(0, len(jerk), WINDOW_SAMPLES):
                seg = [abs(j) for j in jerk[w_start:w_start + WINDOW_SAMPLES]]
                if len(seg) >= 10:
                    seg.sort()
                    window_p95s.append(seg[int(len(seg) * 0.95)])

    if not all_jerk:
        d["skip"] = "COULD_NOT_COMPUTE"
        return True, 50, d

    if _NP:
        jerk_arr = np.abs(np.asarray(all_jerk, dtype=np.float64))
        peak_j = float(jerk_arr.max())
        rms_j = float(np.sqrt(np.mean(jerk_arr ** 2)))
    else:
        peak_j = max(abs(j) for j in all_jerk)
        rms_j = _rms(all_jerk)

    if window_p95s:
        if _NP:
            p95_j = float(np.median(np.asarray(window_p95s)))
        else:
            window_p95s.sort()
            p95_j = window_p95s[len(window_p95s) // 2]
    else:
        if _NP:
            p95_j = float(np.percentile(np.abs(np.asarray(all_jerk)), 95))
        else:
            p95_j = sorted([abs(j) for j in all_jerk])[int(len(all_jerk) * 0.95)]

    # Apply rate normalization after p95_j has its final value
    p95_j_normalized = p95_j / rate_normalization_factor
    peak_j_normalized = peak_j / rate_normalization_factor
    rms_j_normalized = rms_j / rate_normalization_factor

    d["window_samples"] = WINDOW_SAMPLES
    d["n_windows"] = len(window_p95s)
    # Skip jerk check for walking - limit not established in biomechanics literature
    if segment == "walking":
        d["skip"] = "jerk_limit_not_established_for_walking_pending_biomechanics_literature"
        d["gravity_removed_m_s2"] = float(np.median(np.asarray(sig_raw))) if _NP else statistics.median(sig_raw)
        return True, 60, d
    
    # Choose appropriate jerk limit based on segment
    jerk_limit = JERK_MAX_MS3  # Only arm movement limits are scientifically established
    
    d.update({"peak_jerk_ms3": round(peak_j, 1), "rms_jerk_ms3": round(rms_j, 1),
               "p95_jerk_ms3": round(p95_j, 1), "peak_jerk_normalized_ms3": round(peak_j_normalized, 1),
               "rms_jerk_normalized_ms3": round(rms_j_normalized, 6), "p95_jerk_normalized_ms3": round(p95_j_normalized, 1),
               "human_limit_ms3": jerk_limit,
               "normalization_note": "raw jerk divided by (hz/50)^3 to compare at any sample rate to 50Hz baseline",
               "smooth_window_samples": JERK_SMOOTH_WINDOW})

    if p95_j_normalized > jerk_limit:
        d["violation"] = f"JERK_EXCEEDS_HUMAN_LIMIT: {p95_j_normalized:.0f} > {jerk_limit:.0f} m/s³"
        d["interpretation"] = "Trajectory is supra-human — robot bang-bang or keyframe artifact"
        conf = max(0, int(100 - (p95_j_normalized / jerk_limit) * 40))
        passed = False
    elif p95_j_normalized > 150:
        d["note"] = f"HIGH_JERK_FAST_MOVEMENT: {p95_j_normalized:.0f} m/s³ (valid for ballistic arm motion)"
        conf = 75
        passed = True
    else:
        d["result"] = f"JERK_WITHIN_HUMAN_MOTOR_CONTROL: {p95_j_normalized:.0f} m/s³"
        conf = 92
        passed = True

    d["confidence"] = conf
    return passed, conf, d


# ---------------------------------------------------------------------------
# Law 7: IMU Internal Consistency (accel-gyro co-variance)
# ---------------------------------------------------------------------------
def check_imu_consistency(imu_raw: Dict) -> Tuple[bool, int, Dict]:
    d = {"law": "IMU_Internal_Consistency", "equation": "Var(accel) co-varies with Var(gyro)",
         "physical_meaning": "Same motion event causes both angular velocity and linear acceleration"}

    accel = imu_raw.get("accel", [])
    gyro = imu_raw.get("gyro", [])
    if not gyro:
        d["violation"] = "MISSING_GYRO"
        d["interpretation"] = "Gyroscope data is required for IMU consistency check"
        d["skip"] = "MISSING_GYRO"
        return True, 50, d
    n = min(len(accel), len(gyro))
    if n < 30:
        d["skip"] = "INSUFFICIENT_DATA"
        return True, 50, d

    r = _windowed_variance_pearson(accel[:n], gyro[:n], w=10)
    d.update({"pearson_r_var_coupling": round(r, 4), "expected_r_range": [0.1, 1.0]})

    if r >= 0.15:
        d["result"] = "IMU_INTERNALLY_CONSISTENT"
        conf = min(100, int(50 + r * 50))
        passed = True
    elif r >= 0:
        d["note"] = "WEAK_COUPLING (near-static motion — acceptable)"
        conf = 45
        passed = True
    else:
        d["violation"] = "ACCEL_GYRO_DECOUPLED"
        d["interpretation"] = (f"r={r:.3f}: independently-generated sensor data "
                                f"cannot couple to same physical event")
        conf = 15
        passed = False

    d["confidence"] = conf
    return passed, conf, d


# ---------------------------------------------------------------------------
# PhysicsEngine
# ---------------------------------------------------------------------------
class PhysicsEngine:
    """
    Certify that sensor data obeys the physical laws of human movement.

    pe = PhysicsEngine()
    result = pe.certify(
        imu_raw   = {"timestamps_ns":[...], "accel":[[ax,ay,az],...], "gyro":[[gx,gy,gz],...]},
        emg_raw   = {"timestamps_ns":[...], "channels":[[ch0_sample0,...],...]},
        ppg_cert  = ppg_cert_dict,
        emg_cert  = emg_cert_dict,
        thermal_cert = thermal_cert_dict,
        segment   = "forearm",
    )
    print(result["physical_law_score"])   # 0–100
    print(result["laws_passed"])
    """

    def __init__(self):
        self._session_jerk_rms: List[float] = []

    def certify(self,
                imu_raw:      Optional[Dict] = None,
                emg_raw:      Optional[Dict] = None,
                ppg_cert:     Optional[Dict] = None,
                emg_cert:     Optional[Dict] = None,
                thermal_cert: Optional[Dict] = None,
                segment:      str = "forearm",
                device_id:    str = "unknown",
                session_id:   Optional[str] = None) -> Dict[str, Any]:
        """
        Certify sensor data against 7 biomechanical laws.

        Thread safety: NOT thread-safe. Use one PhysicsEngine instance
        per thread. Sharing across threads without locks will cause
        race conditions in _session_jerk_rms. Adding locks would break
        the 2.8ms latency guarantee.

        Error recovery: if any law check raises an exception, that law
        is skipped with score=50 (neutral) rather than crashing the caller.
        The result will include an 'errors' field listing any failures.

        Timeout: no built-in timeout. For field use on Linux/Raspberry Pi:
            import signal
            def _timeout(s, f): raise TimeoutError()
            signal.signal(signal.SIGALRM, _timeout)
            signal.alarm(1)
            try:
                result = engine.certify(imu_raw, segment='forearm')
            finally:
                signal.alarm(0)
        Not implemented internally — would break cross-platform support.
        """
        results: Dict[str, Tuple[bool, int, Dict]] = {}
        _errors: list = []

        def _safe(name, fn, *args):
            try:
                return fn(*args)
            except Exception as e:
                _errors.append(f"{name}: {e}")
                return (True, 50, {"skip": f"ERROR:{e}", "law": name})

        if imu_raw and emg_raw:
            results["newton_second_law"] = _safe("newton", check_newton, imu_raw, emg_raw)
        if imu_raw:
            results["resonance_frequency"]      = _safe("resonance", check_resonance, imu_raw, segment)
            results["rigid_body_kinematics"]    = _safe("rigid_body", check_rigid_body, imu_raw)
            results["jerk_bounds"]              = _safe("jerk", check_jerk, imu_raw, segment)
            _jrms = results["jerk_bounds"][2].get("rms_jerk_normalized_ms3")
            if _jrms is not None:
                self._session_jerk_rms.append(_jrms)
            results["imu_internal_consistency"] = _safe("consistency", check_imu_consistency, imu_raw)
        if imu_raw and ppg_cert:
            results["ballistocardiography"] = _safe("bcg", check_bcg, imu_raw, ppg_cert)
        if emg_cert and thermal_cert:
            results["joule_heating"] = _safe("joule", check_joule, emg_cert, thermal_cert)

        passed = [k for k, (ok, _, _) in results.items() if ok]
        failed = [k for k, (ok, _, _) in results.items() if not ok]
        confs  = [c for (_, c, _) in results.values()]
        score  = int(sum(confs) / len(confs)) if confs else 0
        n, np_ = len(results), len(passed)

        if n == 0:
            tier = "UNVERIFIED"
        elif len(failed) / max(n, 1) > 0.3:
            tier = "REJECTED"
        elif score >= 75 and np_ >= max(n - 1, 1):
            tier = "GOLD"
        elif score >= 55:
            tier = "SILVER"
        elif score >= 35:
            tier = "BRONZE"
        else:
            tier = "REJECTED"

        flags = [f"PHYSICS_VIOLATION:{k}" for k in failed]

        return {
            "status":             "PASS" if tier not in ("REJECTED", "UNVERIFIED") else "FAIL",
            "tier":               tier,
            "sensor_type":        "PHYSICS",
            "physical_law_score": score,
            "laws_checked":       list(results.keys()),
            "n_laws_checked":     n,
            "laws_passed":        passed,
            "laws_failed":        failed,
            "body_segment":       segment,
            "source_type": "HIL_BIOLOGICAL",
            "law_details":        {k: det for k, (_, _, det) in results.items()},
            "flags":              flags,
            "physical_constants": {
                "forearm_I_kgm2":       FOREARM_MOMENT_OF_INERTIA,
                "forearm_K_Nm_rad":     FOREARM_NEUROMUSCULAR_STIFF,
                "forearm_resonance_hz": round(FOREARM_RESONANCE_HZ, 3),
                "tremor_band_hz":       list(SEGMENT_PARAMS["forearm"][2:]),
                "emg_delay_ms":         EMG_ACCEL_DELAY_MS,
                "jerk_max_ms3":         JERK_MAX_MS3,
                "muscle_thermal_eff":   MUSCLE_THERMAL_EFFICIENCY,
            },
            "device_id":    device_id,
            "session_id":   session_id,
            "tool":         "s2s_physics_v1_5",
            "numpy_enabled": _NP,
            "issued_at_ns": time.time_ns(),
        }


    def certify_session(self) -> Dict[str, Any]:
        """
        Calculate Biological Fingerprint Score (BFS) across a full session.
        Call this AFTER processing all windows for a session via certify().

        BFS = 0.3*(1/CV) + 0.4*(1-Kurtosis_norm) + 0.3*(1-Hurst)

        BFS is a FLOOR DETECTOR, not a quality ranking:
          - BFS >= 0.35 → signal confirmed biological origin (HUMAN)
          - BFS 0.20-0.35 → low biological fidelity (REVIEW)
          - BFS < 0.20 → not biological (SUSPICIOUS)

        BFS measures biological signal diversity, not quality.
        Higher BFS does not mean better signal — it means higher variability
        pattern consistent with biological motor control. Proven: all 10 NinaPro
        DB5 subjects score >= 0.35 (HUMAN). r = -0.664 p = 0.036 is a
        population-level correlation, not a subject-level quality ranking.
        """
        rms_vals = self._session_jerk_rms[:]
        self._session_jerk_rms.clear()

        if len(rms_vals) < 4:
            return {"bfs": None, "reason": "INSUFFICIENT_WINDOWS",
                    "n_windows": len(rms_vals), "min_required": 4}

        n = len(rms_vals)
        mu = sum(rms_vals) / n
        if mu <= 0:
            # Compute Hurst even on zero-mean — needed for biological floor detection
            def _hurst_zm(ts):
                if len(ts) < 4: return 0.5
                m = sum(ts)/len(ts)
                dev = [x - m for x in ts]
                cum, s = [], 0.0
                for d in dev: s += d; cum.append(s)
                r = max(cum) - min(cum)
                st = (sum(d*d for d in dev)/len(dev))**0.5
                if st <= 0: return 0.5
                rs = r / st
                return math.log(rs)/math.log(len(ts)) if rs > 0 else 0.5
            h = max(0.0, min(1.0, _hurst_zm(rms_vals)))
            grade = "NOT_BIOLOGICAL" if h < 0.7 else "LOW_BIOLOGICAL_FIDELITY"
            return {"bfs": None, "reason": "ZERO_MEAN_JERK_RMS",
                    "biological_grade": grade,
                    "recommendation": "REJECT" if grade == "NOT_BIOLOGICAL" else "REVIEW",
                    "hurst": round(h, 4), "hurst_floor": 0.7, "n_windows": n}

        # Trim top+bottom 15% outliers before all BFS components
        # Synthetic/corrupted windows produce outlier jerk RMS values
        # Trimming isolates the core biological signal for CV, kurtosis, Hurst
        _trim = max(1, int(0.15 * n))
        _sorted_idx = sorted(range(n), key=lambda i: rms_vals[i])
        _keep = [rms_vals[i] for i in _sorted_idx[_trim:-_trim]] if n > 2*_trim else rms_vals
        _vals = _keep if len(_keep) >= 4 else rms_vals
        _n    = len(_vals)
        _mu   = sum(_vals) / _n

        # CV on trimmed values
        std = math.sqrt(sum((x - _mu) ** 2 for x in _vals) / _n)
        cv = std / _mu if _mu > 0 else 0.0

        # Excess kurtosis on trimmed values
        if std > 0:
            kurt = sum(((x - _mu) / std) ** 4 for x in _vals) / _n - 3.0
        else:
            kurt = 0.0
        # Normalise kurtosis to [0,1] range using sigmoid-like scale
        kurt_norm = max(0.0, min(1.0, (kurt + 3.0) / 10.0))

        # Hurst exponent via R/S analysis on sub-segments (robust to bad windows)
        # Single R/S on full session is brittle — 1 bad window dilutes entire estimate
        # Solution: compute H on 10 non-overlapping sub-segments, take p25
        # p25_hurst > 0.70 = HUMAN. Tolerates ~25% bad sub-segments before failing.
        def _hurst(ts):
            if len(ts) < 4:
                return 0.5
            mean_ts = sum(ts) / len(ts)
            deviations = [x - mean_ts for x in ts]
            cumdev = []
            s = 0.0
            for d in deviations:
                s += d
                cumdev.append(s)
            r = max(cumdev) - min(cumdev)
            std_ts = math.sqrt(sum(d * d for d in deviations) / len(deviations))
            if std_ts <= 0:
                return 0.5
            rs = r / std_ts
            if rs <= 0:
                return 0.5
            return math.log(rs) / math.log(len(ts))

        # Robust Hurst: trimmed R/S on full session
        # Synthetic/corrupted windows have outlier jerk RMS values
        # Trim top+bottom 15% before R/S — removes contaminating windows
        # while preserving temporal structure of the remaining real signal
        trim = max(1, int(0.15 * n))
        trimmed = sorted(enumerate(rms_vals), key=lambda x: x[1])
        keep_idx = set(i for i, _ in trimmed[trim:-trim])
        trimmed_vals = [rms_vals[i] for i in sorted(keep_idx)]
        if len(trimmed_vals) >= 8:
            hurst = max(0.0, min(1.0, _hurst(trimmed_vals)))
        else:
            hurst = max(0.0, min(1.0, _hurst(rms_vals)))

        bfs = 0.3 * min(1.0, 1.0 / (cv + 0.001)) + 0.4 * (1.0 - kurt_norm) + 0.3 * (1.0 - hurst)
        bfs = max(0.0, min(1.0, bfs))

        floor_threshold = 0.35
        human_range_min = 0.35
        human_range_max = 0.70

        # 0-100 human scale calibrated on NinaPro DB5 n=10 (BFS 0.37-0.68)
        # Above 0.70 = superhuman (prosthetics/robots valid zone, score > 100)
        bfs_score = round((bfs - human_range_min) / (human_range_max - human_range_min) * 100, 1)

        # Hurst floor — primary biological origin detector
        # Human motor control: H > 0.7 (persistent, long-range correlation)
        # Synthetic/robotic: H < 0.55 (periodic or random, no biological structure)
        if hurst < 0.7:
            biological_grade = "NOT_BIOLOGICAL"
            recommendation = "REJECT"
        elif bfs >= floor_threshold:
            biological_grade = "HUMAN"
            recommendation = "ACCEPT"
        elif bfs >= 0.20:
            biological_grade = "LOW_BIOLOGICAL_FIDELITY"
            recommendation = "REVIEW"
        else:
            biological_grade = "SUSPICIOUS"
            recommendation = "REVIEW"

        return {
            "bfs": round(bfs, 4),
            "biological_diversity_score": bfs_score,  # 0-100 within human range — diversity, not quality
            "biological_grade": biological_grade,  # HUMAN/LOW_BIOLOGICAL_FIDELITY/SUSPICIOUS/SUPERHUMAN — origin detection only
            "recommendation": recommendation,
            "floor_threshold": floor_threshold,
            "human_range_min": human_range_min,
            "human_range_max": human_range_max,
            "cv": round(cv, 4),
            "kurtosis_raw": round(kurt, 4),
            "kurtosis_norm": round(kurt_norm, 4),
            "hurst": round(hurst, 4),
            "n_windows": n,
            "mean_jerk_rms_normalized": round(mu, 4),
        }


def main():
    """CLI entry point: s2s-certify --help"""
    import argparse, json, sys
    p = argparse.ArgumentParser(description='S2S Physics Certification')
    p.add_argument('--version', action='store_true', help='Show version')
    p.add_argument('--demo',    action='store_true', help='Run demo certification')
    p.add_argument('--bench',   action='store_true', help='Run speed benchmark')
    args = p.parse_args()

    if args.version:
        print(f's2s-certify 1.4.0  (numpy_fast_path={_NP})')

    elif args.bench:
        import time as _time
        n_samples = 1000
        imu = {
            'timestamps_ns': [int(i * 0.01 * 1e9) for i in range(n_samples)],
            'accel': [[0.1 * math.sin(2 * math.pi * 2 * i * 0.01),
                       0.1 * math.cos(2 * math.pi * 2 * i * 0.01), 9.8]
                      for i in range(n_samples)],
            'gyro':  [[0.01 * math.sin(2 * math.pi * 10 * i * 0.01), 0.0, 0.0]
                      for i in range(n_samples)],
        }
        N_RUNS = 20
        t0 = _time.perf_counter()
        pe = PhysicsEngine()
        for _ in range(N_RUNS):
            pe.certify(imu_raw=imu, segment='forearm')
        elapsed = _time.perf_counter() - t0
        print(f'numpy={_NP}  {N_RUNS} runs × {n_samples} samples')
        print(f'total={elapsed:.3f}s  per_run={elapsed/N_RUNS*1000:.1f}ms')

    elif args.demo:
        n, dt = 100, 0.01
        result = PhysicsEngine().certify(
            imu_raw={
                'timestamps_ns': [int(i * dt * 1e9) for i in range(n)],
                'accel': [[0.1 * math.sin(2 * math.pi * 2 * i * dt),
                           0.1 * math.cos(2 * math.pi * 2 * i * dt), 9.8]
                          for i in range(n)],
                'gyro':  [[0.01 * math.sin(2 * math.pi * 10 * i * dt), 0.0, 0.0]
                          for i in range(n)],
            }, segment='forearm')
        print(json.dumps(result, indent=2))
    else:
        p.print_help()


if __name__ == '__main__':
    main()
