#!/usr/bin/env python3
"""
s2s_ppg_certify_v1_3.py — S2S PPG (Photoplethysmography) Certifier

PPG sensors (wrist, fingertip, ear) measure blood volume pulse via light reflection.
Used in: smartwatches, oximeters, prosthetics, VR controllers, health wearables.

What makes PPG data "real" vs synthetic:
  - Real PPG has a dominant heartbeat frequency (0.5–3.5 Hz = 30–210 BPM)
  - Real PPG has HRV — heart rate variability (beat-to-beat timing is never perfectly regular)
  - Real PPG has breathing modulation (~0.15–0.4 Hz AM envelope on the pulse signal)
  - Real sensors have baseline drift and motion artifacts
  - Synthetic PPG is a perfect sine wave: zero HRV, no breathing modulation, no noise

Certificate tiers:
  GOLD   — clear pulse detected, realistic HRV, breathing modulation, no motion artifact
  SILVER — pulse detected, acceptable HRV, minor noise
  BRONZE — weak pulse, high noise, or signal degraded but still biological
  REJECTED — no pulse, flatline, synthetic, or sensor off skin

Outputs:
  - Estimated heart rate (BPM)
  - HRV metric (RMSSD of beat intervals)
  - Breathing rate estimate (Hz)
  - Signal quality index (SQI)
  - All standard S2S flags

Usage:
  from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import PPGStreamCertifier
  pc = PPGStreamCertifier(n_channels=2, sampling_hz=100.0, device_id='watch_01')
  cert = pc.push_frame(ts_ns, [red_channel, ir_channel])
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import struct
import time
import zlib
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    from .constants import (
        HEADER_CORE_FMT, HEADER_CORE_LEN, HEADER_TOTAL_LEN,
        MAGIC, TLV_META_JSON, TLV_TIMESTAMPS_NS, TLV_SENSOR_DATA,
        TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
    )
except Exception:
    from constants import (
        HEADER_CORE_FMT, HEADER_CORE_LEN, HEADER_TOTAL_LEN,
        MAGIC, TLV_META_JSON, TLV_TIMESTAMPS_NS, TLV_SENSOR_DATA,
        TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
    )

# ---------------------------------------------------------------------------
# PPG physiological constants
# ---------------------------------------------------------------------------
PPG_HR_LO_HZ          = 0.5    # 30 BPM minimum
PPG_HR_HI_HZ          = 3.5    # 210 BPM maximum
PPG_BREATHING_LO_HZ   = 0.15   # 9 breaths/min
PPG_BREATHING_HI_HZ   = 0.40   # 24 breaths/min
PPG_MIN_SAMPLING_HZ   = 25.0   # Nyquist for 3.5 Hz heartbeat × 2 + margin
PPG_HRV_RMSSD_MIN     = 5.0    # ms — minimum realistic HRV (very fit athletes ~20ms)
PPG_HRV_RMSSD_GOLD    = 15.0   # ms — good HRV for GOLD tier
PPG_SNR_GOLD_DB       = 10.0   # signal-to-noise ratio threshold for GOLD
PPG_SNR_SILVER_DB     = 3.0
PPG_CONTACT_VAR_FLOOR = 1.0    # minimum variance — below = sensor lifted off skin
PPG_SYNTHETIC_HRV_MAX = 0.001  # ms — if HRV below this = perfect machine pulse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _variance(data: List[float]) -> float:
    finite = [v for v in data if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    mu = statistics.mean(finite)
    return sum((x - mu) ** 2 for x in finite) / len(finite)


def _freq_projection_normalized(
    signal: List[float], timestamps_ns: List[int], freq_hz: float
) -> float:
    """Normalized sin/cos projection. Returns energy fraction [0,1]."""
    n = min(len(signal), len(timestamps_ns))
    if n < 8:
        return 0.0
    t = [timestamps_ns[i] * 1e-9 for i in range(n)]
    s = [signal[i] for i in range(n)]
    mu = sum(s) / n
    s = [v - mu for v in s]
    sd = sum(s[i] * math.sin(2 * math.pi * freq_hz * t[i]) for i in range(n))
    cd = sum(s[i] * math.cos(2 * math.pi * freq_hz * t[i]) for i in range(n))
    total = sum(v * v for v in s) or 1.0
    return max(0.0, min(1.0, (sd * sd + cd * cd) / (total * n / 2)))


def _band_energy_fraction(
    signal: List[float], timestamps_ns: List[int],
    f_lo: float, f_hi: float, steps: int = 8
) -> float:
    """Mean normalized energy fraction in frequency band."""
    count = steps + 1
    total = sum(
        _freq_projection_normalized(signal, timestamps_ns,
                                    f_lo + (f_hi - f_lo) * i / steps)
        for i in range(count)
    )
    return max(0.0, min(1.0, total / count))


def _find_dominant_freq(
    signal: List[float], timestamps_ns: List[int],
    f_lo: float, f_hi: float, resolution: int = 60
) -> Tuple[float, float]:
    """
    Sweep frequency range and return (dominant_freq_hz, energy_fraction)
    by finding the frequency with the highest projection energy.
    """
    best_freq = f_lo
    best_energy = 0.0
    for i in range(resolution + 1):
        f = f_lo + (f_hi - f_lo) * i / resolution
        e = _freq_projection_normalized(signal, timestamps_ns, f)
        if e > best_energy:
            best_energy = e
            best_freq = f
    return best_freq, best_energy


def _estimate_hrv_rmssd(
    signal: List[float], timestamps_ns: List[int],
    heart_rate_hz: float, sampling_hz: float
) -> float:
    """
    Estimate HRV RMSSD (Root Mean Square of Successive Differences of RR intervals).

    Method: find peaks in the PPG signal, compute beat-to-beat intervals,
    calculate RMSSD. Returns value in milliseconds.

    For real humans: RMSSD typically 20–100ms. Athletes: up to 200ms.
    For synthetic: near 0ms (perfect regularity).
    """
    n = len(signal)
    if n < 20 or sampling_hz <= 0:
        return 0.0

    # Expected samples per beat
    samples_per_beat = max(4, int(sampling_hz / max(heart_rate_hz, 0.1)))

    # Find local maxima (peaks = systolic peaks in PPG)
    peaks: List[int] = []
    half_window = max(2, samples_per_beat // 4)

    mu = statistics.mean(signal)
    sigma = math.sqrt(_variance(signal))
    threshold = mu + 0.2 * sigma  # peaks must be above baseline + 20% std

    for i in range(half_window, n - half_window):
        if signal[i] < threshold:
            continue
        window = signal[max(0, i - half_window): i + half_window + 1]
        if signal[i] >= max(window):
            # Enforce minimum distance between peaks
            if not peaks or (i - peaks[-1]) >= samples_per_beat * 0.5:
                peaks.append(i)

    if len(peaks) < 3:
        return 0.0

    # RR intervals in milliseconds
    rr_intervals_ms = [
        (timestamps_ns[peaks[j+1]] - timestamps_ns[peaks[j]]) / 1e6
        for j in range(len(peaks) - 1)
        if peaks[j+1] < len(timestamps_ns) and peaks[j] < len(timestamps_ns)
    ]

    if len(rr_intervals_ms) < 2:
        return 0.0

    # RMSSD
    successive_diffs = [abs(rr_intervals_ms[j+1] - rr_intervals_ms[j])
                        for j in range(len(rr_intervals_ms) - 1)]
    rmssd = math.sqrt(sum(d * d for d in successive_diffs) / len(successive_diffs))
    return rmssd


def _compute_snr_db(
    signal: List[float], timestamps_ns: List[int],
    hr_hz: float, sampling_hz: float
) -> float:
    """
    Estimate SNR: signal power in HR band vs noise power outside HR band.
    """
    pulse_frac = _freq_projection_normalized(signal, timestamps_ns, hr_hz)
    harmonic   = _freq_projection_normalized(signal, timestamps_ns, hr_hz * 2)
    signal_frac = pulse_frac + 0.5 * harmonic

    noise_frac = max(1.0 - signal_frac - pulse_frac, 1e-9)
    if signal_frac <= 0:
        return 0.0
    return 10.0 * math.log10(signal_frac / noise_frac)


def _compute_jitter(timestamps: List[int]) -> Dict[str, Optional[float]]:
    if len(timestamps) < 2:
        return {"mean_delta_ns": None, "rms_jitter_ns": None, "cv": None}
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:])]
    mu = statistics.mean(deltas)
    sigma = statistics.pstdev(deltas)
    cv = sigma / mu if mu > 0 else None
    return {"mean_delta_ns": mu, "rms_jitter_ns": sigma, "cv": cv}


# ---------------------------------------------------------------------------
# Per-channel PPG analysis
# ---------------------------------------------------------------------------

def analyze_ppg_channel(
    name: str,
    signal: List[float],
    timestamps_ns: List[int],
    sampling_hz: float,
) -> Dict[str, Any]:
    """Full per-channel PPG analysis. Returns structured result dict."""
    flags: List[str] = []
    notes: Dict[str, Any] = {}

    # Contact / skin check
    var = _variance(signal)
    on_skin = var >= PPG_CONTACT_VAR_FLOOR
    if not on_skin:
        flags.append("SENSOR_NOT_ON_SKIN")

    # Dominant heart rate frequency
    hr_hz, hr_energy = _find_dominant_freq(signal, timestamps_ns,
                                            PPG_HR_LO_HZ, PPG_HR_HI_HZ)
    hr_bpm = hr_hz * 60.0

    # Pulse energy fraction (must dominate the signal for it to be real PPG)
    pulse_band_frac = _band_energy_fraction(signal, timestamps_ns,
                                             PPG_HR_LO_HZ, PPG_HR_HI_HZ)
    has_pulse = pulse_band_frac >= 0.05 and on_skin

    if not has_pulse:
        flags.append("NO_PULSE_DETECTED")

    # Breathing modulation
    breathing_frac = _band_energy_fraction(signal, timestamps_ns,
                                            PPG_BREATHING_LO_HZ, PPG_BREATHING_HI_HZ)
    breathing_hz, _ = _find_dominant_freq(signal, timestamps_ns,
                                           PPG_BREATHING_LO_HZ, PPG_BREATHING_HI_HZ,
                                           resolution=20)

    # HRV RMSSD
    rmssd_ms = _estimate_hrv_rmssd(signal, timestamps_ns, hr_hz, sampling_hz)
    is_synthetic_hrv = rmssd_ms > 0 and rmssd_ms < PPG_SYNTHETIC_HRV_MAX and has_pulse

    if is_synthetic_hrv:
        flags.append("SUSPECT_SYNTHETIC")
        notes["synthetic_reason"] = f"HRV RMSSD={rmssd_ms:.4f}ms ≈ 0 (perfect machine pulse)"

    # SNR
    snr_db = _compute_snr_db(signal, timestamps_ns, hr_hz, sampling_hz)

    # Motion artifact (high-frequency noise relative to pulse)
    motion_frac = _band_energy_fraction(signal, timestamps_ns, 5.0, 12.0)
    if motion_frac > 0.3:
        flags.append("MOTION_ARTIFACT_DETECTED")

    # Timing jitter
    jm = _compute_jitter(timestamps_ns)
    cv = jm.get("cv")
    if cv is not None and cv < 1e-8:
        if "SUSPECT_SYNTHETIC" not in flags:
            flags.append("SUSPECT_SYNTHETIC")
        notes["timing_reason"] = "Perfect timing (cv≈0)"

    # Channel quality
    if not on_skin or not has_pulse:
        quality = "UNUSABLE"
    elif ("SUSPECT_SYNTHETIC" in flags):
        quality = TIER_REJECT
    elif snr_db >= PPG_SNR_GOLD_DB and rmssd_ms >= PPG_HRV_RMSSD_GOLD and not flags:
        quality = TIER_GOLD
    elif snr_db >= PPG_SNR_SILVER_DB and rmssd_ms >= PPG_HRV_RMSSD_MIN:
        quality = TIER_SILVER
    elif has_pulse:
        quality = TIER_BRONZE
    else:
        quality = "UNUSABLE"

    return {
        "name":               name,
        "quality":            quality,
        "on_skin":            on_skin,
        "variance":           round(var, 4),
        "heart_rate_hz":      round(hr_hz, 4),
        "heart_rate_bpm":     round(hr_bpm, 1),
        "pulse_band_energy":  round(pulse_band_frac, 4),
        "has_pulse":          has_pulse,
        "hrv_rmssd_ms":       round(rmssd_ms, 3),
        "breathing_hz":       round(breathing_hz, 4),
        "breathing_bpm":      round(breathing_hz * 60, 1),
        "breathing_frac":     round(breathing_frac, 4),
        "snr_db":             round(snr_db, 2),
        "motion_artifact_frac": round(motion_frac, 4),
        "timing_cv":          cv,
        "channel_flags":      flags,
        "notes":              notes,
    }


# ---------------------------------------------------------------------------
# Multi-channel PPG certificate
# ---------------------------------------------------------------------------

def certify_ppg_channels(
    names: List[str],
    channels: List[List[float]],
    timestamps_ns: List[int],
    sampling_hz: float,
    device_id: str = "unknown",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Certify multiple PPG channels and produce a unified certificate."""
    flags: List[str] = []
    notes: Dict[str, Any] = {}

    per_channel: Dict[str, Any] = {}
    qualities: List[str] = []

    for i, ch in enumerate(channels):
        name = names[i] if i < len(names) else f"ppg_{i}"
        result = analyze_ppg_channel(name, ch, timestamps_ns, sampling_hz)
        per_channel[name] = result
        qualities.append(result["quality"])
        flags.extend(result.get("channel_flags", []))

    flags = list(dict.fromkeys(flags))

    # Heart rate consensus across channels
    bpm_values = [
        per_channel[n]["heart_rate_bpm"]
        for n in per_channel
        if per_channel[n]["has_pulse"]
    ]
    rmssd_values = [
        per_channel[n]["hrv_rmssd_ms"]
        for n in per_channel
        if per_channel[n]["has_pulse"]
    ]
    hr_consensus_bpm = statistics.mean(bpm_values) if bpm_values else None

    # Check inter-channel HR agreement (red vs IR should agree within 5 BPM)
    if len(bpm_values) >= 2:
        hr_spread = max(bpm_values) - min(bpm_values)
        if hr_spread > 10.0:
            flags.append("HR_INTER_CHANNEL_DISAGREEMENT")
            notes["hr_spread_bpm"] = round(hr_spread, 1)

    active = sum(1 for q in qualities if q in (TIER_GOLD, TIER_SILVER, TIER_BRONZE))
    total  = len(qualities)

    if active == 0 or not bpm_values:
        tier = TIER_REJECT
        flags.append("NO_VALID_PPG_CHANNELS")
    elif "SUSPECT_SYNTHETIC" in flags:
        tier = TIER_REJECT
    elif all(q == TIER_GOLD for q in qualities if q != "UNUSABLE"):
        tier = TIER_GOLD
    elif sum(q in (TIER_GOLD, TIER_SILVER) for q in qualities) >= max(1, active * 0.7):
        tier = TIER_SILVER
    elif active >= 1:
        tier = TIER_BRONZE
    else:
        tier = TIER_REJECT

    # Vital signs summary (only if confident)
    vitals: Dict[str, Any] = {}
    if hr_consensus_bpm and 30 <= hr_consensus_bpm <= 210:
        vitals["heart_rate_bpm"] = round(hr_consensus_bpm, 1)
    if rmssd_values:
        vitals["hrv_rmssd_ms"] = round(statistics.mean(rmssd_values), 2)
    breathing_vals = [per_channel[n]["breathing_bpm"] for n in per_channel
                      if per_channel[n].get("breathing_frac", 0) > 0.01]
    if breathing_vals:
        vitals["breathing_bpm"] = round(statistics.mean(breathing_vals), 1)

    notes["active_channels"] = active
    notes["total_channels"]  = total

    return {
        "status":          "PASS" if tier != TIER_REJECT else "FAIL",
        "tier":            tier,
        "sensor_type":     "PPG",
        "n_channels":      total,
        "n_active":        active,
        "frame_start_ts_ns": timestamps_ns[0] if timestamps_ns else None,
        "frame_end_ts_ns":   timestamps_ns[-1] if timestamps_ns else None,
        "duration_ms":     (timestamps_ns[-1] - timestamps_ns[0]) / 1e6
                           if len(timestamps_ns) > 1 else 0,
        "sampling_hz":     sampling_hz,
        "vitals":          vitals,
        "per_channel":     per_channel,
        "flags":           flags,
        "notes":           notes,
        "device_id":       device_id,
        "session_id":      session_id,
        "tool":            "s2s_ppg_certify_v1_3",
        "issued_at_ns":    time.time_ns(),
    }


# ---------------------------------------------------------------------------
# PPGStreamCertifier
# ---------------------------------------------------------------------------

class PPGStreamCertifier:
    """
    Real-time PPG certifier.
    push_frame(ts_ns, values) where values = [red, ir] (or any n_channels).
    """

    def __init__(
        self,
        n_channels: int = 2,
        sampling_hz: float = 100.0,
        channel_names: Optional[List[str]] = None,
        window: int = 256,   # 256 samples at 100Hz = 2.56 seconds (covers ~2 heartbeats at 60 BPM)
        step: int = 32,
        device_id: str = "unknown",
        session_id: Optional[str] = None,
    ):
        if sampling_hz < PPG_MIN_SAMPLING_HZ:
            raise ValueError(
                f"PPG sampling_hz must be >= {PPG_MIN_SAMPLING_HZ} Hz. Got {sampling_hz}"
            )
        self.n_channels    = n_channels
        self.sampling_hz   = sampling_hz
        self.channel_names = channel_names or [f"ppg_{i}" for i in range(n_channels)]
        self.window        = window
        self.step          = step
        self.device_id     = device_id
        self.session_id    = session_id

        self._ts_buf:  Deque[int]         = deque(maxlen=window)
        self._ch_bufs: List[Deque[float]] = [deque(maxlen=window) for _ in range(n_channels)]
        self._frames_since_cert = 0
        self._total_frames      = 0
        self._certs_emitted     = 0

    def push_frame(self, ts_ns: int, values: List[float]) -> Optional[Dict[str, Any]]:
        if len(values) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} values, got {len(values)}")

        self._ts_buf.append(ts_ns)
        for i, v in enumerate(values):
            self._ch_bufs[i].append(v)

        self._total_frames      += 1
        self._frames_since_cert += 1

        if len(self._ts_buf) >= self.window and self._frames_since_cert >= self.step:
            self._frames_since_cert = 0
            cert = certify_ppg_channels(
                names         = self.channel_names,
                channels      = [list(buf) for buf in self._ch_bufs],
                timestamps_ns = list(self._ts_buf),
                sampling_hz   = self.sampling_hz,
                device_id     = self.device_id,
                session_id    = self.session_id,
            )
            cert["cert_index"]        = self._certs_emitted
            cert["total_frames_seen"] = self._total_frames
            self._certs_emitted += 1
            return cert

        return None

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "total_frames":  self._total_frames,
            "certs_emitted": self._certs_emitted,
            "buffer_fill":   len(self._ts_buf),
            "window":        self.window,
            "step":          self.step,
            "sampling_hz":   self.sampling_hz,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


def _read_s2s_raw(path: Path):
    data = path.read_bytes()
    if len(data) < HEADER_TOTAL_LEN:
        raise ParseError("FILE_TOO_SMALL")
    header_core = data[0:HEADER_CORE_LEN]
    if (zlib.crc32(header_core) & 0xFFFFFFFF) != int.from_bytes(
            data[HEADER_CORE_LEN:HEADER_CORE_LEN+4], "little"):
        raise ParseError("HEADER_CRC_MISMATCH")
    magic, _, _, payload_len = struct.unpack(HEADER_CORE_FMT, header_core)
    if magic != MAGIC:
        raise ParseError("INVALID_MAGIC")
    payload = data[HEADER_TOTAL_LEN:HEADER_TOTAL_LEN + payload_len]
    pos = 0; meta = {}; timestamps = []; channels = []
    while pos + 6 <= len(payload):
        t = int.from_bytes(payload[pos:pos+2], "little")
        l = int.from_bytes(payload[pos+2:pos+6], "little")
        val = payload[pos+6:pos+6+l]
        if t == int(TLV_META_JSON):
            try: meta = json.loads(val.decode("utf-8"))
            except: pass
        elif t == int(TLV_TIMESTAMPS_NS):
            timestamps = [struct.unpack("<Q", val[i*8:(i+1)*8])[0] for i in range(len(val)//8)]
        elif t == int(TLV_SENSOR_DATA):
            channels.append([struct.unpack("<d", val[i*8:(i+1)*8])[0] for i in range(len(val)//8)])
        pos += 6 + l
    return meta, timestamps, channels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="S2S v1.3 PPG Certifier",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out-json-dir", default="library/ppg_certs")
    p.add_argument("--sampling-hz", type=float, default=100.0)
    p.add_argument("--device-id", default="unknown")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_json_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import glob as _glob
    paths: List[Path] = []
    for token in args.inputs:
        p = Path(token)
        if p.is_file():   paths.append(p)
        elif p.is_dir():  paths.extend(sorted(p.rglob("*.s2s")))
        else:             paths.extend(sorted(Path(x) for x in _glob.glob(token, recursive=True)))

    if not args.quiet:
        print(f"S2S PPG Certifier v1.3 — {len(paths)} file(s) → {out_dir}")

    for path in paths:
        try:
            meta, timestamps, channels = _read_s2s_raw(path)
        except ParseError as e:
            (out_dir / (path.name + ".ppg.json")).write_text(
                json.dumps({"status": "FAIL", "reason": str(e)}, indent=2))
            continue

        names = meta.get("sensor_map") or [f"ppg_{i}" for i in range(len(channels))]
        mean_delta = (
            statistics.mean([b-a for a,b in zip(timestamps, timestamps[1:])])
            if len(timestamps) > 1 else None
        )
        sampling_hz = (1e9 / mean_delta) if mean_delta else args.sampling_hz

        result = certify_ppg_channels(
            names=names, channels=channels,
            timestamps_ns=timestamps, sampling_hz=sampling_hz,
            device_id=args.device_id,
        )
        out_path = out_dir / (path.name + ".ppg.json")
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not args.quiet:
            hr = result.get("vitals", {}).get("heart_rate_bpm", "?")
            print(f"[PPG] {path.name} → {result['tier']} HR={hr}bpm flags={result['flags']}")

    if not args.quiet:
        print("Done.")


if __name__ == "__main__":
    main()
