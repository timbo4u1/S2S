#!/usr/bin/env python3
"""
s2s_emg_certify_v1_3.py — S2S EMG (Electromyography) Certifier

EMG is fundamentally different from IMU:
  - Sampling rates: 500 Hz – 4000 Hz (vs IMU ~240 Hz)
  - Human signal band: 20–500 Hz (muscle activation frequencies)
  - No physiological tremor check (8–12 Hz) — that's for IMU
  - Instead: checks for muscle burst activity, baseline noise floor,
    electrode contact quality, and motion artifact contamination

Certificate tiers (EMG-specific):
  GOLD   — clean signal, confirmed muscle activity bursts, good SNR, no motion artifact
  SILVER — valid signal, mild noise or weak bursts
  BRONZE — usable but noisy or low-activity recording
  REJECTED — electrode off, saturated, pure noise, or synthetic

Usage:
  # File-based (batch):
  python3 -m s2s_standard_v1_3.s2s_emg_certify_v1_3 data.s2s --out-json-dir certs/

  # Python API (streaming):
  from s2s_standard_v1_3.s2s_emg_certify_v1_3 import EMGStreamCertifier
  ec = EMGStreamCertifier(n_channels=8, sampling_hz=1000.0)
  result = ec.push_frame(ts_ns, [ch0, ch1, ..., ch7])
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

try:
    from .constants import (
        HEADER_CORE_FMT, HEADER_CORE_LEN, HEADER_TOTAL_LEN,
        MAGIC, TLV_META_JSON, TLV_TIMESTAMPS_NS, TLV_SENSOR_DATA, TLV_SIGNATURE,
        TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
        STREAM_WINDOW_DEFAULT, STREAM_STEP_DEFAULT,
        V1_2_CV_GOLD, V1_2_CV_SILVER, V1_2_CV_BRONZE,
    )
except Exception:
    from constants import (
        HEADER_CORE_FMT, HEADER_CORE_LEN, HEADER_TOTAL_LEN,
        MAGIC, TLV_META_JSON, TLV_TIMESTAMPS_NS, TLV_SENSOR_DATA, TLV_SIGNATURE,
        TIER_GOLD, TIER_SILVER, TIER_BRONZE, TIER_REJECT,
        STREAM_WINDOW_DEFAULT, STREAM_STEP_DEFAULT,
        V1_2_CV_GOLD, V1_2_CV_SILVER, V1_2_CV_BRONZE,
    )

import struct, zlib

# ---------------------------------------------------------------------------
# EMG-specific constants
# ---------------------------------------------------------------------------
EMG_BAND_LO_HZ        = 20.0     # lower edge of EMG frequency band
EMG_BAND_HI_HZ        = 500.0    # upper edge
EMG_MIN_SAMPLING_HZ   = 500.0    # Nyquist for EMG_BAND_HI
EMG_SNR_GOLD_DB       = 20.0     # signal-to-noise ratio threshold for GOLD (dB)
EMG_SNR_SILVER_DB     = 10.0     # minimum SNR for SILVER
EMG_SATURATION_PCT    = 95.0     # if >X% of samples at rail → SATURATED
EMG_CONTACT_VAR_FLOOR = 1e-4     # minimum variance; below → electrode off skin
EMG_BURST_THRESHOLD   = 3.0      # RMS multiplier above baseline to detect burst
EMG_MIN_BURST_FRAC    = 0.05     # minimum fraction of window that must be in burst
EMG_MOTION_ARTIFACT_HZ_MAX = 10.0  # frequencies below this in EMG = motion artifact

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def _rms(channel: List[float]) -> float:
    finite = [v for v in channel if math.isfinite(v)]
    if not finite:
        return 0.0
    return math.sqrt(sum(v * v for v in finite) / len(finite))


def _variance(channel: List[float]) -> float:
    finite = [v for v in channel if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    mu = statistics.mean(finite)
    return sum((x - mu) ** 2 for x in finite) / len(finite)


def _freq_band_energy_fraction(
    channel: List[float],
    timestamps_ns: List[int],
    f_lo: float,
    f_hi: float,
    steps: int = 8,
) -> float:
    """
    Normalized band energy fraction using sin/cos projections.
    Returns fraction of signal energy in [f_lo, f_hi] band. Range [0, 1].
    Normalization: divide by (n/2) per projection to handle noise correctly.
    """
    n = min(len(channel), len(timestamps_ns))
    if n < 16:
        return 0.0
    t = [timestamps_ns[i] * 1e-9 for i in range(n)]
    s = [channel[i] for i in range(n)]
    mu = sum(s) / n
    s = [v - mu for v in s]
    total_e = sum(v * v for v in s) or 1.0

    band_e = 0.0
    count = steps + 1
    for i in range(count):
        f = f_lo + (f_hi - f_lo) * i / steps
        sd = sum(s[j] * math.sin(2 * math.pi * f * t[j]) for j in range(n))
        cd = sum(s[j] * math.cos(2 * math.pi * f * t[j]) for j in range(n))
        # normalized projection
        band_e += (sd * sd + cd * cd) / (total_e * n / 2)

    return max(0.0, min(1.0, band_e / count))


def _detect_muscle_bursts(
    channel: List[float],
    threshold_multiplier: float = EMG_BURST_THRESHOLD,
    window_size: int = 50,
) -> Tuple[float, int]:
    """
    Detect muscle activation bursts using a sliding RMS envelope.
    Returns (burst_fraction, burst_count) where burst_fraction is the
    proportion of samples classified as active muscle burst.
    """
    n = len(channel)
    if n < window_size * 2:
        return 0.0, 0

    # Compute sliding RMS envelope
    envelope: List[float] = []
    for i in range(n):
        start = max(0, i - window_size // 2)
        end   = min(n, i + window_size // 2)
        seg   = [channel[j] for j in range(start, end) if math.isfinite(channel[j])]
        envelope.append(math.sqrt(sum(v * v for v in seg) / len(seg)) if seg else 0.0)

    # Baseline = median of lower 30% of envelope values (rest state)
    sorted_env = sorted(envelope)
    baseline_end = max(1, int(len(sorted_env) * 0.30))
    baseline_rms = statistics.mean(sorted_env[:baseline_end]) or 1e-9

    threshold = baseline_rms * threshold_multiplier
    in_burst   = False
    burst_count = 0
    burst_samples = 0

    for v in envelope:
        if v >= threshold:
            burst_samples += 1
            if not in_burst:
                burst_count += 1
                in_burst = True
        else:
            in_burst = False

    return burst_samples / n, burst_count


def _check_saturation(channel: List[float], rail_pct: float = EMG_SATURATION_PCT) -> Tuple[bool, float]:
    """
    Check if signal is clipping / saturated.
    Returns (is_saturated, saturation_fraction).
    """
    finite = [v for v in channel if math.isfinite(v)]
    if not finite:
        return True, 1.0
    peak = max(abs(v) for v in finite)
    if peak == 0:
        return False, 0.0
    near_rail = sum(1 for v in finite if abs(v) >= peak * 0.98)
    sat_frac = near_rail / len(finite)
    return sat_frac * 100 >= rail_pct, sat_frac


def _compute_snr_db(channel: List[float], timestamps_ns: List[int], sampling_hz: float) -> float:
    """
    Estimate SNR in dB:
      - Signal power = energy in EMG band (20-500 Hz)
      - Noise power  = energy outside band (0-20 Hz motion artifact + >500 Hz)
    Returns SNR in dB. Returns 0.0 if cannot be computed.
    """
    finite = [v for v in channel if math.isfinite(v)]
    if len(finite) < 32:
        return 0.0

    signal_frac   = _freq_band_energy_fraction(channel, timestamps_ns, EMG_BAND_LO_HZ, EMG_BAND_HI_HZ)
    artifact_frac = _freq_band_energy_fraction(channel, timestamps_ns, 0.5, EMG_MOTION_ARTIFACT_HZ_MAX)

    # Noise power = artifact + residual (1 - signal)
    noise_frac = max(artifact_frac, 1.0 - signal_frac, 1e-9)
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
# Per-channel EMG analysis
# ---------------------------------------------------------------------------

def analyze_emg_channel(
    name: str,
    channel: List[float],
    timestamps_ns: List[int],
    sampling_hz: float,
    snr_gold_db: float = EMG_SNR_GOLD_DB,
    snr_silver_db: float = EMG_SNR_SILVER_DB,
    contact_var_floor: float = EMG_CONTACT_VAR_FLOOR,
    burst_threshold: float = EMG_BURST_THRESHOLD,
    min_burst_frac: float = EMG_MIN_BURST_FRAC,
) -> Dict[str, Any]:
    """
    Full per-channel EMG analysis. Returns a structured result dict.
    """
    flags: List[str] = []

    # Variance / electrode contact
    var = _variance(channel)
    electrode_on = var >= contact_var_floor
    if not electrode_on:
        flags.append("ELECTRODE_OFF_OR_DISCONNECTED")

    # Saturation check
    is_sat, sat_frac = _check_saturation(channel)
    if is_sat:
        flags.append("SIGNAL_SATURATED")

    # Signal RMS
    sig_rms = _rms(channel)

    # Synthetic timing check
    jm = _compute_jitter(timestamps_ns)
    cv = jm.get("cv")
    if cv is not None and cv < 1e-8:
        flags.append("SUSPECT_SYNTHETIC")

    # EMG band energy
    emg_band_frac = _freq_band_energy_fraction(
        channel, timestamps_ns, EMG_BAND_LO_HZ, EMG_BAND_HI_HZ
    )

    # Motion artifact energy (low-frequency contamination)
    motion_artifact_frac = _freq_band_energy_fraction(
        channel, timestamps_ns, 0.5, EMG_MOTION_ARTIFACT_HZ_MAX
    )
    if motion_artifact_frac > 0.3:
        flags.append("MOTION_ARTIFACT_DETECTED")

    # SNR
    snr_db = _compute_snr_db(channel, timestamps_ns, sampling_hz)

    # Muscle burst detection
    burst_frac, burst_count = _detect_muscle_bursts(channel, burst_threshold)
    has_bursts = burst_frac >= min_burst_frac
    if not has_bursts and electrode_on and not is_sat:
        flags.append("NO_MUSCLE_ACTIVITY_DETECTED")

    # Channel quality score
    if not electrode_on or is_sat:
        quality = "UNUSABLE"
    elif snr_db >= snr_gold_db and has_bursts and not flags:
        quality = "GOLD"
    elif snr_db >= snr_silver_db and electrode_on:
        quality = "SILVER"
    else:
        quality = "BRONZE"

    return {
        "name":                name,
        "quality":             quality,
        "variance":            var,
        "electrode_on":        electrode_on,
        "signal_rms_uv":       sig_rms,
        "snr_db":              round(snr_db, 2),
        "emg_band_energy_frac": round(emg_band_frac, 4),
        "motion_artifact_frac": round(motion_artifact_frac, 4),
        "burst_fraction":      round(burst_frac, 4),
        "burst_count":         burst_count,
        "has_muscle_activity": has_bursts,
        "saturation_fraction": round(sat_frac, 4),
        "timing_cv":           cv,
        "channel_flags":       flags,
    }


# ---------------------------------------------------------------------------
# Multi-channel EMG certificate
# ---------------------------------------------------------------------------

def certify_emg_channels(
    names: List[str],
    channels: List[List[float]],
    timestamps_ns: List[int],
    sampling_hz: float,
    device_id: str = "unknown",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full EMG certification on multiple channels.
    Returns a certificate dict matching the S2S schema.
    """
    flags: List[str] = []
    notes: Dict[str, Any] = {}

    per_channel: Dict[str, Any] = {}
    qualities: List[str] = []

    for i, ch in enumerate(channels):
        name = names[i] if i < len(names) else f"emg_{i}"
        result = analyze_emg_channel(
            name=name,
            channel=ch,
            timestamps_ns=timestamps_ns,
            sampling_hz=sampling_hz,
        )
        per_channel[name] = result
        qualities.append(result["quality"])
        flags.extend(result["channel_flags"])

    # De-duplicate flags
    flags = list(dict.fromkeys(flags))

    # Active channel count (electrode on + not saturated)
    active = sum(1 for q in qualities if q in (TIER_GOLD, TIER_SILVER, TIER_BRONZE))
    total  = len(qualities)

    if active == 0:
        tier = TIER_REJECT
        flags.append("ALL_CHANNELS_UNUSABLE")
    elif all(q == TIER_GOLD for q in qualities):
        tier = TIER_GOLD
    elif sum(q in (TIER_GOLD, TIER_SILVER) for q in qualities) >= total * 0.7:
        tier = TIER_SILVER
    elif active >= max(1, total // 2):
        tier = TIER_BRONZE
    else:
        tier = TIER_REJECT

    # Timing check
    jm = _compute_jitter(timestamps_ns)

    # Summary stats
    burst_fracs = [per_channel[n]["burst_fraction"] for n in per_channel]
    snr_values  = [per_channel[n]["snr_db"] for n in per_channel]

    notes["active_channels"] = active
    notes["total_channels"]  = total
    notes["mean_snr_db"]     = round(statistics.mean(snr_values), 2) if snr_values else 0.0
    notes["mean_burst_frac"] = round(statistics.mean(burst_fracs), 4) if burst_fracs else 0.0
    notes["sampling_hz"]     = sampling_hz

    return {
        "status":        "PASS" if tier != TIER_REJECT else "FAIL",
        "tier":          tier,
        "sensor_type":   "EMG",
        "n_channels":    total,
        "n_active":      active,
        "frame_start_ts_ns": timestamps_ns[0] if timestamps_ns else None,
        "frame_end_ts_ns":   timestamps_ns[-1] if timestamps_ns else None,
        "duration_ms":   (timestamps_ns[-1] - timestamps_ns[0]) / 1e6 if len(timestamps_ns) > 1 else 0,
        "sampling_hz":   sampling_hz,
        "timing_metrics": jm,
        "per_channel":   per_channel,
        "flags":         flags,
        "notes":         notes,
        "device_id":     device_id,
        "session_id":    session_id,
        "tool":          "s2s_emg_certify_v1_3",
        "issued_at_ns":  time.time_ns(),
    }


# ---------------------------------------------------------------------------
# EMGStreamCertifier — real-time streaming class
# ---------------------------------------------------------------------------

class EMGStreamCertifier:
    """
    Real-time EMG certifier. Push frames, get certificates.

    Push format: push_frame(ts_ns, values) where values is a list of
    floats, one per EMG channel (in µV or raw ADC units, consistent).
    """

    def __init__(
        self,
        n_channels: int,
        sampling_hz: float = 1000.0,
        channel_names: Optional[List[str]] = None,
        window: int = 512,
        step: int = 64,
        device_id: str = "unknown",
        session_id: Optional[str] = None,
    ):
        if sampling_hz < EMG_MIN_SAMPLING_HZ:
            raise ValueError(
                f"EMG sampling_hz must be >= {EMG_MIN_SAMPLING_HZ} Hz "
                f"(Nyquist for {EMG_BAND_HI_HZ} Hz band). Got {sampling_hz}"
            )
        self.n_channels   = n_channels
        self.sampling_hz  = sampling_hz
        self.channel_names = channel_names or [f"emg_{i}" for i in range(n_channels)]
        self.window       = window
        self.step         = step
        self.device_id    = device_id
        self.session_id   = session_id

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
            cert = certify_emg_channels(
                names       = self.channel_names,
                channels    = [list(buf) for buf in self._ch_bufs],
                timestamps_ns = list(self._ts_buf),
                sampling_hz = self.sampling_hz,
                device_id   = self.device_id,
                session_id  = self.session_id,
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
# File-based certifier (batch mode — reads .s2s files)
# ---------------------------------------------------------------------------

class ParseError(Exception):
    pass


def _read_s2s_raw(path: Path):
    data = path.read_bytes()
    if len(data) < HEADER_TOTAL_LEN:
        raise ParseError("FILE_TOO_SMALL")
    header_core = data[0:HEADER_CORE_LEN]
    header_crc_claimed = int.from_bytes(data[HEADER_CORE_LEN:HEADER_CORE_LEN+4], "little")
    if (zlib.crc32(header_core) & 0xFFFFFFFF) != header_crc_claimed:
        raise ParseError("HEADER_CRC_MISMATCH")
    magic, version_int, creation_ns, payload_len = struct.unpack(HEADER_CORE_FMT, header_core)
    if magic != MAGIC:
        raise ParseError("INVALID_MAGIC")
    payload = data[HEADER_TOTAL_LEN:HEADER_TOTAL_LEN + payload_len]
    if (zlib.crc32(payload) & 0xFFFFFFFF) != int.from_bytes(data[HEADER_CORE_LEN+4:HEADER_CORE_LEN+8], "little"):
        raise ParseError("PAYLOAD_CRC_MISMATCH")
    pos = 0
    meta: Dict[str, Any] = {}
    timestamps: List[int] = []
    channels: List[List[float]] = []
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="S2S v1.3 — EMG Certifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("inputs", nargs="+", help=".s2s files or directories")
    p.add_argument("--out-json-dir", default="library/emg_certs")
    p.add_argument("--sampling-hz", type=float, default=1000.0)
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
        print(f"S2S EMG Certifier v1.3 — {len(paths)} file(s) → {out_dir}")

    for path in paths:
        try:
            meta, timestamps, channels = _read_s2s_raw(path)
        except ParseError as e:
            result = {"status": "FAIL", "reason": str(e)}
            (out_dir / (path.name + ".emg.json")).write_text(json.dumps(result, indent=2))
            continue

        names = meta.get("sensor_map") or [f"emg_{i}" for i in range(len(channels))]
        mean_delta = (
            statistics.mean([b - a for a, b in zip(timestamps, timestamps[1:])])
            if len(timestamps) > 1 else None
        )
        sampling_hz = (1e9 / mean_delta) if mean_delta else args.sampling_hz

        result = certify_emg_channels(
            names=names, channels=channels,
            timestamps_ns=timestamps, sampling_hz=sampling_hz,
            device_id=args.device_id,
        )
        out_path = out_dir / (path.name + ".emg.json")
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"[EMG] {path.name} → {result['tier']} "
                  f"active={result['n_active']}/{result['n_channels']} flags={result['flags']}")

    if not args.quiet:
        print("Done.")


if __name__ == "__main__":
    main()
