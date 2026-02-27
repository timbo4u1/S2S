#!/usr/bin/env python3
"""
constants.py (s2s v1.3) - canonical constants and TLV registry for SCAN2SELL (S2S)

Changelog v1.2 -> v1.3:
- Added new sensor TLV types: EMG, THERMAL, LIDAR_SCALAR, FUSION_RESULT
- Added stream mode constants: STREAM_WINDOW sizes, frame rate defaults
- Added sensor profile registry for multi-sensor fusion
"""
from enum import IntEnum

# Magic + version
MAGIC = b"S2S\0"
VERSION_MAJOR = 1
VERSION_MINOR = 3

# Header layout (unchanged from v1.2):
# 0..3   4s   MAGIC
# 4..7   uint32 VERSION (major<<16 | minor)
# 8..15  uint64 TIMESTAMP_NS
# 16..19 uint32 PAYLOAD_LEN (bytes)
# 20..23 uint32 HEADER_CRC (CRC32 of bytes 0..19)
# 24..27 uint32 PAYLOAD_CRC (CRC32 of payload)
HEADER_CORE_FMT = "<4sIQI"  # 4s, uint32, uint64, uint32 -> 20 bytes
HEADER_CORE_LEN = 20
HEADER_TOTAL_LEN = 28  # core + two CRCs

# ---------------------------------------------------------------------------
# TLV type registry (uint16 values)
# ---------------------------------------------------------------------------
class TLVType(IntEnum):
    # Core (v1.0+)
    META_JSON        = 0x0001  # utf-8 json metadata blob
    TIMESTAMPS_NS    = 0x0002  # uint64 array (nanoseconds)
    SENSOR_DATA      = 0x0003  # float64 array (per-sensor, generic)
    SENSOR_STATS     = 0x0004  # sensor stats summary (json)
    SIGNATURE        = 0x00FF  # signature blob (Ed25519, varlen)

    # IMU family (v1.0+)
    ACCEL            = 0x0010  # float64 xyz accelerometer m/s²
    GYRO             = 0x0011  # float64 xyz gyroscope rad/s
    MAG              = 0x0012  # float64 xyz magnetometer T
    POSE             = 0x0013  # float64 roll/pitch/yaw radians

    # Environmental (v1.2+)
    TEMPERATURE      = 0x0101  # float32 array °C
    PRESSURE         = 0x0102  # float32 array Pa
    HUMIDITY         = 0x0103  # float32 array %RH

    # Extended sensors (v1.3 NEW)
    EMG              = 0x0200  # float32 array — electromyography (µV)
    THERMAL_FRAME    = 0x0201  # float32 array — thermal camera pixels (°C flat)
    LIDAR_SCALAR     = 0x0202  # float32 array — 1D lidar distance samples (m)
    LIDAR_POINTCLOUD = 0x0203  # float32 xyz triples — 3D point cloud
    HAPTIC_FORCE     = 0x0204  # float32 array — tactile/force sensor (N)
    PPG              = 0x0205  # float32 array — photoplethysmography (raw ADU)

    # Fusion (v1.3 NEW)
    FUSION_RESULT    = 0x0300  # json blob — multi-sensor fusion certificate
    STREAM_FRAME     = 0x0301  # json blob — single streaming window result

# Convenience int aliases
TLV_META_JSON        = int(TLVType.META_JSON)
TLV_TIMESTAMPS_NS    = int(TLVType.TIMESTAMPS_NS)
TLV_SENSOR_DATA      = int(TLVType.SENSOR_DATA)
TLV_SENSOR_STATS     = int(TLVType.SENSOR_STATS)
TLV_SIGNATURE        = int(TLVType.SIGNATURE)
TLV_EMG              = int(TLVType.EMG)
TLV_THERMAL_FRAME    = int(TLVType.THERMAL_FRAME)
TLV_LIDAR_SCALAR     = int(TLVType.LIDAR_SCALAR)
TLV_HAPTIC_FORCE     = int(TLVType.HAPTIC_FORCE)
TLV_PPG              = int(TLVType.PPG)
TLV_FUSION_RESULT    = int(TLVType.FUSION_RESULT)
TLV_STREAM_FRAME     = int(TLVType.STREAM_FRAME)

# ---------------------------------------------------------------------------
# Quality tiers
# ---------------------------------------------------------------------------
TIER_REJECT = "REJECTED"
TIER_BRONZE = "BRONZE"
TIER_SILVER = "SILVER"
TIER_GOLD   = "GOLD"

# ---------------------------------------------------------------------------
# v1.2 / v1.3 sample & jitter thresholds (shared defaults)
# ---------------------------------------------------------------------------
V1_2_SAMPLE_COUNT_GOLD   = 10_000
V1_2_SAMPLE_COUNT_SILVER = 1_000
V1_2_CV_GOLD             = 0.01   # CV < 1% → GOLD
V1_2_CV_SILVER           = 0.05
V1_2_CV_BRONZE           = 0.10
V1_2_REMOVED_PCT_REJECT  = 50.0   # >50% rows removed → REJECT

# ---------------------------------------------------------------------------
# Human physiological tremor detection (8–12 Hz)
# ---------------------------------------------------------------------------
HUMAN_TREMOR_LOW_HZ              = 8.0
HUMAN_TREMOR_HIGH_HZ             = 12.0
MIN_HUMAN_TREMOR_ENERGY_FRACTION = 0.02   # conservative default
MIN_HUMAN_SAMPLING_HZ            = 40.0   # Nyquist guard

# ---------------------------------------------------------------------------
# Streaming / windowing constants (v1.3 NEW)
# ---------------------------------------------------------------------------
# Default sliding window size (samples). At 240Hz this is ~1 second.
STREAM_WINDOW_DEFAULT    = 256
# Minimum samples before any certificate can be issued
STREAM_WINDOW_MIN        = 64
# Step size: how many new samples trigger a re-evaluation (overlap = window - step)
STREAM_STEP_DEFAULT      = 32
# Maximum age (ns) of a frame before it is considered stale and evicted
STREAM_MAX_FRAME_AGE_NS  = 5_000_000_000  # 5 seconds

# ---------------------------------------------------------------------------
# Sensor profile registry (v1.3 NEW)
# Maps a profile name to expected channel names + TLV type + expected Hz range
# Used by the fusion certifier to validate multi-sensor coherence
# ---------------------------------------------------------------------------
SENSOR_PROFILES = {
    "imu_9dof": {
        "channels": ["accel_x", "accel_y", "accel_z",
                     "gyro_x",  "gyro_y",  "gyro_z",
                     "mag_x",   "mag_y",   "mag_z"],
        "tlv": TLVType.SENSOR_DATA,
        "hz_min": 40.0,
        "hz_max": 1000.0,
        "tremor_check": True,
    },
    "imu_6dof": {
        "channels": ["accel_x", "accel_y", "accel_z",
                     "gyro_x",  "gyro_y",  "gyro_z"],
        "tlv": TLVType.SENSOR_DATA,
        "hz_min": 40.0,
        "hz_max": 1000.0,
        "tremor_check": True,
    },
    "emg_8ch": {
        "channels": [f"emg_{i}" for i in range(8)],
        "tlv": TLVType.EMG,
        "hz_min": 500.0,
        "hz_max": 4000.0,
        "tremor_check": False,   # EMG uses its own biological signal checks
        "bio_freq_lo": 20.0,     # EMG band 20–500 Hz
        "bio_freq_hi": 500.0,
    },
    "lidar_1d": {
        "channels": ["distance_m"],
        "tlv": TLVType.LIDAR_SCALAR,
        "hz_min": 10.0,
        "hz_max": 200.0,
        "tremor_check": False,
    },
    "thermal": {
        "channels": ["temp_frame"],
        "tlv": TLVType.THERMAL_FRAME,
        "hz_min": 1.0,
        "hz_max": 60.0,
        "tremor_check": False,
    },
    "ppg": {
        "channels": ["ppg_red", "ppg_ir"],
        "tlv": TLVType.PPG,
        "hz_min": 25.0,
        "hz_max": 500.0,
        "tremor_check": False,
        "bio_freq_lo": 0.5,      # heartbeat + HRV range
        "bio_freq_hi": 4.0,
    },
}

# ---------------------------------------------------------------------------
# Legacy jitter thresholds (v1.0 gatekeeper, kept for backward compat)
# ---------------------------------------------------------------------------
JITTER_STRICT_HIGH   = 2.0
JITTER_STRICT_MED    = 1.5
JITTER_TOLERANT_HIGH = 3.0
JITTER_TOLERANT_MED  = 2.0
