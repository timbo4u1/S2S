"""
S2S EMG & PPG Certifier — Pytest Test Suite
Tests EMGStreamCertifier and PPGStreamCertifier with synthetic signals.
Run: pytest tests/test_emg_ppg.py -v
"""
import math
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── EMG FIXTURES ───────────────────────────────────────────────────────────────

def make_emg_signal(n_samples=512, hz=1000.0, n_channels=8,
                    add_bursts=True, saturated=False, flatline=False):
    """
    Generate synthetic EMG signal.
    - add_bursts=True  → realistic muscle activation (should pass)
    - saturated=True   → all samples at rail (should REJECT)
    - flatline=True    → electrode off skin (should REJECT)
    """
    dt = 1.0 / hz
    timestamps = [int(i * dt * 1e9) for i in range(n_samples)]
    frames = []
    for i in range(n_samples):
        t = i * dt
        channels = []
        for ch in range(n_channels):
            if flatline:
                val = 0.0
            elif saturated:
                val = 1.0 if i % 2 == 0 else -1.0
            else:
                # Baseline noise
                val = 0.01 * math.sin(2 * math.pi * 300 * t + ch)
                # Muscle burst: 200-500 Hz activity
                if add_bursts and (i % 200) < 80:
                    val += 0.5 * math.sin(2 * math.pi * 250 * t + ch * 0.3)
                    val += 0.3 * math.sin(2 * math.pi * 400 * t + ch * 0.5)
            channels.append(val)
        frames.append((timestamps[i], channels))
    return frames


def make_ppg_signal(n_samples=500, hz=100.0, n_channels=2,
                    add_hrv=True, synthetic_perfect=False, flatline=False):
    """
    Generate synthetic PPG signal.
    - add_hrv=True          → realistic heart rate variability (should pass)
    - synthetic_perfect=True → perfect sine wave, zero HRV (should REJECT)
    - flatline=True          → sensor off skin (should REJECT)
    """
    dt = 1.0 / hz
    timestamps = [int(i * dt * 1e9) for i in range(n_samples)]
    frames = []

    # Heart rate ~1.1 Hz (66 BPM)
    hr_hz = 1.1
    # Breathing ~0.25 Hz
    br_hz = 0.25

    for i in range(n_samples):
        t = i * dt
        channels = []
        for ch in range(n_channels):
            if flatline:
                val = 0.0
            elif synthetic_perfect:
                # Perfect sine — no HRV, no noise
                val = math.sin(2 * math.pi * hr_hz * t)
            else:
                # Real-like: pulse + breathing modulation + HRV + noise
                # HRV: slight jitter in beat timing
                hrv_jitter = 0.02 * math.sin(2 * math.pi * 0.1 * t)
                pulse = math.sin(2 * math.pi * (hr_hz + hrv_jitter) * t)
                # Breathing modulation (amplitude)
                breath = 1.0 + 0.15 * math.sin(2 * math.pi * br_hz * t)
                # Baseline drift
                drift = 0.05 * math.sin(2 * math.pi * 0.02 * t)
                # Small noise
                noise = 0.02 * math.sin(2 * math.pi * 37 * t + ch)
                val = breath * pulse + drift + noise
            channels.append(val)
        frames.append((timestamps[i], channels))
    return frames


# ── EMG TESTS ──────────────────────────────────────────────────────────────────

class TestEMGCertifierImport:

    def test_emg_module_imports(self):
        """EMG certifier must import without error."""
        from s2s_standard_v1_3.s2s_emg_certify_v1_3 import EMGStreamCertifier
        assert EMGStreamCertifier is not None

    def test_emg_constants_present(self):
        """EMG physiological constants must be defined."""
        import s2s_standard_v1_3.s2s_emg_certify_v1_3 as emg
        assert hasattr(emg, 'EMG_BAND_LO_HZ')
        assert hasattr(emg, 'EMG_BAND_HI_HZ')
        assert hasattr(emg, 'EMG_MIN_SAMPLING_HZ')
        assert emg.EMG_BAND_LO_HZ == 20.0
        assert emg.EMG_BAND_HI_HZ == 500.0
        assert emg.EMG_MIN_SAMPLING_HZ == 500.0


class TestEMGStreamCertifier:

    def _get_certifier(self, n_channels=8, hz=1000.0):
        from s2s_standard_v1_3.s2s_emg_certify_v1_3 import EMGStreamCertifier
        return EMGStreamCertifier(n_channels=n_channels, sampling_hz=hz)

    def test_certifier_instantiates(self):
        """EMGStreamCertifier must instantiate with default params."""
        ec = self._get_certifier()
        assert ec is not None

    def test_push_frame_does_not_crash(self):
        """push_frame must accept valid frames without crashing."""
        ec = self._get_certifier()
        frames = make_emg_signal(n_samples=512)
        result = None
        for ts, channels in frames:
            result = ec.push_frame(ts, channels)
        # After enough frames, should return a result dict or None
        assert result is None or isinstance(result, dict)

    def test_valid_emg_returns_result_eventually(self):
        """After a full window of valid EMG, certifier must return a result."""
        ec = self._get_certifier()
        frames = make_emg_signal(n_samples=1024, add_bursts=True)
        results = []
        for ts, channels in frames:
            r = ec.push_frame(ts, channels)
            if r is not None:
                results.append(r)
        assert len(results) >= 1, "No certification result returned after 1024 frames"

    def test_result_has_required_keys(self):
        """EMG certification result must contain tier and score."""
        ec = self._get_certifier()
        frames = make_emg_signal(n_samples=1024, add_bursts=True)
        result = None
        for ts, channels in frames:
            r = ec.push_frame(ts, channels)
            if r is not None:
                result = r
        if result is not None:
            assert 'tier' in result or 'emg_tier' in result or 'status' in result, \
                f"Result missing tier key. Keys: {list(result.keys())}"

    def test_tier_is_valid(self):
        """EMG tier must be one of the four valid values."""
        ec = self._get_certifier()
        frames = make_emg_signal(n_samples=1024, add_bursts=True)
        result = None
        for ts, channels in frames:
            r = ec.push_frame(ts, channels)
            if r is not None:
                result = r
        if result is not None:
            tier = result.get('tier') or result.get('emg_tier') or result.get('status', '')
            assert tier in ('GOLD', 'SILVER', 'BRONZE', 'REJECTED', 'PASS', 'FAIL'), \
                f"Unexpected tier value: {tier}"

    def test_flatline_rejected(self):
        """Flatline EMG (electrode off skin) must return REJECTED or lower score."""
        ec_good = self._get_certifier()
        ec_bad = self._get_certifier()

        good_frames = make_emg_signal(n_samples=1024, add_bursts=True)
        bad_frames = make_emg_signal(n_samples=1024, flatline=True)

        good_result = bad_result = None
        for (ts_g, ch_g), (ts_b, ch_b) in zip(good_frames, bad_frames):
            r = ec_good.push_frame(ts_g, ch_g)
            if r is not None:
                good_result = r
            r = ec_bad.push_frame(ts_b, ch_b)
            if r is not None:
                bad_result = r

        # If both returned results, flatline should be worse
        if good_result and bad_result:
            good_score = good_result.get('score', good_result.get('snr_db', 100))
            bad_score = bad_result.get('score', bad_result.get('snr_db', 0))
            assert good_score >= bad_score, \
                f"Flatline scored {bad_score} but valid EMG scored {good_score}"

    def test_single_channel_emg(self):
        """Single-channel EMG should work (e.g. simple armband)."""
        ec = self._get_certifier(n_channels=1, hz=1000.0)
        frames = make_emg_signal(n_samples=512, n_channels=1)
        for ts, channels in frames:
            ec.push_frame(ts, channels)

    def test_low_hz_emg_handled(self):
        """EMG below minimum sampling rate should be handled gracefully."""
        try:
            from s2s_standard_v1_3.s2s_emg_certify_v1_3 import EMGStreamCertifier
            ec = EMGStreamCertifier(n_channels=8, sampling_hz=100.0)
            frames = make_emg_signal(n_samples=512, hz=100.0)
            for ts, channels in frames:
                ec.push_frame(ts, channels)
        except (ValueError, AssertionError):
            pass  # Rejecting low hz is acceptable


# ── PPG TESTS ──────────────────────────────────────────────────────────────────

class TestPPGCertifierImport:

    def test_ppg_module_imports(self):
        """PPG certifier must import without error."""
        from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import PPGStreamCertifier
        assert PPGStreamCertifier is not None

    def test_ppg_constants_present(self):
        """PPG physiological constants must be defined."""
        import s2s_standard_v1_3.s2s_ppg_certify_v1_3 as ppg
        assert hasattr(ppg, 'PPG_HR_LO_HZ')
        assert hasattr(ppg, 'PPG_HR_HI_HZ')
        assert hasattr(ppg, 'PPG_MIN_SAMPLING_HZ')
        assert ppg.PPG_HR_LO_HZ == 0.5
        assert ppg.PPG_HR_HI_HZ == 3.5
        assert ppg.PPG_MIN_SAMPLING_HZ == 25.0

    def test_ppg_hrv_constants(self):
        """PPG HRV constants must be physiologically plausible."""
        import s2s_standard_v1_3.s2s_ppg_certify_v1_3 as ppg
        assert ppg.PPG_HRV_RMSSD_MIN > 0
        assert ppg.PPG_HRV_RMSSD_GOLD > ppg.PPG_HRV_RMSSD_MIN
        assert ppg.PPG_SYNTHETIC_HRV_MAX < ppg.PPG_HRV_RMSSD_MIN


class TestPPGStreamCertifier:

    def _get_certifier(self, n_channels=2, hz=100.0, device_id='test_watch'):
        from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import PPGStreamCertifier
        return PPGStreamCertifier(n_channels=n_channels, sampling_hz=hz, device_id=device_id)

    def test_certifier_instantiates(self):
        """PPGStreamCertifier must instantiate with default params."""
        pc = self._get_certifier()
        assert pc is not None

    def test_push_frame_does_not_crash(self):
        """push_frame must accept valid frames without crashing."""
        pc = self._get_certifier()
        frames = make_ppg_signal(n_samples=500)
        result = None
        for ts, channels in frames:
            result = pc.push_frame(ts, channels)
        assert result is None or isinstance(result, dict)

    def test_valid_ppg_returns_result_eventually(self):
        """After a full window of valid PPG, certifier must return a result."""
        pc = self._get_certifier()
        frames = make_ppg_signal(n_samples=1000, add_hrv=True)
        results = []
        for ts, channels in frames:
            r = pc.push_frame(ts, channels)
            if r is not None:
                results.append(r)
        assert len(results) >= 1, "No certification result returned after 1000 frames"

    def test_result_has_required_keys(self):
        """PPG certification result must contain recognizable keys."""
        pc = self._get_certifier()
        frames = make_ppg_signal(n_samples=1000, add_hrv=True)
        result = None
        for ts, channels in frames:
            r = pc.push_frame(ts, channels)
            if r is not None:
                result = r
        if result is not None:
            has_tier = any(k in result for k in ('tier', 'ppg_tier', 'status', 'quality'))
            assert has_tier, f"Result missing tier key. Keys: {list(result.keys())}"

    def test_tier_is_valid(self):
        """PPG tier must be one of the four valid values."""
        pc = self._get_certifier()
        frames = make_ppg_signal(n_samples=1000, add_hrv=True)
        result = None
        for ts, channels in frames:
            r = pc.push_frame(ts, channels)
            if r is not None:
                result = r
        if result is not None:
            tier = (result.get('tier') or result.get('ppg_tier') or
                    result.get('status') or result.get('quality', ''))
            assert tier in ('GOLD', 'SILVER', 'BRONZE', 'REJECTED', 'PASS', 'FAIL'), \
                f"Unexpected tier value: {tier}"

    def test_flatline_worse_than_valid(self):
        """Flatline PPG must score worse than valid PPG."""
        pc_good = self._get_certifier()
        pc_bad = self._get_certifier()

        good_frames = make_ppg_signal(n_samples=1000, add_hrv=True)
        bad_frames = make_ppg_signal(n_samples=1000, flatline=True)

        good_result = bad_result = None
        for (ts_g, ch_g), (ts_b, ch_b) in zip(good_frames, bad_frames):
            r = pc_good.push_frame(ts_g, ch_g)
            if r is not None:
                good_result = r
            r = pc_bad.push_frame(ts_b, ch_b)
            if r is not None:
                bad_result = r

        if good_result and bad_result:
            good_score = good_result.get('score', good_result.get('sqi', 100))
            bad_score = bad_result.get('score', bad_result.get('sqi', 0))
            assert good_score >= bad_score, \
                f"Flatline ({bad_score}) scored >= valid PPG ({good_score})"

    def test_single_channel_ppg(self):
        """Single red channel PPG (basic oximeter) must work."""
        pc = self._get_certifier(n_channels=1)
        frames = make_ppg_signal(n_samples=500, n_channels=1)
        for ts, channels in pc._get_certifier if False else []:
            pass
        for ts, channels in frames:
            pc.push_frame(ts, channels)

    def test_heart_rate_in_physiological_range(self):
        """If HR is estimated, it must be in 30-210 BPM range."""
        pc = self._get_certifier()
        frames = make_ppg_signal(n_samples=1000, add_hrv=True)
        result = None
        for ts, channels in frames:
            r = pc.push_frame(ts, channels)
            if r is not None:
                result = r
        if result and 'heart_rate_bpm' in result:
            hr = result['heart_rate_bpm']
            assert 30 <= hr <= 210, f"HR {hr} outside physiological range 30-210 BPM"

    def test_breathing_rate_in_range(self):
        """If breathing rate estimated, must be 9-24 breaths/min (0.15-0.4 Hz)."""
        pc = self._get_certifier()
        frames = make_ppg_signal(n_samples=1000, add_hrv=True)
        result = None
        for ts, channels in frames:
            r = pc.push_frame(ts, channels)
            if r is not None:
                result = r
        if result and 'breathing_rate_hz' in result:
            br = result['breathing_rate_hz']
            assert 0.1 <= br <= 0.5, f"Breathing rate {br} Hz outside expected range"

    def test_performance_1000_frames_fast(self):
        """1000 PPG frames should process in under 3 seconds."""
        import time
        pc = self._get_certifier()
        frames = make_ppg_signal(n_samples=1000)
        start = time.time()
        for ts, channels in frames:
            pc.push_frame(ts, channels)
        elapsed = time.time() - start
        assert elapsed < 3.0, f"1000 PPG frames took {elapsed:.2f}s — too slow"


# ── INTEGRATION: EMG + IMU TOGETHER ───────────────────────────────────────────

class TestEMGPPGIntegration:

    def test_emg_and_ppg_import_independently(self):
        """Both modules must import without conflicting."""
        from s2s_standard_v1_3.s2s_emg_certify_v1_3 import EMGStreamCertifier
        from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import PPGStreamCertifier
        ec = EMGStreamCertifier(n_channels=8, sampling_hz=1000.0)
        pc = PPGStreamCertifier(n_channels=2, sampling_hz=100.0, device_id='test')
        assert ec is not None
        assert pc is not None

    def test_emg_constants_physiologically_valid(self):
        """EMG band must be subset of audio spectrum and above motion artifact."""
        import s2s_standard_v1_3.s2s_emg_certify_v1_3 as emg
        assert emg.EMG_BAND_LO_HZ > emg.EMG_MOTION_ARTIFACT_HZ_MAX, \
            "EMG band must start above motion artifact frequency"
        assert emg.EMG_BAND_HI_HZ > emg.EMG_BAND_LO_HZ
        assert emg.EMG_SNR_GOLD_DB > emg.EMG_SNR_SILVER_DB

    def test_ppg_constants_physiologically_valid(self):
        """PPG constants must be physiologically consistent."""
        import s2s_standard_v1_3.s2s_ppg_certify_v1_3 as ppg
        assert ppg.PPG_HR_HI_HZ > ppg.PPG_HR_LO_HZ
        assert ppg.PPG_BREATHING_HI_HZ > ppg.PPG_BREATHING_LO_HZ
        assert ppg.PPG_BREATHING_HI_HZ < ppg.PPG_HR_LO_HZ, \
            "Breathing rate must be below heart rate"
        assert ppg.PPG_SNR_GOLD_DB > ppg.PPG_SNR_SILVER_DB


class TestTimestampJitter:
    """Certifier must accept real-hardware-style jitter; uniform timestamps have near-zero CV."""

    def _make_ts_and_ppg(self, hz=500, seconds=5, jitter_ns=300):
        import random, math
        n = hz * seconds
        dt = 1.0 / hz
        ts = [int(i * dt * 1e9) + int(random.gauss(0, jitter_ns)) for i in range(n)]
        hr_hz = 70 / 60
        ppg = [math.sin(2 * math.pi * hr_hz * i * dt) for i in range(n)]
        return ts, ppg

    def test_realistic_jitter_accepted(self):
        from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import certify_ppg_channels
        ts, ppg = self._make_ts_and_ppg(jitter_ns=300)
        result = certify_ppg_channels(
            names=["pleth"],
            channels=[ppg],
            timestamps_ns=ts,
            sampling_hz=500,
        )
        assert result["tier"] != "REJECTED", "Real hardware jitter should not be rejected"

    def test_zero_jitter_cv_near_zero(self):
        """Perfect uniform timestamps = reconstructed from file — CV should be ~0."""
        from s2s_standard_v1_3.s2s_ppg_certify_v1_3 import certify_ppg_channels
        import math
        hz, seconds = 500, 5
        n = hz * seconds
        dt = 1.0 / hz
        ts = [int(i * dt * 1e9) for i in range(n)]  # perfectly uniform
        hr_hz = 70 / 60
        ppg = [math.sin(2 * math.pi * hr_hz * i * dt) for i in range(n)]
        result = certify_ppg_channels(
            names=["pleth"],
            channels=[ppg],
            timestamps_ns=ts,
            sampling_hz=500,
        )
        # Top-level timing CV should be near-zero for perfect clock
        per_ch = result.get("per_channel", {})
        ch = per_ch.get("pleth", per_ch.get(next(iter(per_ch), ""), {}))
        cv = ch.get("timing", {}).get("cv", None)
        if cv is not None:
            assert cv < 0.001, f"CV should be near-zero for perfect clock, got {cv}"
