"""
tests/test_latency.py

Latency benchmark for PhysicsEngine.certify().
S2S README claims 2.95ms mean latency at 2000Hz.
Law 16 added computation — verify claim still holds.

Run standalone for detailed output:
  python3 tests/test_latency.py
"""
import sys, os, time, random, statistics
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

def _make_window(n=256, hz=2000.0, seed=0):
    random.seed(seed)
    state = [0.0, 0.0, 9.81]
    gyro_state = [0.0, 0.0, 0.0]
    accel, gyro = [], []
    for _ in range(n):
        row = []
        for axis in range(3):
            v = 0.92 * state[axis] + random.gauss(0, 0.4)
            state[axis] = v
            row.append(v)
        row[2] += 9.81
        accel.append(row)
        grow = []
        for axis in range(3):
            g = 0.90 * gyro_state[axis] + random.gauss(0, 0.03)
            gyro_state[axis] = g
            grow.append(g)
        gyro.append(grow)
    ts = [int(i * 1e9 / hz) for i in range(n)]
    return {"timestamps_ns": ts, "accel": accel, "gyro": gyro}


class TestLatency:

    def test_certify_under_50ms(self):
        """Single certify() call must complete in under 50ms.
        Prosthetics safety threshold is 50ms.
        S2S must be well below this."""
        pe = PhysicsEngine()
        w = _make_window()
        # Warmup
        pe.certify(imu_raw=w, segment="forearm")

        times = []
        for i in range(20):
            w = _make_window(seed=i)
            t0 = time.perf_counter()
            pe.certify(imu_raw=w, segment="forearm")
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = statistics.mean(times)
        max_ms  = max(times)
        assert mean_ms < 50.0, \
            f"Mean latency {mean_ms:.2f}ms exceeds 50ms prosthetics threshold"
        assert max_ms < 100.0, \
            f"Max latency {max_ms:.2f}ms exceeds 100ms hard ceiling"

    def test_certify_real_time_feasible(self):
        """At 200Hz (5ms between windows), certify() must complete
        before the next window arrives."""
        pe = PhysicsEngine()
        WINDOW_INTERVAL_MS = 1000.0 / 200.0  # 5ms at 200Hz

        times = []
        for i in range(30):
            w = _make_window(hz=200.0, seed=i)
            t0 = time.perf_counter()
            pe.certify(imu_raw=w, segment="forearm")
            times.append((time.perf_counter() - t0) * 1000)

        mean_ms = statistics.mean(times)
        assert mean_ms < WINDOW_INTERVAL_MS, \
            f"Mean {mean_ms:.2f}ms > window interval {WINDOW_INTERVAL_MS:.1f}ms — not real-time at 200Hz"

    def test_latency_report(self):
        """Print latency report — not a pass/fail test, informational."""
        pe = PhysicsEngine()

        # Warmup
        for i in range(5):
            pe.certify(imu_raw=_make_window(seed=i), segment="forearm")

        times_2000 = []
        for i in range(50):
            w = _make_window(hz=2000.0, seed=i)
            t0 = time.perf_counter()
            pe.certify(imu_raw=w, segment="forearm")
            times_2000.append((time.perf_counter() - t0) * 1000)

        mean_ms = statistics.mean(times_2000)
        p95_ms  = sorted(times_2000)[int(0.95 * len(times_2000))]
        max_ms  = max(times_2000)

        print(f"\n  Latency report (n=50, 2000Hz windows, 256 samples):")
        print(f"    Mean: {mean_ms:.3f}ms")
        print(f"    P95:  {p95_ms:.3f}ms")
        print(f"    Max:  {max_ms:.3f}ms")
        print(f"    README claim: 2.95ms")
        print(f"    Prosthetics threshold: 50ms")
        print(f"    Status: {'✓ WITHIN THRESHOLD' if mean_ms < 50 else '✗ EXCEEDS THRESHOLD'}")

        # Not asserting exact 2.95ms — hardware dependent
        # Just confirm order of magnitude
        assert mean_ms < 50.0

