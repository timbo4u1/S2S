"""
Negative case proof for BFS biological floor detector.
Tests three synthetic signals that should score below 0.35 (NOT HUMAN).

Signals:
  1. Pure sine wave at 50Hz — robot motion (perfectly periodic, zero biological noise)
  2. White noise — sensor malfunction (no motor control structure)
  3. Step function — bang-bang robot control (instantaneous transitions, not biological)

Reference: all 10 NinaPro DB5 human subjects scored >= 0.35 (HUMAN).
"""

import sys, math, random
sys.path.insert(0, '/Users/timbo/S2S')
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

HZ = 100
WINDOW = 512
STEP = 256
N_WINDOWS = 40  # enough for stable BFS (need >= 4)
SAMPLES = N_WINDOWS * STEP + WINDOW

def make_timestamps(n, hz):
    dt_ns = int(1e9 / hz)
    return [i * dt_ns for i in range(n)]

def run_signal(name, signal_fn):
    """Run certify_session() on a synthetic signal."""
    pe = PhysicsEngine()
    ts_full = make_timestamps(SAMPLES, HZ)
    accel_full = [signal_fn(i, HZ) for i in range(SAMPLES)]

    certified = 0
    for i in range(0, SAMPLES - WINDOW + 1, STEP):
        window_acc = accel_full[i:i+WINDOW]
        window_ts  = ts_full[i:i+WINDOW]
        pe.certify(
            imu_raw={"timestamps_ns": window_ts, "accel": window_acc},
            segment="forearm"
        )
        certified += 1

    r = pe.certify_session()
    bfs   = r.get("bfs")
    score = r.get("biological_diversity_score")
    grade = r.get("biological_grade")
    nw    = r.get("n_windows")
    cv    = r.get("cv")
    hurst = r.get("hurst")

    passed = (grade is not None and grade not in ("HUMAN", "SUPERHUMAN"))
    status = "✅ PASS" if passed else "❌ FAIL"

    print(f"\n{'─'*60}")
    print(f"Signal : {name}")
    print(f"Status : {status}")
    print(f"BFS    : {bfs}  (must be < 0.35)")
    print(f"Score  : {score}/100")
    print(f"Grade  : {grade}  (must NOT be HUMAN)")
    print(f"CV     : {cv}  |  Hurst: {hurst}  |  Windows: {nw}")
    return passed

# ── Signal definitions ────────────────────────────────────────────────────────

def sine_robot(i, hz):
    """Pure 50Hz sine — perfectly periodic robot motion."""
    t = i / hz
    v = math.sin(2 * math.pi * 50 * t)
    return [v, v * 0.5, v * 0.25]

def white_noise(i, hz):
    """White noise — sensor malfunction, no motor control structure."""
    return [random.gauss(0, 1.0) for _ in range(3)]

def step_function(i, hz):
    """Bang-bang robot control — alternates every 0.5s."""
    t = i / hz
    v = 1.0 if int(t * 2) % 2 == 0 else -1.0
    return [v, v * 0.8, 0.0]

# ── Run all three ─────────────────────────────────────────────────────────────

print("=" * 60)
print("S2S BFS NEGATIVE CASE PROOF")
print("Hypothesis: synthetic signals score BFS < 0.35 (NOT HUMAN)")
print("Human reference: NinaPro DB5 n=10, BFS range 0.37–0.68")
print("=" * 60)

results = [
    run_signal("1. Pure sine 50Hz (robot motion)",    sine_robot),
    run_signal("2. White noise (sensor malfunction)", white_noise),
    run_signal("3. Step function (bang-bang control)", step_function),
]

print(f"\n{'='*60}")
passed = sum(results)
print(f"RESULT: {passed}/3 synthetic signals correctly scored NOT HUMAN")
if passed == 3:
    print("✅ PROOF COMPLETE — BFS floor detector validated on negative cases")
else:
    print("❌ PARTIAL — review failing signals before committing")
print("="*60)
