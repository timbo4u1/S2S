"""
S2S Adversarial Stress Test
Signals designed to fool the biological origin detector.
Goal: find what the pipeline CANNOT catch — honest gap analysis.

Each signal is crafted to pass specific gates:
  Gate 1 — Physics (jerk bounds, resonance)
  Gate 2 — Hurst floor (H > 0.70)
  Gate 3 — BFS floor (BFS > 0.35)
  Gate 4 — biological_grade == HUMAN
"""

import sys, os, math, random
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

HZ      = 100
WINDOW  = 512
STEP    = 256
N_WIN   = 60    # enough for stable Hurst
SAMPLES = N_WIN * STEP + WINDOW

G  = "\033[92m"
R  = "\033[91m"
Y  = "\033[93m"
W  = "\033[97m"
D  = "\033[2m"
X  = "\033[0m"

def ts(n, hz):
    return [i * int(1e9/hz) for i in range(n)]

def run_signal(name, description, signal_fn, hz=HZ):
    """Run a signal through all pipeline gates and report which pass/fail."""
    pe = PhysicsEngine()
    scores, tiers = [], []
    samples = N_WIN * STEP + WINDOW
    accel_full = [signal_fn(i, hz) for i in range(samples)]
    ts_full    = ts(samples, hz)

    for i in range(0, samples - WINDOW + 1, STEP):
        r = pe.certify(
            imu_raw={"timestamps_ns": ts_full[i:i+WINDOW], "accel": accel_full[i:i+WINDOW]},
            segment="forearm"
        )
        scores.append(r["physical_law_score"])
        tiers.append(r["tier"])

    session = pe.certify_session()

    from collections import Counter
    tc      = Counter(tiers)
    n       = len(tiers)
    avg_sc  = sum(scores)/n if n else 0
    hurst   = session.get("hurst") or 0
    bfs     = session.get("bfs") or 0
    grade   = session.get("biological_grade", "UNKNOWN")
    rec     = session.get("recommendation", "?")

    rej_pct = 100 * tc.get("REJECTED", 0) // n if n else 0

    g1_pass = rej_pct < 50          # Gate 1: physics — less than 50% rejected
    g2_pass = hurst >= 0.70         # Gate 2: Hurst floor
    g3_pass = bfs >= 0.35           # Gate 3: BFS floor
    g4_pass = grade == "HUMAN"      # Gate 4: biological grade

    fooled  = g1_pass and g2_pass and g3_pass and g4_pass

    def gate(passed, label):
        c = G if passed else R
        s = "PASS" if passed else "FAIL"
        return f"{c}{s}{X} {D}{label}{X}"

    print(f"\n{'─'*60}")
    print(f"{W}{name}{X}")
    print(f"{D}{description}{X}")
    print(f"")
    print(f"  Avg physics score : {avg_sc:.1f}/100")
    print(f"  Rejected windows  : {rej_pct}%")
    print(f"  Hurst (H)         : {hurst}")
    print(f"  BFS               : {bfs}")
    print(f"  Grade             : {grade}")
    print(f"")
    print(f"  Gate 1 (Physics)  : {gate(g1_pass, f'rej={rej_pct}%')}")
    print(f"  Gate 2 (Hurst)    : {gate(g2_pass, f'H={hurst} threshold=0.70')}")
    print(f"  Gate 3 (BFS)      : {gate(g3_pass, f'BFS={bfs} threshold=0.35')}")
    print(f"  Gate 4 (Grade)    : {gate(g4_pass, grade)}")
    print(f"")
    if fooled:
        print(f"  {Y}⚠ PIPELINE FOOLED — signal passed all gates as HUMAN{X}")
    else:
        first_fail = (
            "Physics" if not g1_pass else
            "Hurst"   if not g2_pass else
            "BFS"     if not g3_pass else
            "Grade"
        )
        print(f"  {G}✓ Pipeline held — caught at Gate: {first_fail}{X}")

    return {
        "name": name, "hurst": hurst, "bfs": bfs, "grade": grade,
        "g1": g1_pass, "g2": g2_pass, "g3": g3_pass, "g4": g4_pass,
        "fooled": fooled
    }

# ── Signal definitions ────────────────────────────────────────────────────────

def pink_noise(i, hz, _state={"b": [0.0]*5}):
    """
    Pink noise (1/f) — Hurst ≈ 0.75-0.85.
    Generated via Voss-McCartney algorithm approximation.
    Designed to mimic biological long-range correlation.
    """
    b = _state["b"]
    white = random.gauss(0, 1.0)
    b[0] = 0.99886*b[0] + white*0.0555179
    b[1] = 0.99332*b[1] + white*0.0750759
    b[2] = 0.96900*b[2] + white*0.1538520
    b[3] = 0.86650*b[3] + white*0.3104856
    b[4] = 0.55000*b[4] + white*0.5329522
    pink = (b[0]+b[1]+b[2]+b[3]+b[4]+white*0.5362) * 0.11
    _state["b"] = b
    amp = 2.0
    return [pink*amp, pink*amp*0.7 + random.gauss(0,0.05), pink*amp*0.4 + random.gauss(0,0.03)]

def fbm_h075(i, hz, _state={"prev": [0.0,0.0,0.0], "hist": [[],[],[]]}):
    """
    Fractional Brownian Motion H=0.75 — by construction biological-looking.
    Uses Cholesky approximation with H=0.75 autocorrelation.
    """
    H = 0.75
    s = _state
    result = []
    for axis in range(3):
        hist = s["hist"][axis]
        noise = random.gauss(0, 1.0)
        if len(hist) > 0:
            # fBm increment: correlated with history via H
            corr = sum(((k+1)**(2*H) - k**(2*H)) * hist[-(k+1)]
                      for k in range(min(len(hist), 20))) * 0.05
            val = s["prev"][axis] + noise * 0.3 + corr * 0.1
        else:
            val = noise * 0.3
        val = max(-3.0, min(3.0, val))
        hist.append(val)
        if len(hist) > 50: hist.pop(0)
        s["prev"][axis] = val
        result.append(val)
    return result

def ar1_biological(i, hz, _state={"x": [0.0,0.0,0.0]}):
    """
    AR(1) process with phi=0.95 — strong autocorrelation.
    Mimics the slow drift of biological signals.
    phi=0.95 gives theoretical ACF that looks persistent.
    """
    phi = 0.95
    result = []
    for axis in range(3):
        noise = random.gauss(0, 0.5)
        val = phi * _state["x"][axis] + noise * math.sqrt(1 - phi**2)
        _state["x"][axis] = val
        result.append(val * 2.0)
    return result

def human_replay_with_drift(i, hz, _state={"phase": 0.0}):
    """
    Human motion replay (walk cycle simulation) with slow sensor drift.
    Walk: ~1.8Hz dominant, harmonics at 3.6Hz, 5.4Hz.
    Drift: 0.001 per sample — mimics temperature/calibration drift.
    """
    t = i / hz
    _state["phase"] += 0.001
    drift = _state["phase"]
    # Realistic walk spectrum
    x = (1.2 * math.sin(2*math.pi*1.8*t) +
         0.4 * math.sin(2*math.pi*3.6*t + 0.5) +
         0.15 * math.sin(2*math.pi*5.4*t + 1.1) +
         random.gauss(0, 0.08) + drift)
    y = (0.8 * math.sin(2*math.pi*1.8*t + math.pi/4) +
         0.25 * math.sin(2*math.pi*3.6*t + 1.0) +
         random.gauss(0, 0.06) + drift * 0.7)
    z = (9.81 +
         0.3 * math.sin(2*math.pi*3.6*t + 0.8) +
         random.gauss(0, 0.04) + drift * 0.3)
    return [x, y, z]

def sine_with_amplitude_envelope(i, hz):
    """
    Pure 50Hz sine with biological amplitude envelope.
    Envelope: slow 0.3Hz modulation mimicking muscle fatigue/tremor variation.
    Designed to look like tremor with natural amplitude variation.
    """
    t = i / hz
    envelope = 0.5 + 0.4 * math.sin(2 * math.pi * 0.3 * t)  # slow amplitude variation
    carrier  = math.sin(2 * math.pi * 10 * t)                 # 10Hz — tremor range
    noise    = random.gauss(0, 0.02)                           # tiny noise
    v = envelope * carrier + noise
    return [v, v * 0.6 + random.gauss(0, 0.01), v * 0.3 + random.gauss(0, 0.01)]

# ── Run all 5 ─────────────────────────────────────────────────────────────────

print("=" * 60)
print(f"{W}S2S ADVERSARIAL STRESS TEST{X}")
print("Signals designed to fool the biological origin detector.")
print("Human reference: NinaPro DB5 n=10, all HUMAN (H=0.73-0.81)")
print("=" * 60)

signals = [
    ("1. Pink Noise (1/f)",
     "Hurst ≈ 0.75-0.85 by construction. Designed to mimic biological LRC.",
     pink_noise),
    ("2. Fractional Brownian Motion H=0.75",
     "fBm with H=0.75 — mathematically persistent, mimics biological memory.",
     fbm_h075),
    ("3. AR(1) phi=0.95",
     "Strong autocorrelation. Theoretical Hurst-like persistence.",
     ar1_biological),
    ("4. Human Walk Replay + Drift",
     "Realistic walk spectrum (1.8Hz+harmonics) with slow sensor drift.",
     human_replay_with_drift),
    ("5. Sine with Biological Amplitude Envelope",
     "10Hz tremor-range carrier with 0.3Hz biological envelope modulation.",
     sine_with_amplitude_envelope),
]

results = []
for name, desc, fn in signals:
    r = run_signal(name, desc, fn)
    results.append(r)

print(f"\n{'='*60}")
print(f"{W}STRESS TEST SUMMARY{X}")
print(f"{'='*60}")
fooled = [r for r in results if r["fooled"]]
held   = [r for r in results if not r["fooled"]]
print(f"  {G}Pipeline held     : {len(held)}/5{X}")
print(f"  {Y if fooled else G}Pipeline fooled   : {len(fooled)}/5{X}")
if fooled:
    print(f"\n  {Y}Gaps found:{X}")
    for r in fooled:
        print(f"    ⚠ {r['name']} — H={r['hurst']} BFS={r['bfs']} grade={r['grade']}")
    print(f"\n  {Y}These signals require additional gates to detect.{X}")
else:
    print(f"\n  {G}All adversarial signals caught. Pipeline robust on this test set.{X}")
print(f"{'='*60}\n")
