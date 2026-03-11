"""
S2S Mixed Contamination Stress Test
Find the exact contamination % where the pipeline breaks.

Method:
  - Load real s5 NinaPro windows (confirmed HUMAN, H=0.74-0.80)
  - Inject pink noise windows at 0-50% contamination
  - Run certify_session() at each level
  - Find the boundary where HUMAN → NOT_BIOLOGICAL

This is the true adversarial boundary — not synthetic-only signals,
but realistic contamination of real biological data.
"""

import sys, os, random, math
import numpy as np
import scipy.io, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

G  = "\033[92m"
R  = "\033[91m"
Y  = "\033[93m"
W  = "\033[97m"
D  = "\033[2m"
X  = "\033[0m"

HZ         = 2000
WINDOW     = 2000
STEP       = 1000
HURST_GATE = 0.70

# ── Load all s5 windows ───────────────────────────────────────────────────────
def load_s5_windows():
    subject_dir = os.path.expanduser("~/ninapro_db5/s5")
    mat_files   = sorted(glob.glob(os.path.join(subject_dir, "*.mat")))
    segments    = []
    for mf in mat_files:
        try:
            md  = scipy.io.loadmat(mf)
            acc = md.get('acc', md.get('ACC'))
            if acc is not None:
                segments.append(acc[:, :3])
        except: continue
    if not segments:
        raise RuntimeError("Could not load s5 NinaPro data")
    acc = np.concatenate(segments, axis=0)
    windows = []
    for i in range(0, len(acc) - WINDOW + 1, STEP):
        windows.append(acc[i:i+WINDOW, :3].tolist())
    return windows

# ── Generate pink noise window ────────────────────────────────────────────────
def pink_noise_window(n=WINDOW):
    b = [0.0]*5
    result = []
    for _ in range(n):
        white = random.gauss(0, 1.0)
        b[0] = 0.99886*b[0] + white*0.0555179
        b[1] = 0.99332*b[1] + white*0.0750759
        b[2] = 0.96900*b[2] + white*0.1538520
        b[3] = 0.86650*b[3] + white*0.3104856
        b[4] = 0.55000*b[4] + white*0.5329522
        pink = (b[0]+b[1]+b[2]+b[3]+b[4]+white*0.5362) * 0.11 * 2.0
        result.append([pink, pink*0.7+random.gauss(0,0.05), pink*0.4+random.gauss(0,0.03)])
    return result

# ── Run certify_session on mixed pool ────────────────────────────────────────
def run_contaminated(real_windows, contamination_pct, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    n_total     = len(real_windows)
    n_synthetic = int(n_total * contamination_pct / 100)

    # Keep temporal order — Hurst R/S analysis requires time-ordered sequence
    # Inject synthetic windows at random positions, real windows stay ordered
    inject_positions = set(random.sample(range(n_total), n_synthetic))
    synth_iter = iter([pink_noise_window() for _ in range(n_synthetic)])
    real_iter  = iter(real_windows)
    pool = []
    for idx in range(n_total):
        if idx in inject_positions:
            pool.append(("SYNTHETIC", next(synth_iter)))
        else:
            pool.append(("REAL", next(real_iter)))
    n_real = n_total - n_synthetic
    # NOTE: NO shuffle — temporal order preserved for valid Hurst estimation

    pe     = PhysicsEngine()
    dt_ns  = int(1e9 / HZ)
    scores = []
    tiers  = []

    for label, window in pool:
        ts = [j * dt_ns for j in range(WINDOW)]
        r  = pe.certify(imu_raw={"timestamps_ns": ts, "accel": window}, segment="forearm")
        scores.append(r["physical_law_score"])
        tiers.append(r["tier"])

    session = pe.certify_session()
    hurst   = session.get("hurst") or 0
    bfs     = session.get("bfs") or 0
    grade   = session.get("biological_grade", "UNKNOWN")
    rec     = session.get("recommendation", "?")

    return {
        "pct":      contamination_pct,
        "n_total":  n_total,
        "n_real":   n_real,
        "n_synth":  n_synthetic,
        "hurst":    hurst,
        "bfs":      bfs,
        "grade":    grade,
        "rec":      rec,
        "passed":   grade in ("HUMAN", "SUPERHUMAN"),
        "avg_score": round(sum(scores)/len(scores), 1) if scores else 0,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 62)
print(f"{W}S2S MIXED CONTAMINATION STRESS TEST{X}")
print("Real s5 NinaPro windows + pink noise injection")
print("Finding exact contamination boundary where pipeline fails")
print("=" * 62)

print("\nLoading s5 NinaPro data (all 3 exercises)...")
real_windows = load_s5_windows()
print(f"Loaded {len(real_windows)} real windows from s5\n")

# Test contamination levels 0-50% in 5% steps, then fine-grain around boundary
levels   = list(range(0, 55, 5))
results  = []
boundary = None

print(f"{'Contam%':<10} {'H':<8} {'BFS':<8} {'Grade':<22} {'Decision':<10} {'Status'}")
print("─" * 72)

for pct in levels:
    r = run_contaminated(real_windows, pct)
    results.append(r)

    gc     = G if r["passed"] else R
    status = f"{G}HUMAN ✓{X}" if r["passed"] else f"{R}BROKEN ✗{X}"
    hc     = G if r["hurst"] >= HURST_GATE else R

    print(f"  {pct:>3}%      {hc}{r['hurst']:<8}{X} {r['bfs']:<8} {gc}{r['grade']:<22}{X} {gc}{r['rec']:<10}{X} {status}")

    # Find first failure
    if boundary is None and not r["passed"]:
        boundary = pct

# Fine-grain search around boundary
if boundary and boundary > 5:
    print(f"\n{Y}Boundary found near {boundary}% — fine-grain search {boundary-5}% to {boundary}%...{X}\n")
    fine_levels = [boundary - 4, boundary - 3, boundary - 2, boundary - 1, boundary]
    for pct in fine_levels:
        if any(r["pct"] == pct for r in results):
            continue
        r = run_contaminated(real_windows, pct)
        results.append(r)
        gc     = G if r["passed"] else R
        status = f"{G}HUMAN ✓{X}" if r["passed"] else f"{R}BROKEN ✗{X}"
        hc     = G if r["hurst"] >= HURST_GATE else R
        print(f"  {pct:>3}%      {hc}{r['hurst']:<8}{X} {r['bfs']:<8} {gc}{r['grade']:<22}{X} {gc}{r['rec']:<10}{X} {status}")
        if not r["passed"]:
            boundary = pct

results.sort(key=lambda x: x["pct"])
last_pass = max((r["pct"] for r in results if r["passed"]), default=0)
first_fail = min((r["pct"] for r in results if not r["passed"]), default=100)

print(f"\n{'='*62}")
print(f"{W}ADVERSARIAL BOUNDARY RESULT{X}")
print(f"{'='*62}")
print(f"  Last HUMAN grade  : {G}{last_pass}% contamination{X}")
print(f"  First failure     : {R}{first_fail}% contamination{X}")
print(f"  Pipeline boundary : between {last_pass}% and {first_fail}%")
print(f"")
print(f"  Interpretation:")
print(f"  Up to {last_pass}% pink noise injection → still graded HUMAN")
print(f"  At {first_fail}% pink noise injection → pipeline detects contamination")
if first_fail <= 20:
    print(f"  {G}Strong boundary — pipeline catches contamination early{X}")
elif first_fail <= 35:
    print(f"  {Y}Moderate boundary — pipeline catches at medium contamination{X}")
else:
    print(f"  {R}Weak boundary — high contamination required to trigger{X}")
print(f"{'='*62}\n")
