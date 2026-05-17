"""
minjerk_fill.py — Phase 2 of head-tail-fill pipeline.

Generates minimum-jerk body between real head and tail anchors.
Certifies each filled window with S2S Laws 1-15.
Keeps GOLD and SILVER only.

Known limitation: minimum-jerk applied in acceleration space directly,
not pose space. Empirically motivated, not biomechanically derived.
See roadmap for correct long-term pipeline.

Input:  experiments/data_gen/gaps.json (from gap_detector.py)
Output: experiments/data_gen/certified_fills.json
"""

import sys, json, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

# ── config ────────────────────────────────────────────────────────────────────
HZ       = 200
N        = max(5, int(HZ * 0.05))    # 10 — head/tail length, must match gap_detector
WIN      = 256
BODY_LEN = WIN - 2 * N               # 236 samples
MAX_GAPS = None                       # None = all gaps

# ── minimum-jerk in 3D acceleration space ────────────────────────────────────
def minjerk_3d(start, end, n_steps):
    """
    Flash & Hogan 1985 minimum-jerk trajectory.
    Applied directly in acceleration space (known limitation — see roadmap).

    start, end: [ax, ay, az] endpoint states
    n_steps:    number of intermediate samples to generate
    Returns list of n_steps [ax, ay, az] samples.
    """
    out = []
    for i in range(n_steps):
        tau = (i + 1) / (n_steps + 1)          # normalised time 0→1
        # minimum-jerk position polynomial
        w   = 10*tau**3 - 15*tau**4 + 6*tau**5
        sample = [start[k] + (end[k] - start[k]) * w for k in range(3)]
        out.append(sample)
    return out

def make_ts(n, hz, offset=0):
    return [int(offset * 1e9 / hz) + int(i * 1e9 / hz) for i in range(n)]

def certify_window(acc_list):
    ts = make_ts(len(acc_list), HZ)
    return PhysicsEngine().certify({
        "timestamps_ns": ts,
        "accel":         acc_list,
        "gyro":          [[0, 0, 0]] * len(acc_list)
    })

def check_jerk_bound(body, hz, limit=500.0):
    """Verify jerk ≤ 500 m/s³ throughout body (Law 6 — not a guarantee, a check)."""
    dt = 1.0 / hz
    violations = 0
    for i in range(1, len(body)):
        for k in range(3):
            j = abs(body[i][k] - body[i-1][k]) / dt
            if j > limit:
                violations += 1
    return violations

# ── main ─────────────────────────────────────────────────────────────────────
def run():
    gaps_path = Path(__file__).parent / "gaps.json"
    if not gaps_path.exists():
        print("ERROR: gaps.json not found — run gap_detector.py first")
        sys.exit(1)

    with open(gaps_path) as f:
        data = json.load(f)

    gaps = data["gaps"]
    if MAX_GAPS:
        gaps = gaps[:MAX_GAPS]

    print(f"MinJerk Fill — {len(gaps)} gaps")
    print(f"Body length: {BODY_LEN} samples = {BODY_LEN/HZ*1000:.0f}ms")
    print(f"Total window: {N} + {BODY_LEN} + {N} = {N+BODY_LEN+N} samples")
    print()

    certified_fills = []
    stats = {
        "total": 0, "gold": 0, "silver": 0,
        "bronze": 0, "rejected": 0,
        "jerk_violations": 0
    }
    by_gesture = {}

    for gap in gaps:
        tail   = gap["tail_samples"]   # last N real samples — immutable
        head   = gap["head_samples"]   # first N real samples — immutable
        label  = gap["gesture_label"]
        subj   = gap["subject"]

        # endpoint states for minimum-jerk
        start_state = tail[-1]         # last sample of tail
        end_state   = head[0]          # first sample of head

        # generate body
        body = minjerk_3d(start_state, end_state, BODY_LEN)

        # check jerk before certification (Law 6 hypothesis check)
        viols = check_jerk_bound(body, HZ)
        if viols > 0:
            stats["jerk_violations"] += viols

        # assemble full window: [tail | body | head]
        full_window = tail + body + head   # N + BODY_LEN + N = WIN samples

        # S2S certification
        r    = certify_window(full_window)
        tier = r["tier"]
        stats["total"] += 1
        stats[tier.lower()] = stats.get(tier.lower(), 0) + 1

        if tier in ("GOLD", "SILVER"):
            fill = {
                "subject":            subj,
                "gesture_label":      label,
                "tier":               tier,
                "score":              r.get("score", r.get("physical_law_score", 0)),
                "laws_failed":        r.get("laws_failed", []),
                "euclidean_distance": gap["euclidean_distance"],
                "cosine_similarity":  gap["cosine_similarity"],
                "jerk_violations":    viols,
                "full_window":        full_window,  # N+BODY+N samples
                "head_samples":       head,          # original — immutable
                "tail_samples":       tail,          # original — immutable
                "body_samples":       body,          # synthetic
            }
            certified_fills.append(fill)
            by_gesture[label] = by_gesture.get(label, 0) + 1

    # ── report ────────────────────────────────────────────────────────────────
    total   = stats["total"]
    kept    = stats["gold"] + stats["silver"]
    rej_rate = 100 * (total - kept) // max(total, 1)

    print("=" * 52)
    print(f"FILL RESULTS")
    print(f"  Total generated:    {total}")
    print(f"  GOLD:               {stats['gold']} ({100*stats['gold']//max(total,1)}%)")
    print(f"  SILVER:             {stats['silver']} ({100*stats['silver']//max(total,1)}%)")
    print(f"  BRONZE:             {stats.get('bronze',0)}")
    print(f"  REJECTED:           {stats.get('rejected',0)}")
    print(f"  Rejection rate:     {rej_rate}%")
    print(f"  Jerk violations:    {stats['jerk_violations']} (Law 6 hypothesis)")
    print(f"  Gesture classes:    {len(by_gesture)}")
    print()

    if rej_rate < 1:
        print("  WARNING: 0% rejection — S2S may not be discriminating fills")
        print("  Consider tightening fillability criteria in gap_detector.py")
    elif rej_rate > 50:
        print("  WARNING: >50% rejection — min-jerk parameters may need tuning")
    else:
        print("  Rejection rate in expected range — S2S is doing real work")

    print()
    print(f"  Certified fills by gesture:")
    for lbl, cnt in sorted(by_gesture.items(), key=lambda x:-x[1])[:8]:
        print(f"    gesture {lbl:3d}: {cnt} fills")

    # save
    out = Path(__file__).parent / "certified_fills.json"
    with open(out, "w") as f:
        json.dump({"meta": stats, "by_gesture": by_gesture,
                   "fills": certified_fills}, f, indent=2)
    print()
    print(f"  Saved: {out} ({len(certified_fills)} certified fills)")
    print("=" * 52)

if __name__ == "__main__":
    run()
