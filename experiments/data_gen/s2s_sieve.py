"""
s2s_sieve.py — Filter OU-generated windows through S2S 12-law engine.

The sieve: spray random OU windows → keep only GOLD/SILVER → certified dataset.
Yield rate (% passing) is the key metric for generator quality.
"""
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

TIER_RANK = {"GOLD": 3, "SILVER": 2, "BRONZE": 1, "REJECTED": 0}


def sieve(
    windows: List[Dict],
    min_tier: str = "SILVER",
    segment: str = "forearm",
    verbose: bool = False,
) -> List[Dict]:
    """
    Run S2S certification on each window, return only those passing min_tier.

    Each window dict must have: {timestamps_ns, accel, gyro}
    Returns list of dicts with added: tier, score, laws_passed, laws_failed
    """
    engine = PhysicsEngine()
    min_rank = TIER_RANK.get(min_tier, 2)
    certified = []

    for i, w in enumerate(windows):
        # Fresh engine state — windows are independent
        engine._last_terminal_state = None
        r = engine.certify(w, segment=segment)
        tier_rank = TIER_RANK.get(r["tier"], 0)

        if verbose and i % 100 == 0:
            print(f"  [{i}/{len(windows)}] tier={r['tier']} score={r['physical_law_score']}")

        if tier_rank >= min_rank:
            certified.append({
                **w,
                "tier":         r["tier"],
                "score":        r["physical_law_score"],
                "laws_passed":  r["laws_passed"],
                "laws_failed":  r["laws_failed"],
            })

    return certified


def sieve_report(
    n_generated: int,
    certified: List[Dict],
    elapsed: float,
) -> None:
    tiers = {}
    for w in certified:
        tiers[w["tier"]] = tiers.get(w["tier"], 0) + 1
    yield_pct = len(certified) / max(n_generated, 1) * 100
    print(f"\n  Generated:  {n_generated}")
    print(f"  Certified:  {len(certified)} ({yield_pct:.1f}% yield)")
    print(f"  Tier split: {tiers}")
    print(f"  Time:       {elapsed:.1f}s  ({elapsed/max(n_generated,1)*1000:.1f}ms/window)")
