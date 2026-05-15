"""
param_sweep.py — Find OU parameters that maximize S2S yield rate.

Run this first to calibrate the generator before building datasets.
Goal: find (theta, sigma, rho) combinations that produce >30% GOLD/SILVER yield.
"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from experiments.data_gen.ou_generator import generate_batch
from experiments.data_gen.s2s_sieve import sieve

N_PER_CONFIG = 50  # windows per configuration

configs = [
    {"theta": 3.0,  "sigma": 1.0, "rho_xy": 0.5, "rho_xz": 0.3, "rho_yz": 0.4},
    {"theta": 5.0,  "sigma": 1.5, "rho_xy": 0.5, "rho_xz": 0.3, "rho_yz": 0.4},
    {"theta": 8.0,  "sigma": 1.5, "rho_xy": 0.6, "rho_xz": 0.4, "rho_yz": 0.5},
    {"theta": 5.0,  "sigma": 2.0, "rho_xy": 0.6, "rho_xz": 0.4, "rho_yz": 0.5},
    {"theta": 10.0, "sigma": 1.0, "rho_xy": 0.7, "rho_xz": 0.5, "rho_yz": 0.6},
    {"theta": 3.0,  "sigma": 0.5, "rho_xy": 0.4, "rho_xz": 0.2, "rho_yz": 0.3},
]

print("S2S OU Parameter Sweep")
print("=" * 60)

best_yield = 0
best_config = None

for cfg in configs:
    windows = generate_batch(N_PER_CONFIG, seed_start=0,
                             n_samples=256, hz=100.0, **cfg)
    t0 = time.perf_counter()
    certified = sieve(windows, min_tier="SILVER")
    elapsed = time.perf_counter() - t0
    yield_pct = len(certified) / N_PER_CONFIG * 100
    tiers = {}
    for w in certified:
        tiers[w["tier"]] = tiers.get(w["tier"], 0) + 1
    print(f"  θ={cfg['theta']:<4} σ={cfg['sigma']:<3} ρ=({cfg['rho_xy']},{cfg['rho_xz']},{cfg['rho_yz']})  "
          f"→ yield={yield_pct:.0f}%  {tiers}")
    if yield_pct > best_yield:
        best_yield = yield_pct
        best_config = cfg

print(f"\nBest config ({best_yield:.0f}% yield): {best_config}")
