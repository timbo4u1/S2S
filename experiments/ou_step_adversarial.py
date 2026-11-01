"""OU + common-mode step test. EXPLORATORY, post-hoc: designed after C-MHAD Part B.
N=256 @100 Hz, segment=forearm, seeds 1000-1299.
OU: theta~U(1,4) sigma~U(3,8) rho~U(0.2,0.8); accel=OU+[0,0,9.81]; gyro=OU(sigma/4).
Attack: +5 m/s^2 on all accel axes from sample 128.
Prediction made before the run: Law 16 clears in nearly all flagged windows."""
import os, sys
import numpy as np
from collections import Counter
sys.path.insert(0, os.path.expanduser("~/S2S"))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

N, HZ = 256, 100
ts = [int(i * 1e9 / HZ) for i in range(N)]

def ou(rng, theta, sigma, rho):
    C = np.full((3, 3), rho); np.fill_diagonal(C, 1.0)
    L = np.linalg.cholesky(C)
    x = np.zeros((N, 3))
    for i in range(1, N):
        x[i] = x[i-1] - theta * x[i-1] / HZ + sigma * np.sqrt(1 / HZ) * (L @ rng.standard_normal(3))
    return x

def cert(a, g):
    r = PhysicsEngine().certify(imu_raw={"timestamps_ns": ts, "accel": a.tolist(),
                                         "gyro": g.tolist()}, segment="forearm")
    return r["tier"], set(r.get("laws_failed", [])), set(r.get("laws_not_assessed", []))

FW = ["cross_axis_cohesion", "temporal_autocorrelation", "innovation_kurtosis"]
before, after, had, cleared = Counter(), Counter(), Counter(), Counter()
n = 300
for seed in range(n):
    rng = np.random.default_rng(1000 + seed)
    th, sg, rho = rng.uniform(1, 4), rng.uniform(3, 8), rng.uniform(0.2, 0.8)
    a = ou(rng, th, sg, rho) + np.array([0, 0, 9.81])
    g = ou(rng, th, sg / 4, rho)
    b = a.copy(); b[N // 2:] += 5.0
    ta, fa, na = cert(a, g); tb, fb, nb = cert(b, g)
    before[ta] += 1; after[tb] += 1
    for law in FW:
        if law in fa:
            had[law] += 1
            cleared[law] += (law not in fb) and (law not in nb)
print("tiers before:", dict(before))
print("tiers after: ", dict(after))
for law in FW:
    print("%-26s flagged in %3d windows, cleared by the step in %3d" % (law, had[law], cleared[law]))
