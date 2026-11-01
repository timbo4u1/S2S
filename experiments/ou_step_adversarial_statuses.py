import numpy as np
from collections import Counter
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

N, HZ = 256, 100
ts = [int(i * 1e9 / HZ) for i in range(N)]

FW = [
    "cross_axis_cohesion",
    "temporal_autocorrelation",
    "innovation_kurtosis"
]

def ou(rng, theta, sigma, rho):
    C = np.full((3, 3), rho)
    np.fill_diagonal(C, 1.0)
    L = np.linalg.cholesky(C)
    x = np.zeros((N, 3))

    for i in range(1, N):
        x[i] = (
            x[i-1]
            - theta * x[i-1] / HZ
            + sigma * np.sqrt(1 / HZ) *
            (L @ rng.standard_normal(3))
        )
    return x

def cert(a, g):
    return PhysicsEngine().certify(
        imu_raw={
            "timestamps_ns": ts,
            "accel": a.tolist(),
            "gyro": g.tolist()
        },
        segment="forearm"
    )

before = Counter()
after = Counter()
tier_transitions = Counter()
law_transitions = {law: Counter() for law in FW}
triple_before = Counter()
triple_after = Counter()
score_deltas = []

metric_keys = {
    "cross_axis_cohesion": "max_r",
    "temporal_autocorrelation": "acf1",
    "innovation_kurtosis": "excess_kurtosis"
}
metrics = {
    law: {"before": [], "after": []}
    for law in FW
}

for seed in range(1000, 1300):
    rng = np.random.default_rng(seed)

    theta = rng.uniform(1, 4)
    sigma = rng.uniform(3, 8)
    rho = rng.uniform(0.2, 0.8)

    a = ou(rng, theta, sigma, rho) + np.array([0, 0, 9.81])
    g = ou(rng, theta, sigma / 4, rho)

    b = a.copy()
    b[N // 2:] += 5.0

    r0 = cert(a, g)
    r1 = cert(b, g)

    before[r0["tier"]] += 1
    after[r1["tier"]] += 1
    tier_transitions[(r0["tier"], r1["tier"])] += 1

    score_deltas.append(
        r1["physical_law_score"] - r0["physical_law_score"]
    )

    s0 = r0["law_statuses"]
    s1 = r1["law_statuses"]

    state0 = tuple(s0.get(law, "MISSING") for law in FW)
    state1 = tuple(s1.get(law, "MISSING") for law in FW)

    triple_before[state0] += 1
    triple_after[state1] += 1

    for law in FW:
        status0 = s0.get(law, "MISSING")
        status1 = s1.get(law, "MISSING")
        law_transitions[law][status0 + "->" + status1] += 1

        key = metric_keys[law]
        d0 = r0.get("law_details", {}).get(law, {})
        d1 = r1.get("law_details", {}).get(law, {})

        if key in d0 and key in d1:
            metrics[law]["before"].append(float(d0[key]))
            metrics[law]["after"].append(float(d1[key]))

print("WINDOWS: 300")
print("TIERS BEFORE:", dict(before))
print("TIERS AFTER:", dict(after))
print("TIER TRANSITIONS:", dict(tier_transitions))
print("MEAN SCORE DELTA:", round(float(np.mean(score_deltas)), 4))

for law in FW:
    print("\nLAW:", law)
    print("STATUS TRANSITIONS:", dict(law_transitions[law]))

    x = metrics[law]["before"]
    y = metrics[law]["after"]

    if x and y:
        print(
            "METRIC: n=%d before=%.5f after=%.5f delta=%+.5f"
            % (len(x), np.mean(x), np.mean(y), np.mean(y)-np.mean(x))
        )

print("\nCOMPLETE FIREWALL STATES BEFORE:")
for state, count in triple_before.items():
    print(state, count)

print("\nCOMPLETE FIREWALL STATES AFTER:")
for state, count in triple_after.items():
    print(state, count)

print("\nNo source files modified.")
