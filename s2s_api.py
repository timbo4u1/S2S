"""
S2S Certification API
POST /certify — certify a window of IMU data
GET  /health  — health check
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from s2s_standard_v1_3.s2s_physics_v1_3 import (
    check_resonance, check_rigid_body, check_jerk, check_imu_consistency
)

def _real_score(accel, hz=50.0):
    """Data-driven signal quality scorer — measures actual signal properties."""
    a = np.array(accel, dtype=float)
    scores = {}
    laws_passed = []
    laws_failed = []

    # 1. FREEZE — consecutive identical values = dead sensor
    diffs = np.diff(a, axis=0)
    freeze_ratio = float(np.mean(np.all(np.abs(diffs) < 1e-6, axis=1)))
    s = max(0.0, 100.0 - freeze_ratio * 500.0)
    scores['no_freeze'] = s
    (laws_passed if s > 50 else laws_failed).append('no_freeze')

    # 2. CLIPPING — values stuck at max = saturated ADC
    amax = float(np.max(np.abs(a)))
    if amax > 0:
        clip_ratio = float(np.mean(np.abs(np.abs(a) - amax) < 0.01 * amax))
        s = max(0.0, 100.0 - clip_ratio * 400.0)
    else:
        s = 0.0
    scores['no_clipping'] = s
    (laws_passed if s > 60 else laws_failed).append('no_clipping')

    # 3. VARIANCE — too low=frozen, too high=pure noise, right=human motion
    var = float(np.mean(np.var(a, axis=0)))
    if var < 0.001:
        s = 10.0
    elif var > 100:
        s = 15.0
    elif var > 20:
        s = 40.0
    elif var > 5:
        s = 65.0
    else:
        s = 88.0
    scores['variance_ok'] = s
    (laws_passed if s > 60 else laws_failed).append('variance_ok')

    # 4. GRAVITY — median magnitude should be near 9.81 m/s²
    mag = np.linalg.norm(a, axis=1)
    err = abs(float(np.median(mag)) - 9.81)
    s = max(0.0, 100.0 - err * 12.0)
    scores['gravity_ok'] = s
    (laws_passed if s > 50 else laws_failed).append('gravity_ok')

    # 5. JERK — human motion has bounded smooth jerk
    dt = 1.0 / hz
    j = np.diff(np.diff(np.diff(a, axis=0), axis=0), axis=0) / (dt**3)
    p95 = float(np.percentile(np.abs(j), 95))
    if p95 < 200:
        s = 92.0
    elif p95 < 3000:
        s = 72.0
    elif p95 < 15000:
        s = 42.0
    else:
        s = 12.0
    scores['jerk_ok'] = s
    (laws_passed if s > 50 else laws_failed).append('jerk_ok')

    # Also run physics checks for law names
    try:
        n = len(a); dt_ns = int(1e9/hz)
        imu = {'accel': a.tolist(), 'gyro': [[0,0,0]]*n,
               'timestamps_ns': [i*dt_ns for i in range(n)], 'sample_rate_hz': hz}
        for name, fn in [('resonance', check_resonance), ('rigid_body', check_rigid_body),
                         ('jerk_physics', check_jerk), ('imu_consistency', check_imu_consistency)]:
            passed, sc, _ = fn(imu)
            if passed and name not in laws_passed:
                laws_passed.append(name)
            elif not passed and name not in laws_failed:
                laws_failed.append(name)
    except Exception:
        pass

    avg = float(np.mean(list(scores.values())))

    # Force REJECTED for pathological cases regardless of avg
    var = float(np.mean(np.var(np.array(accel), axis=0)))
    all_noise = var > 20
    all_frozen = var < 0.01
    if all_frozen or all_noise:
        tier = 'REJECTED'
    elif avg >= 75:
        tier = 'SILVER'
    elif avg >= 52:
        tier = 'BRONZE'
    else:
        tier = 'REJECTED'

    return tier, round(avg, 1), laws_passed, laws_failed

app = FastAPI(title="S2S Certification API", version="1.4.4")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class CertifyRequest(BaseModel):
    accel: List[List[float]]
    gyro: Optional[List[List[float]]] = None
    sample_rate_hz: Optional[float] = 50.0


class CertifyResponse(BaseModel):
    tier: str
    score: float
    laws_passed: list
    laws_failed: list
    windows: int
    sample_rate_hz: float


def _to_imu_raw(accel, gyro, hz):
    n = len(accel)
    dt_ns = int(1e9 / hz)
    ts = [i * dt_ns for i in range(n)]
    imu = {
        "accel": accel,
        "gyro": gyro if gyro else [[0.0, 0.0, 0.0]] * n,
        "timestamps_ns": ts,
        "sample_rate_hz": hz,
    }
    return imu


def _certify_window(accel, gyro, hz):
    return _real_score(accel, hz)


@app.get("/")
def root():
    return {
        "name": "S2S Certification API",
        "version": "1.4.4",
        "usage": "POST /certify with {accel: [[ax,ay,az],...], sample_rate_hz: 50}",
        "github": "https://github.com/timbo4u1/S2S",
        "pypi": "pip install s2s-certify"
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.4.4"}


@app.post("/certify", response_model=CertifyResponse)
def certify_endpoint(req: CertifyRequest):
    if len(req.accel) < 32:
        raise HTTPException(status_code=400, detail=f"Need at least 32 samples, got {len(req.accel)}")
    if len(req.accel[0]) != 3:
        raise HTTPException(status_code=400, detail="accel must be [[ax,ay,az],...]")

    accel = req.accel
    gyro = req.gyro
    hz = req.sample_rate_hz or 50.0

    window_size = 256
    accel_np = np.array(accel)
    all_tiers, all_scores, all_passed, all_failed = [], [], set(), set()

    for start in range(0, len(accel_np) - window_size + 1, window_size):
        w_acc = accel_np[start:start+window_size].tolist()
        w_gyr = gyro[start:start+window_size] if gyro else None
        tier, score, lp, lf = _certify_window(w_acc, w_gyr, hz)
        all_tiers.append(tier)
        all_scores.append(score)
        all_passed.update(lp)
        all_failed.update(lf)

    if not all_tiers:
        tier, score, lp, lf = _certify_window(accel, gyro, hz)
        all_tiers, all_scores = [tier], [score]
        all_passed, all_failed = set(lp), set(lf)

    tier_order = {"GOLD": 4, "SILVER": 3, "BRONZE": 2, "REJECTED": 1}
    worst = min(all_tiers, key=lambda t: tier_order.get(t, 0))

    return CertifyResponse(
        tier=worst,
        score=round(float(np.mean(all_scores)), 1),
        laws_passed=sorted(all_passed - all_failed),
        laws_failed=sorted(all_failed),
        windows=len(all_tiers),
        sample_rate_hz=hz
    )
