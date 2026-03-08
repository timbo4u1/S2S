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
    imu_raw = _to_imu_raw(accel, gyro, hz)
    laws_passed = []
    laws_failed = []
    scores = []

    checks = [
        ("resonance",        lambda: check_resonance(imu_raw)),
        ("rigid_body",       lambda: check_rigid_body(imu_raw)),
        ("jerk",             lambda: check_jerk(imu_raw)),
        ("imu_consistency",  lambda: check_imu_consistency(imu_raw)),
    ]

    for name, fn in checks:
        try:
            passed, score, _ = fn()
            if passed:
                laws_passed.append(name)
            else:
                laws_failed.append(name)
            scores.append(max(0, min(100, score)))
        except Exception:
            laws_failed.append(name)
            scores.append(0)

    avg_score = float(np.mean(scores)) if scores else 0.0

    if avg_score >= 87:
        tier = "GOLD"
    elif avg_score >= 75:
        tier = "SILVER"
    elif avg_score >= 60:
        tier = "BRONZE"
    else:
        tier = "REJECTED"

    return tier, avg_score, laws_passed, laws_failed


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
