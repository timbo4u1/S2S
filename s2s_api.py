"""
S2S Certification API
POST /certify — certify a window of IMU data
GET  /health  — health check
GET  /        — usage info

Deploy free: render.com, railway.app, or fly.io
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

app = FastAPI(
    title="S2S Certification API",
    description="Physics certification for human motion sensor data",
    version="1.4.4"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class CertifyRequest(BaseModel):
    accel: List[List[float]]        # [[ax,ay,az], ...] — required
    gyro: Optional[List[List[float]]] = None   # [[gx,gy,gz], ...] — optional
    sample_rate_hz: Optional[float] = 50.0


class CertifyResponse(BaseModel):
    tier: str           # GOLD / SILVER / BRONZE / REJECTED
    score: float        # 0–100
    laws_passed: list
    laws_failed: list
    windows: int
    sample_rate_hz: float


@app.get("/")
def root():
    return {
        "name": "S2S Certification API",
        "version": "1.4.4",
        "usage": {
            "endpoint": "POST /certify",
            "body": {
                "accel": "[[ax,ay,az], ...] — list of 3-axis accelerometer samples",
                "gyro":  "[[gx,gy,gz], ...] — optional gyroscope samples",
                "sample_rate_hz": "sampling rate in Hz (default: 50)"
            },
            "example_python": (
                "import requests\n"
                "import numpy as np\n"
                "data = np.random.randn(256, 3).tolist()\n"
                "r = requests.post('https://api.s2s.dev/certify', json={'accel': data, 'sample_rate_hz': 50})\n"
                "print(r.json())"
            )
        },
        "github": "https://github.com/timbo4u1/S2S",
        "pypi": "pip install s2s-certify"
    }


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.4.4"}


@app.post("/certify", response_model=CertifyResponse)
def certify_endpoint(req: CertifyRequest):
    # Validate input
    if len(req.accel) < 32:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 32 samples, got {len(req.accel)}. Recommended: 256 samples."
        )

    if len(req.accel[0]) != 3:
        raise HTTPException(
            status_code=400,
            detail="accel must be [[ax,ay,az], ...] — 3 values per sample"
        )

    if req.gyro is not None:
        if len(req.gyro) != len(req.accel):
            raise HTTPException(
                status_code=400,
                detail="gyro and accel must have the same number of samples"
            )
        if len(req.gyro[0]) != 3:
            raise HTTPException(
                status_code=400,
                detail="gyro must be [[gx,gy,gz], ...] — 3 values per sample"
            )

    try:
        from s2s_certify import certify
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="s2s-certify not installed on this server. Contact admin."
        )

    try:
        accel_np = np.array(req.accel, dtype=float)

        # Certify in 256-sample windows
        window_size = 256
        results = []

        for start in range(0, len(accel_np) - window_size + 1, window_size):
            window = accel_np[start:start + window_size]
            result = certify(window, sample_rate_hz=req.sample_rate_hz)
            results.append(result)

        if not results:
            # Single window smaller than 256
            result = certify(accel_np, sample_rate_hz=req.sample_rate_hz)
            results = [result]

        # Aggregate: worst tier wins (conservative)
        tier_order = {"GOLD": 4, "SILVER": 3, "BRONZE": 2, "REJECTED": 1}
        scores = [r["score"] for r in results]
        tiers = [r["tier"] for r in results]
        worst_tier = min(tiers, key=lambda t: tier_order.get(t, 0))
        avg_score = float(np.mean(scores))

        # Aggregate laws
        all_passed = set(results[0].get("laws_passed", []))
        all_failed = set(results[0].get("laws_failed", []))
        for r in results[1:]:
            all_passed &= set(r.get("laws_passed", []))   # only laws that ALWAYS pass
            all_failed |= set(r.get("laws_failed", []))   # any law that ever fails

        return CertifyResponse(
            tier=worst_tier,
            score=round(avg_score, 1),
            laws_passed=sorted(list(all_passed)),
            laws_failed=sorted(list(all_failed)),
            windows=len(results),
            sample_rate_hz=req.sample_rate_hz
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
