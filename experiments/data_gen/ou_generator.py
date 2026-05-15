"""
ou_generator.py — Coupled OU process for S2S-certified synthetic IMU data.

The key insight from S2S development:
- Plain iid Gaussian fails cross_axis_cohesion (independent axes) AND temporal_acf
- OU process with coupled axes passes temporal_acf but fails cross_axis
- Coupled OU (Cholesky covariance) passes BOTH — this is the generator sweet spot

OU process: dx = -θ(x - μ)dt + σ√dt * dW
Parameters tuned against S2S 12-law benchmark:
  theta: 3-8   → tremor-like mean reversion (matches forearm biomechanics)
  sigma: 0.5-2 → realistic forearm acceleration amplitude
  rho:  0.4-0.6 → axis coupling from rigid body constraint (NinaPro calibrated)
"""
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_window(
    n_samples: int = 256,
    hz: float = 100.0,
    theta: float = 5.0,        # mean reversion speed
    sigma: float = 1.5,        # volatility (m/s²)
    rho_xy: float = 0.55,      # axis coupling x-y
    rho_xz: float = 0.35,      # axis coupling x-z
    rho_yz: float = 0.45,      # axis coupling y-z
    gyro_scale: float = 0.08,  # gyro amplitude ratio to accel
    gravity: float = 9.81,
    seed: Optional[int] = None,
    jitter_ns: float = 50000.0,
) -> Dict:
    """
    Generate one coupled OU window that targets S2S GOLD/SILVER certification.

    Returns imu_raw dict: {timestamps_ns, accel, gyro}
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / hz

    # Build correlation matrix and Cholesky factor
    corr = np.array([
        [1.0,    rho_xy, rho_xz],
        [rho_xy, 1.0,    rho_yz],
        [rho_xz, rho_yz, 1.0   ]
    ])
    # Ensure positive definite
    corr = np.clip(corr, -0.95, 0.95)
    np.fill_diagonal(corr, 1.0)
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.eye(3)  # fallback

    def _ou_path(scale: float) -> np.ndarray:
        x = np.zeros((n_samples, 3))
        for i in range(1, n_samples):
            noise = rng.standard_normal(3)
            x[i] = (x[i-1]
                    - theta * x[i-1] * dt
                    + scale * math.sqrt(dt) * (L @ noise))
        return x

    # Accelerometer: OU + gravity on z
    acc = _ou_path(sigma)
    acc[:, 2] += gravity

    # Gyroscope: coupled OU, lower amplitude
    gyro = _ou_path(sigma * gyro_scale)

    # Timestamps with realistic jitter
    ts = [int(j * 1e9 / hz + rng.normal(0, jitter_ns)) for j in range(n_samples)]

    return {
        "timestamps_ns": ts,
        "accel":         acc.tolist(),
        "gyro":          gyro.tolist(),
    }


def generate_batch(
    n_windows: int,
    seed_start: int = 0,
    **kwargs,
) -> List[Dict]:
    """Generate n_windows OU windows with sequential seeds."""
    return [
        generate_window(seed=seed_start + i, **kwargs)
        for i in range(n_windows)
    ]
