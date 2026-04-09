"""
s2s-certify: Physics certification layer for human motion sensor data.
Detects biological origin, certifies signal quality before ML training.
"""

from .s2s_physics_v1_3 import PhysicsEngine

__all__ = ["PhysicsEngine"]
__version__ = "1.6.2"

from s2s_standard_v1_3.s2s_pipeline import S2SPipeline
from s2s_standard_v1_3.s2s_safety_gate import RealTimeSafetyGate

__all__ = ["PhysicsEngine", "S2SPipeline", "RealTimeSafetyGate"]
