"""
S2S Quick Start — physics certification in 30 lines.
pip install s2s-certify
"""
import math, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine, audit_report

engine = PhysicsEngine()

# 1 second of 200Hz forearm IMU data
n = 200
ts  = [int(i * 1e9 / 200) for i in range(n)]
acc = [[0.1 * math.sin(2 * math.pi * 5 * i / 200), 0.0, 9.81] for i in range(n)]
gyro = [[0.01, 0.0, 0.0]] * n

result = engine.certify(
    {'timestamps_ns': ts, 'accel': acc, 'gyro': gyro},
    segment='forearm'
)

report = audit_report(result)

print(f"Tier:           {result['tier']}")
print(f"Score:          {result['physical_law_score']}/100")
print(f"Verdict:        {report['verdict']}")
print(f"Recommendation: {report['recommendation']}")
print(f"Laws passed:    {result['laws_passed']}")

if report['issues']:
    print("\nIssues:")
    for issue in report['issues']:
        print(f"  ⚠️  {issue['law']}: {issue['message']}")
        print(f"     Fix: {issue['fix']}")
