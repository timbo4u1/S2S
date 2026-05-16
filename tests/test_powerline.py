import math, pytest

def test_powerline_clean():
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_powerline_interference
    n, hz = 256, 200
    acc = [[math.sin(i*0.1)*2, math.cos(i*0.1)*1.5, 9.81+math.sin(i*0.05)] for i in range(n)]
    ts  = [int(i*1e9/hz) for i in range(n)]
    passed, conf, d = check_powerline_interference({"timestamps_ns": ts, "accel": acc})
    assert passed, f"Clean biological signal flagged as powerline: {d}"

def test_powerline_50hz_detected():
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_powerline_interference
    n, hz = 256, 200
    acc = [[math.sin(i*0.1)*2 + 5.0*math.sin(2*math.pi*50*i/hz),
            math.cos(i*0.1)*1.5, 9.81] for i in range(n)]
    ts  = [int(i*1e9/hz) for i in range(n)]
    passed, conf, d = check_powerline_interference({"timestamps_ns": ts, "accel": acc})
    assert not passed, f"50Hz powerline not detected: {d}"
    assert "50" in d.get("result",""), f"Wrong freq in result: {d}"

def test_powerline_skips_low_hz():
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_powerline_interference
    n = 256
    acc = [[0.1*i, 0.0, 9.81] for i in range(n)]
    ts  = [int(i*1e9/100) for i in range(n)]   # 100Hz < 125 threshold
    passed, conf, d = check_powerline_interference({"timestamps_ns": ts, "accel": acc})
    assert passed and "skip" in d, f"Should skip at 100Hz: {d}"
