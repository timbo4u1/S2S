import math

def test_splice_clean():
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_intra_window_splice
    n = 256
    acc = [[math.sin(i*0.1)*2, math.cos(i*0.1)*1.5, 9.81+math.sin(i*0.05)] for i in range(n)]
    ts  = [int(i*5e6) for i in range(n)]
    passed, conf, d = check_intra_window_splice({"timestamps_ns": ts, "accel": acc})
    assert passed, f"Clean signal flagged as splice: {d}"

def test_splice_detected():
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_intra_window_splice
    # First half: rest. Second half: vigorous motion with very different mean magnitude
    n = 256
    half = n // 2
    acc  = [[0.1, 0.0, 9.81]] * half          # rest  mean_mag ≈ 9.81
    acc += [[15.0, 12.0, 20.0]] * half         # vigorous mean_mag ≈ 27.0, diff > 8.0
    ts   = [int(i*5e6) for i in range(n)]
    passed, conf, d = check_intra_window_splice({"timestamps_ns": ts, "accel": acc})
    assert not passed, f"Splice not detected: {d}"
    assert "SPLICE_DETECTED" in d.get("result",""), f"Wrong result: {d}"
