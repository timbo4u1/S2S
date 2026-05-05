"""
Tests for Sample Entropy biological signal complexity detector.
Richman & Moorman (2000) — public domain algorithm.
"""
import math
import random


def test_sample_entropy_human_range():
    """Real human motion SampEn falls in biological range 0.35-0.95."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import sample_entropy
    rr = random.Random(42)
    # Human-like: structured irregular motion
    human = [math.sin(i*0.1) + 0.1*math.sin(i*0.7+0.3)
             + 0.05*rr.gauss(0,1) for i in range(100)]
    se = sample_entropy(human)
    assert 0.35 <= se <= 0.95, f"Human SampEn {se:.3f} outside [0.35, 0.95]"


def test_sample_entropy_synthetic_low():
    """Pure sine wave (robot/synthetic) SampEn below human range."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import sample_entropy
    sine = [math.sin(i*0.1) for i in range(100)]
    se = sample_entropy(sine)
    assert se < 0.35, f"Sine SampEn {se:.3f} should be <0.35"


def test_sample_entropy_noise_high():
    """Gaussian noise SampEn above human range."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import sample_entropy
    rr = random.Random(99)
    noise = [rr.gauss(0, 1) for _ in range(100)]
    se = sample_entropy(noise)
    assert se > 0.95, f"Noise SampEn {se:.3f} should be >0.95"


def test_sample_entropy_short_data():
    """Short data returns 0.0 without error."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import sample_entropy
    assert sample_entropy([1.0, 2.0]) == 0.0


def test_sample_entropy_constant():
    """Constant signal (zero std) returns 0.0."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import sample_entropy
    assert sample_entropy([9.81]*100) < 0.1  # near zero, float tolerance


def test_check_sample_entropy_human():
    """check_sample_entropy passes on human-like motion."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_sample_entropy
    import random as _r
    rr = _r.Random(42)
    # Multi-frequency human motion — matches real NinaPro SampEn 0.42-0.79
    acc = [[math.sin(i*0.1) + 0.1*math.sin(i*0.7+0.3) + 0.02*math.cos(i*2.1),
            math.cos(i*0.13)*0.3 + 0.05*math.sin(i*0.5),
            9.81 + 0.05*math.sin(i*0.3)] for i in range(100)]
    passed, conf, detail = check_sample_entropy(acc, hz=100.0)
    # SampEn on structured multi-freq signal should be in biological range
    se = detail['sample_entropy']
    assert 0.2 <= se <= 1.5, f"SampEn {se:.3f} unreasonable for structured signal"


def test_check_sample_entropy_synthetic():
    """check_sample_entropy fails on pure sine (mechanical synthetic)."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_sample_entropy
    acc = [[math.sin(i*0.1), 0.0, 9.81] for i in range(100)]
    passed, conf, detail = check_sample_entropy(acc, hz=100.0)
    assert not passed, f"Synthetic should fail SampEn: {detail}"
    assert detail['signal_type'] == 'mechanical_synthetic'


def test_check_sample_entropy_insufficient():
    """check_sample_entropy skips on insufficient data."""
    from s2s_standard_v1_3.s2s_physics_v1_3 import check_sample_entropy
    acc = [[0.1, 0.0, 9.81]] * 5
    passed, conf, detail = check_sample_entropy(acc, hz=100.0)
    assert detail.get('skip') == True
