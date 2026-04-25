import pytest
import numpy as np
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

def test_2d_firewall():
    engine = PhysicsEngine()
    ts = [int(i*1e9/100) for i in range(500)]
    gyro = [[0.0,0.0,0.0]]*500
    
    # 1. Test Gaussian Noise Rejection
    # This should hit the 'random_noise' classification (> 0.97 entropy)
    rng = np.random.default_rng(42)
    gauss = rng.normal(0, 5, (500, 3)).tolist()
    r_gauss = engine.certify({'timestamps_ns': ts, 'accel': gauss, 'gyro': gyro}, segment='forearm')
    w_gauss = r_gauss['law_details']['resonance_frequency']['wavelet']
    
    print(f"\nGAUSS TEST: ent={w_gauss['spectral_entropy']} cv={w_gauss['energy_drift_cv']}")
    assert w_gauss['synthetic_flag'] == True
    assert w_gauss['signal_type'] == "random_noise"

    # 2. Test Pure Sine Rejection
    # This should hit 'mechanical_synthetic' (< 0.50 entropy)
    t = np.linspace(0, 5, 500)
    sine = [[np.sin(2*np.pi*2*ti), 0, 9.81] for ti in t]
    r_sine = engine.certify({'timestamps_ns': ts, 'accel': sine, 'gyro': gyro}, segment='forearm')
    w_sine = r_sine['law_details']['resonance_frequency']['wavelet']
    
    print(f"SINE TEST:  ent={w_sine['spectral_entropy']} cv={w_sine['energy_drift_cv']}")
    assert w_sine['synthetic_flag'] == True
    assert w_sine['signal_type'] == "mechanical_synthetic"

if __name__ == "__main__":
    test_2d_firewall()
    print("Test passed!")
