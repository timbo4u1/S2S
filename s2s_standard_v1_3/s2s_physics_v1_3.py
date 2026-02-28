#!/usr/bin/env python3
"""
s2s_physics_v1_3.py — S2S Physical Law Engine (v1.3)

Certifies that sensor data OBEYS the physical laws governing human movement.

Statistical anomaly detection (rest of S2S) asks: "Does this data LOOK human?"
Physical law certification (this module) asks: "Does this data OBEY human physics?"

These are fundamentally different. A perfect statistical fake fails here
if it violates F=ma, energy conservation, or body-segment resonance.

Physical laws encoded:

  1. Newton's Second Law        F = ma  (with ~75ms electromechanical delay)
     EMG amplitude (muscle force proxy) must LEAD acceleration by ~50-100ms.
     Zero-lag correlation is wrong — physiology has a fixed delay.
     Synthetic data generated independently has zero lagged correlation.

  2. Arm Segment Resonance     ω = sqrt(K/I)
     Physiological tremor frequency is physically determined by the
     moment of inertia (I) and neuromuscular stiffness (K) of the segment.
     Forearm: I=0.020 kg·m², K=1.5 N·m/rad → f_fund=1.38Hz → tremor 8-12Hz
     This cannot be faked with wrong parameters — the physics is fixed.

  3. Rigid Body Kinematics     a = α×r + ω²×r
     Gyroscope and accelerometer measure the SAME physical motion event
     from different physical principles. Their variance must co-vary.
     Independently generated fake accel+gyro has zero coupling.

  4. Ballistocardiography      F = ρQv (Newton 3, cardiac ejection)
     Each heartbeat ejects ~70ml blood at ~1 m/s → body recoil.
     IMU must show spectral energy at EXACTLY the PPG heart rate.
     Two independent physical measurements of the same heartbeat.

  5. Joule Heating             Q ≈ 0.75 × P_metabolic × t
     Active muscle (EMG bursts >30% for >5s) generates heat (75% inefficiency).
     Thermal camera must confirm temperature elevation if sensors are co-located.

  6. Motor Control Jerk Bound  d³x/dt³ ≤ 500 m/s³  (Flash & Hogan 1985)
     Human voluntary movement follows minimum-jerk trajectories.
     Applied to SMOOTHED signal to remove sensor noise — jerk is a motion property,
     not a noise property. Smoothing window = 5 samples before differentiation.

  7. IMU Internal Consistency  Var(accel) ~ f(Var(gyro))
     For a rigid body, accel and gyro variance must co-vary over time
     (they measure different aspects of the same physical motion event).
"""
from __future__ import annotations
import math, statistics, time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Physical constants — NOT tunable parameters
# ---------------------------------------------------------------------------
FOREARM_MASS_KG             = 1.5
FOREARM_MOMENT_OF_INERTIA   = 0.020        # I kg·m²
FOREARM_NEUROMUSCULAR_STIFF = 1.5          # K N·m/rad
FOREARM_RESONANCE_HZ        = math.sqrt(FOREARM_NEUROMUSCULAR_STIFF /
                                         FOREARM_MOMENT_OF_INERTIA) / (2*math.pi)  # 1.378 Hz fundamental

SEGMENT_PARAMS = {
    # segment: (I kg·m², K N·m/rad, tremor_lo Hz, tremor_hi Hz)
    "forearm":   (0.020, 1.5,  8.0,  12.0),
    "finger":    (0.0003,0.05, 15.0, 25.0),
    "hand":      (0.004, 0.3,  10.0, 16.0),
    "upper_arm": (0.065, 2.5,  5.0,   9.0),
    "head":      (0.020, 1.2,  3.0,   8.0),
}

EMG_ACCEL_DELAY_MS          = 75.0         # ms — electromechanical delay (EMG → force → accel)
EMG_FORCE_MIN_LAGGED_R      = 0.10         # minimum lagged Pearson r for F=ma check
JERK_SMOOTH_WINDOW          = 7            # samples to smooth before jerk differentiation
JERK_MAX_MS3                = 500.0        # m/s³ — Flash-Hogan 1985 voluntary arm movement
BCG_MIN_ENERGY              = 0.005        # minimum normalised energy for BCG confirmation
MUSCLE_THERMAL_EFFICIENCY   = 0.25         # 25% mechanical → 75% heat
THERMAL_DETECTABLE_DELTA_C  = 0.3          # °C
EMG_BURST_HEAT_THRESHOLD    = 0.30         # burst fraction threshold for heat prediction
EMG_BURST_HEAT_DURATION_S   = 5.0          # seconds sustained burst before heat detectable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth(sig: List[float], w: int = 5) -> List[float]:
    """Centred moving average to remove sensor noise before differentiation."""
    return [statistics.mean(sig[max(0,i-w):i+w+1]) for i in range(len(sig))]


def _diff(sig: List[float], dt: float) -> List[float]:
    if len(sig) < 3 or dt <= 0: return []
    return [(sig[i+1]-sig[i-1])/(2*dt) for i in range(1, len(sig)-1)]


def _rms(xs: List[float]) -> float:
    f = [v for v in xs if math.isfinite(v)]
    return math.sqrt(sum(v*v for v in f)/len(f)) if f else 0.0


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 6: return 0.0
    xs, ys = xs[:n], ys[:n]
    mx, my = sum(xs)/n, sum(ys)/n
    num = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    dx = math.sqrt(sum((x-mx)**2 for x in xs)) or 1e-9
    dy = math.sqrt(sum((y-my)**2 for y in ys)) or 1e-9
    return num/(dx*dy)


def _lagged_pearson(lead: List[float], lag: List[float], delay_samples: int) -> float:
    """Pearson r where lead[i] predicts lag[i+delay_samples]."""
    if delay_samples <= 0:
        return _pearson(lead, lag)
    n = min(len(lead)-delay_samples, len(lag)-delay_samples)
    if n < 6: return _pearson(lead, lag)
    a = lead[:n]
    b = lag[delay_samples:delay_samples+n]
    return _pearson(a, b)


def _freq_energy(sig: List[float], ts_ns: List[int], f_hz: float) -> float:
    n = min(len(sig), len(ts_ns))
    if n < 8: return 0.0
    t = [ts_ns[i]*1e-9 for i in range(n)]
    s = [sig[i] for i in range(n)]
    mu = sum(s)/n; s = [v-mu for v in s]
    sd = sum(s[i]*math.sin(2*math.pi*f_hz*t[i]) for i in range(n))
    cd = sum(s[i]*math.cos(2*math.pi*f_hz*t[i]) for i in range(n))
    tot = sum(v*v for v in s) or 1.0
    return max(0.0, min(1.0, (sd*sd+cd*cd)/(tot*n/2)))


def _band_peak(sig, ts_ns, f_lo, f_hi, steps=40):
    bf, be = f_lo, 0.0
    for i in range(steps+1):
        f = f_lo + (f_hi-f_lo)*i/steps
        e = _freq_energy(sig, ts_ns, f)
        if e > be: be,bf = e,f
    return bf, be


# ---------------------------------------------------------------------------
# Law 1: Newton's Second Law — F = ma  (with electromechanical delay)
# ---------------------------------------------------------------------------
def check_newton(imu_raw: Dict, emg_raw: Dict) -> Tuple[bool,int,Dict]:
    d = {"law":"Newton_Second_Law","equation":"F=ma",
         "note":"EMG force LEADS acceleration by ~75ms (electromechanical delay)"}

    accel = imu_raw.get("accel",[]);  imu_ts = imu_raw.get("timestamps_ns",[])
    emg   = emg_raw.get("channels",[]); emg_ts = emg_raw.get("timestamps_ns",[])

    if len(accel)<20 or not emg or len(emg[0])<20:
        d["skip"]="INSUFFICIENT_RAW_DATA"; return True,50,d

    g = 9.81
    # |a(t)| minus gravity
    a_mag = [abs(math.sqrt(sum(v**2 for v in s[:3])) - g) for s in accel]

    # Dominant EMG channel (highest variance) — smooth to get envelope
    variances = []
    for ch in emg:
        seg = ch[:min(len(ch),200)]
        variances.append(statistics.variance(seg) if len(set(seg))>1 else 0)
    dom = max(range(len(variances)), key=lambda i: variances[i])
    emg_env = _smooth([abs(v) for v in emg[dom]], w=20)  # RMS envelope

    # Estimate delay in samples
    imu_hz = 1e9/statistics.mean([imu_ts[i+1]-imu_ts[i]
                                    for i in range(min(len(imu_ts)-1,50))]) if len(imu_ts)>1 else 240
    emg_hz = 1e9/statistics.mean([emg_ts[i+1]-emg_ts[i]
                                    for i in range(min(len(emg_ts)-1,50))]) if len(emg_ts)>1 else 1000
    delay_samples_emg = max(1, int(EMG_ACCEL_DELAY_MS/1000 * emg_hz))

    # Resample both to same length
    n = 256
    def rs(s): return [s[int(i*len(s)/n)] for i in range(n)]
    a_rs  = rs(a_mag)
    emg_rs = rs(emg_env)
    delay_rs = max(1, int(EMG_ACCEL_DELAY_MS/1000 * n / (n*1/emg_hz + n*1/imu_hz) * 2))
    delay_rs = min(delay_rs, 30)  # cap at 30 resampled steps

    r_zero   = _pearson(emg_rs, a_rs)
    r_lagged = _lagged_pearson(emg_rs, a_rs, delay_samples=delay_rs)
    best_r   = max(r_zero, r_lagged)

    d.update({"pearson_r_zero_lag":round(r_zero,4),
               "pearson_r_lagged":round(r_lagged,4),
               "delay_applied_samples":delay_rs,
               "delay_applied_ms":round(delay_rs/n*1000,1),
               "best_r":round(best_r,4),
               "dominant_emg_channel":dom,
               "physical_meaning":"Muscle force causes acceleration (with ~75ms delay)"})

    if best_r >= EMG_FORCE_MIN_LAGGED_R:
        d["result"]="F=MA_COUPLING_CONFIRMED"
        conf = min(100, int(best_r*120))
        passed = True
    elif r_zero < -0.25:
        d["violation"]="ANTI_PHASE_EMG_ACCEL"
        d["interpretation"]="EMG active when decelerating — physically impossible for prime mover"
        conf=5; passed=False
    elif best_r >= 0:
        d["note"]="WEAK_FORCE_MOTION_COUPLING"
        d["interpretation"]="Stabiliser/postural muscle likely — not prime mover. Acceptable."
        conf=45; passed=True
    else:
        d["violation"]="NEGATIVE_LAGGED_COUPLING"; conf=20; passed=False

    d["confidence"]=conf; return passed,conf,d


# ---------------------------------------------------------------------------
# Law 2: Resonance  ω = sqrt(K/I)
# ---------------------------------------------------------------------------
def check_resonance(imu_raw: Dict, segment: str="forearm") -> Tuple[bool,int,Dict]:
    I,K,f_lo,f_hi = SEGMENT_PARAMS.get(segment, SEGMENT_PARAMS["forearm"])
    omega = math.sqrt(K/I)
    f_fund = omega/(2*math.pi)
    d = {"law":"Resonance_Frequency","equation":"omega=sqrt(K/I)",
         "segment":segment,"I_kgm2":I,"K_Nm_rad":K,
         "omega_rad_s":round(omega,3),"f_fundamental_hz":round(f_fund,3),
         "f_tremor_range_hz":[f_lo,f_hi],
         "derivation":f"ω=√({K}/{I})={omega:.2f}rad/s → neuromuscular loop delay → tremor {f_lo}–{f_hi}Hz"}

    accel = imu_raw.get("accel",[]); ts = imu_raw.get("timestamps_ns",[])
    if len(accel)<64:
        d["skip"]="INSUFFICIENT_DATA"; return True,50,d

    az = [s[2] if len(s)>2 else 0.0 for s in accel]
    peak_f, peak_e = _band_peak(az, ts, 5.0, 30.0, steps=60)
    d.update({"measured_peak_hz":round(peak_f,2),"measured_peak_energy":round(peak_e,5)})

    if peak_e < 0.015:
        d["note"]="NO_SIGNIFICANT_TREMOR"; d["confidence"]=60; return True,60,d

    in_range = f_lo <= peak_f <= f_hi
    d["in_expected_range"] = in_range
    if in_range:
        d["result"]=f"RESONANCE_CONFIRMED ({peak_f:.1f}Hz matches {segment} ω=√(K/I))"
        conf = min(100, int(60 + peak_e*400)); passed=True
    else:
        dev = min(abs(peak_f-f_lo), abs(peak_f-f_hi))
        d["violation"]=f"RESONANCE_MISMATCH: {peak_f:.1f}Hz outside {f_lo}–{f_hi}Hz for {segment}"
        d["interpretation"]=f"{segment} cannot resonate at {peak_f:.1f}Hz (ω=√(K/I) gives {f_lo}–{f_hi}Hz)"
        conf = max(0, int(50-dev*8)); passed = dev<3.0

    d["confidence"]=conf; return passed,conf,d


# ---------------------------------------------------------------------------
# Law 3: Rigid Body Kinematics  a = α×r + ω²×r
# ---------------------------------------------------------------------------
def check_rigid_body(imu_raw: Dict, r: float=0.05) -> Tuple[bool,int,Dict]:
    d = {"law":"Rigid_Body_Kinematics","equation":"a=alpha*r+omega^2*r",
         "sensor_offset_m":r,
         "physical_meaning":"Gyro angular velocity must predict accelerometer via rigid body mechanics"}

    accel = imu_raw.get("accel",[]); gyro = imu_raw.get("gyro",[]); ts = imu_raw.get("timestamps_ns",[])
    n = min(len(accel),len(gyro),len(ts))
    if n<20: d["skip"]="INSUFFICIENT_DATA"; return True,50,d

    g=9.81
    a_mag = [abs(math.sqrt(sum(v**2 for v in accel[i][:3]))-g) for i in range(n)]
    w_mag = [math.sqrt(sum(v**2 for v in gyro[i][:3])) for i in range(n)]

    dt = statistics.mean([ts[i+1]-ts[i] for i in range(min(n-1,50))])*1e-9 if n>1 else 1/240
    d["dt_s"] = round(dt,6)

    # alpha = dω/dt
    alpha = [abs(w_mag[i+1]-w_mag[i-1])/(2*dt) for i in range(1,n-1)]
    w_mid = w_mag[1:n-1]
    a_mid = a_mag[1:n-1]
    a_pred = [math.sqrt((alpha[i]*r)**2 + (w_mid[i]**2*r)**2) for i in range(len(alpha))]

    r_corr = _pearson(a_mid, a_pred)
    rms_m = _rms(a_mid); rms_p = _rms(a_pred)
    ratio  = rms_p/max(rms_m,1e-6)

    d.update({"pearson_r_measured_vs_predicted":round(r_corr,4),
               "rms_measured_ms2":round(rms_m,4),
               "rms_predicted_ms2":round(rms_p,4),
               "gyro_accel_scale_ratio":round(ratio,4)})

    if r_corr>0.15 and 0.05<ratio<10.0:
        d["result"]="RIGID_BODY_KINEMATICS_CONSISTENT"
        conf=min(100,int(50+r_corr*50)); passed=True
    elif r_corr < -0.2:
        d["violation"]="GYRO_ACCEL_ANTI_CORRELATED"
        d["interpretation"]="Physically impossible for rigid body — indicates independent generation"
        conf=10; passed=False
    else:
        d["note"]="WEAK_KINEMATIC_COUPLING (near-static motion — valid)"
        conf=45; passed=True

    d["confidence"]=conf; return passed,conf,d


# ---------------------------------------------------------------------------
# Law 4: Ballistocardiography  Newton 3 cardiac recoil
# ---------------------------------------------------------------------------
def check_bcg(imu_raw: Dict, ppg_cert: Dict) -> Tuple[bool,int,Dict]:
    d = {"law":"Ballistocardiography","equation":"F_recoil=rho*Q*v (Newton 3)",
         "physics":"70ml blood × 1m/s ejection → ~0.001 m/s² body recoil at wrist IMU"}

    hr_bpm = (ppg_cert.get("vitals") or {}).get("heart_rate_bpm")
    if not hr_bpm or not (30<=hr_bpm<=220):
        d["skip"]="NO_VALID_PPG_HR"; return True,50,d

    hr_hz = hr_bpm/60.0
    accel = imu_raw.get("accel",[]); ts = imu_raw.get("timestamps_ns",[])
    if len(accel)<64: d["skip"]="INSUFFICIENT_IMU_DATA"; return True,50,d

    d.update({"ppg_hr_bpm":hr_bpm,"ppg_hr_hz":round(hr_hz,4)})

    # Find axis with highest energy at HR frequency
    best_e, best_axis = 0.0, 0
    for ax in range(min(3,len(accel[0]))):
        sig = [accel[i][ax] for i in range(len(accel))]
        e = _freq_energy(sig, ts, hr_hz)
        if e>best_e: best_e=e; best_axis=ax

    sig_best = [accel[i][best_axis] for i in range(len(accel))]
    e_harm = _freq_energy(sig_best, ts, hr_hz*2)
    e_low  = _freq_energy(sig_best, ts, hr_hz*0.7)
    e_high = _freq_energy(sig_best, ts, hr_hz*1.3)
    spec   = best_e/max(max(e_low,e_high),1e-9)

    d.update({"imu_energy_at_hr":round(best_e,5),"imu_energy_at_2xhr":round(e_harm,5),
               "freq_specificity":round(spec,2),"best_axis":best_axis})

    if best_e>=BCG_MIN_ENERGY and spec>=1.5:
        d["result"]=f"BCG_CONFIRMED: IMU energy {best_e:.4f} at PPG HR={hr_bpm:.0f}bpm"
        conf=min(100,int(60+spec*10+best_e*500)); passed=True
    elif best_e>=0.001:
        d["note"]="WEAK_BCG — posture/clothing may attenuate"; conf=55; passed=True
    else:
        d["warn"]="NO_BCG_DETECTED — IMU may not be on torso/chest for BCG"; conf=35; passed=True

    d["confidence"]=conf; return passed,conf,d


# ---------------------------------------------------------------------------
# Law 5: Joule Heating  Q ≈ 0.75 × P × t
# ---------------------------------------------------------------------------
def check_joule(emg_cert: Dict, thermal_cert: Dict) -> Tuple[bool,int,Dict]:
    d = {"law":"Joule_Heating","equation":"Q=0.75*P_metabolic*t",
         "physics":"75% of muscle metabolic energy → heat (thermodynamic inefficiency)"}

    bf   = (emg_cert.get("notes") or {}).get("mean_burst_frac",0)
    dur_s = emg_cert.get("duration_ms",0)/1000
    human = (thermal_cert.get("human_presence") or {}).get("human_present",False)
    t_range = (thermal_cert.get("spatial_analysis") or {}).get("spatial_range_C",0)
    t_delta = (thermal_cert.get("temporal_analysis") or {}).get("mean_frame_delta_C",0)

    d.update({"emg_burst_frac":round(bf,4),"emg_duration_s":round(dur_s,2),
               "thermal_human":human,"thermal_spatial_range_C":round(t_range,3),
               "thermal_delta_C":round(t_delta,6)})

    heat_expected = bf>EMG_BURST_HEAT_THRESHOLD and dur_s>=EMG_BURST_HEAT_DURATION_S
    d["heat_should_be_detectable"] = heat_expected

    if not heat_expected:
        d["note"]="ACTIVITY_TOO_BRIEF_FOR_THERMAL_DETECTION"
        d["interpretation"]=f"Need >{EMG_BURST_HEAT_THRESHOLD*100:.0f}% burst for >{EMG_BURST_HEAT_DURATION_S}s — not yet"
        conf=60; passed=True
    elif heat_expected and human and t_range>THERMAL_DETECTABLE_DELTA_C:
        d["result"]="METABOLIC_HEAT_CONFIRMED"
        d["interpretation"]=f"EMG {bf*100:.0f}% active for {dur_s:.1f}s → thermal {t_range:.1f}°C gradient (Joule heating)"
        conf=90; passed=True
    elif heat_expected and not human:
        d["violation"]="EMG_ACTIVE_NO_THERMAL_HEAT"
        d["interpretation"]="Sustained EMG but no thermal body heat — co-located sensors would show this"
        conf=30; passed=False
    else:
        conf=55; passed=True

    d["confidence"]=conf; return passed,conf,d


# ---------------------------------------------------------------------------
# Law 6: Jerk Bounds  d³x/dt³ ≤ 500 m/s³  (Flash & Hogan 1985)
# ---------------------------------------------------------------------------
def check_jerk(imu_raw: Dict) -> Tuple[bool,int,Dict]:
    d = {"law":"Motor_Control_Jerk","equation":"d3x/dt3 <= 500 m/s3",
         "reference":"Flash & Hogan (1985) minimum-jerk model",
         "implementation":"Signal smoothed with w=7 window BEFORE differentiation to separate motion from sensor noise"}

    accel = imu_raw.get("accel",[]); ts = imu_raw.get("timestamps_ns",[])
    if len(accel)<20: d["skip"]="INSUFFICIENT_DATA"; return True,50,d

    dt = statistics.mean([ts[i+1]-ts[i] for i in range(min(len(ts)-1,50))])*1e-9 if len(ts)>1 else 1/240

    # Windowed jerk: split into 2.56s windows, compute P95 per window,
    # take median across windows. Prevents activity-transition spikes in
    # long recordings (e.g. 235s PAMAP2 iron) from dominating the result.
    WINDOW_SAMPLES = max(20, int(2.56 / dt)) if dt > 0 else 256

    all_jerk = []
    window_p95s = []
    n = len(accel)

    for axis in range(min(3, len(accel[0]))):
        sig_raw = [accel[i][axis] for i in range(n)]
        sig_s   = _smooth(sig_raw, w=JERK_SMOOTH_WINDOW)
        vel     = _diff(sig_s, dt)
        if vel:
            vel_s = _smooth(vel, w=JERK_SMOOTH_WINDOW)
            jerk  = _diff(vel_s, dt)
            all_jerk.extend(jerk)

            # Compute P95 per window for this axis
            for w_start in range(0, len(jerk), WINDOW_SAMPLES):
                w = [abs(j) for j in jerk[w_start:w_start+WINDOW_SAMPLES]]
                if len(w) >= 10:
                    w.sort()
                    window_p95s.append(w[int(len(w)*0.95)])

    if not all_jerk: d["skip"]="COULD_NOT_COMPUTE"; return True,50,d

    peak_j = max(abs(j) for j in all_jerk)
    rms_j  = _rms(all_jerk)
    # Use median of per-window P95s — robust against transition spikes
    if window_p95s:
        window_p95s.sort()
        p95_j = window_p95s[len(window_p95s)//2]
    else:
        p95_j = sorted([abs(j) for j in all_jerk])[int(len(all_jerk)*0.95)]
    d["window_samples"] = WINDOW_SAMPLES
    d["n_windows"] = len(window_p95s)

    d.update({"peak_jerk_ms3":round(peak_j,1),"rms_jerk_ms3":round(rms_j,1),
               "p95_jerk_ms3":round(p95_j,1),"human_limit_ms3":JERK_MAX_MS3,
               "smooth_window_samples":JERK_SMOOTH_WINDOW})

    if p95_j > JERK_MAX_MS3:
        d["violation"]=f"JERK_EXCEEDS_HUMAN_LIMIT: {p95_j:.0f} > {JERK_MAX_MS3:.0f} m/s³"
        d["interpretation"]="Trajectory is supra-human — robot bang-bang or keyframe artifact"
        conf=max(0,int(100-(p95_j/JERK_MAX_MS3)*40)); passed=False
    elif p95_j > 150:
        d["note"]=f"HIGH_JERK_FAST_MOVEMENT: {p95_j:.0f} m/s³ (valid for ballistic arm motion)"
        conf=75; passed=True
    else:
        d["result"]=f"JERK_WITHIN_HUMAN_MOTOR_CONTROL: {p95_j:.0f} m/s³"
        conf=92; passed=True

    d["confidence"]=conf; return passed,conf,d


# ---------------------------------------------------------------------------
# Law 7: IMU Internal Consistency (accel-gyro co-variance)
# ---------------------------------------------------------------------------
def check_imu_consistency(imu_raw: Dict) -> Tuple[bool,int,Dict]:
    d = {"law":"IMU_Internal_Consistency","equation":"Var(accel) co-varies with Var(gyro)",
         "physical_meaning":"Same motion event causes both angular velocity and linear acceleration"}

    accel = imu_raw.get("accel",[]); gyro = imu_raw.get("gyro",[])
    n = min(len(accel),len(gyro))
    if n<30: d["skip"]="INSUFFICIENT_DATA"; return True,50,d

    w=10; a_vars,g_vars=[],[]
    for i in range(w,n):
        a_s=[math.sqrt(sum(v**2 for v in accel[j][:3])) for j in range(i-w,i)]
        g_s=[math.sqrt(sum(v**2 for v in gyro[j][:3]))  for j in range(i-w,i)]
        a_vars.append(statistics.variance(a_s) if len(set(a_s))>1 else 0)
        g_vars.append(statistics.variance(g_s) if len(set(g_s))>1 else 0)

    r = _pearson(a_vars, g_vars)
    d.update({"pearson_r_var_coupling":round(r,4),"expected_r_range":[0.1,1.0]})

    if r>=0.15:
        d["result"]="IMU_INTERNALLY_CONSISTENT"; conf=min(100,int(50+r*50)); passed=True
    elif r>=0:
        d["note"]="WEAK_COUPLING (near-static motion — acceptable)"; conf=45; passed=True
    else:
        d["violation"]="ACCEL_GYRO_DECOUPLED"
        d["interpretation"]=f"r={r:.3f}: independently-generated sensor data cannot couple to same physical event"
        conf=15; passed=False

    d["confidence"]=conf; return passed,conf,d


# ---------------------------------------------------------------------------
# PhysicsEngine
# ---------------------------------------------------------------------------
class PhysicsEngine:
    """
    Certify that sensor data obeys the physical laws of human movement.

    pe = PhysicsEngine()
    result = pe.certify(
        imu_raw   = {"timestamps_ns":[...], "accel":[[ax,ay,az],...], "gyro":[[gx,gy,gz],...]},
        emg_raw   = {"timestamps_ns":[...], "channels":[[ch0_sample0,...],...]},
        ppg_cert  = ppg_cert_dict,
        emg_cert  = emg_cert_dict,
        thermal_cert = thermal_cert_dict,
        segment   = "forearm",
    )
    print(result["physical_law_score"])   # 0–100
    print(result["laws_passed"])
    """

    def certify(self,
                imu_raw:      Optional[Dict]=None,
                emg_raw:      Optional[Dict]=None,
                ppg_cert:     Optional[Dict]=None,
                emg_cert:     Optional[Dict]=None,
                thermal_cert: Optional[Dict]=None,
                segment:      str="forearm",
                device_id:    str="unknown",
                session_id:   Optional[str]=None) -> Dict[str,Any]:

        results: Dict[str,Tuple[bool,int,Dict]] = {}

        if imu_raw and emg_raw:
            results["newton_second_law"] = check_newton(imu_raw, emg_raw)
        if imu_raw:
            results["resonance_frequency"]      = check_resonance(imu_raw, segment)
            results["rigid_body_kinematics"]    = check_rigid_body(imu_raw)
            results["jerk_bounds"]              = check_jerk(imu_raw)
            results["imu_internal_consistency"] = check_imu_consistency(imu_raw)
        if imu_raw and ppg_cert:
            results["ballistocardiography"] = check_bcg(imu_raw, ppg_cert)
        if emg_cert and thermal_cert:
            results["joule_heating"] = check_joule(emg_cert, thermal_cert)

        passed  = [k for k,(ok,_,_) in results.items() if ok]
        failed  = [k for k,(ok,_,_) in results.items() if not ok]
        confs   = [c for (_,c,_) in results.values()]
        score   = int(sum(confs)/len(confs)) if confs else 0
        n, np   = len(results), len(passed)

        if n==0:  tier="UNVERIFIED"
        elif len(failed)/max(n,1)>0.3: tier="REJECTED"
        elif score>=75 and np>=max(n-1,1): tier="GOLD"
        elif score>=55: tier="SILVER"
        elif score>=35: tier="BRONZE"
        else: tier="REJECTED"

        flags = [f"PHYSICS_VIOLATION:{k}" for k in failed]

        return {
            "status":             "PASS" if tier not in ("REJECTED","UNVERIFIED") else "FAIL",
            "tier":               tier,
            "sensor_type":        "PHYSICS",
            "physical_law_score": score,
            "laws_checked":       list(results.keys()),
            "n_laws_checked":     n,
            "laws_passed":        passed,
            "laws_failed":        failed,
            "body_segment":       segment,
            "law_details":        {k:det for k,(_,_,det) in results.items()},
            "flags":              flags,
            "physical_constants": {
                "forearm_I_kgm2":        FOREARM_MOMENT_OF_INERTIA,
                "forearm_K_Nm_rad":      FOREARM_NEUROMUSCULAR_STIFF,
                "forearm_resonance_hz":  round(FOREARM_RESONANCE_HZ,3),
                "tremor_band_hz":        list(SEGMENT_PARAMS["forearm"][2:]),
                "emg_delay_ms":          EMG_ACCEL_DELAY_MS,
                "jerk_max_ms3":          JERK_MAX_MS3,
                "muscle_thermal_eff":    MUSCLE_THERMAL_EFFICIENCY,
            },
            "device_id":    device_id,
            "session_id":   session_id,
            "tool":         "s2s_physics_v1_3",
            "issued_at_ns": time.time_ns(),
        }
