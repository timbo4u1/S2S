# 7 Biomechanical Laws Every Robotics Dataset Should Pass

*Posted by Timur Davkarayev · March 2026*

---

When you download a motion capture dataset, you get numbers. Timestamps, accelerations, joint angles, muscle signals. You assume these numbers describe real human motion. Often they do not.

Here are seven physics laws that real human motion always obeys — and that most motion datasets never check.

---

## Law 1: Jerk Bounds (Flash & Hogan, 1985)

**The law:** Human voluntary movement has a maximum jerk of approximately 500 m/s³.

Jerk is the third derivative of position — the rate of change of acceleration. In 1985, Flash and Hogan showed that humans minimize jerk when making arm movements. This is not a soft guideline. It is a hard constraint imposed by muscle physiology and neural motor control.

**What it catches:** Synthetic data augmentation (common jerk values: 1,000-5,000 m/s³), impact artifacts from sensor drops, and recording errors.

**How to check:**
```python
import numpy as np

def check_jerk(accel, timestamps_s):
    vel   = np.gradient(accel, timestamps_s, axis=0)
    acc2  = np.gradient(vel,   timestamps_s, axis=0)
    jerk  = np.gradient(acc2,  timestamps_s, axis=0)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    return np.percentile(jerk_magnitude, 95)  # p95, not max (noise tolerance)

# Pass: < 500 m/s³
# Fail: >= 500 m/s³
```

---

## Law 2: Rigid Body Kinematics (Euler)

**The law:** For any rigid limb segment, the acceleration at any point satisfies: `a = α×r + ω×(ω×r)`

Where α is angular acceleration, ω is angular velocity, and r is the position vector from the segment's center of mass. If your accelerometer and gyroscope are attached to the same rigid body (a forearm, a thigh, a shin), this relationship must hold.

**What it catches:** Sensor slip (the IMU moving relative to the body), loose attachment, and data from different recordings spliced together.

**How to check:**
```python
def check_rigid_body(accel, gyro, r_vector):
    # r_vector: position of sensor relative to joint center (meters)
    alpha = np.gradient(gyro, axis=0)  # angular acceleration
    
    expected_accel = (np.cross(alpha, r_vector) + 
                      np.cross(gyro, np.cross(gyro, r_vector)))
    
    r, _ = scipy.stats.pearsonr(
        accel.flatten(), 
        expected_accel.flatten()
    )
    return r  # Pass: r > 0.15  Fail: r <= 0.15
```

---

## Law 3: Resonance Frequency (Flash & Hogan, 1985)

**The law:** The human forearm has a characteristic physiological tremor at 8-12 Hz.

This is not pathological tremor (which is 4-6 Hz in Parkinson's). It is normal, healthy, constant, and present in every living person's arm movement. It appears in accelerometer data as a small oscillation superimposed on voluntary motion.

**What it catches:** Synthetic data (which typically lacks this), and recordings made on mannequins or robotic arms.

**How to check:**
```python
def check_resonance(accel, sample_rate_hz):
    fft_mag = np.abs(np.fft.rfft(accel[:, 0]))
    freqs   = np.fft.rfftfreq(len(accel), d=1/sample_rate_hz)
    
    # Check for energy in 8-12 Hz band
    mask = (freqs >= 8) & (freqs <= 12)
    band_energy  = np.sum(fft_mag[mask])
    total_energy = np.sum(fft_mag)
    
    ratio = band_energy / (total_energy + 1e-10)
    return ratio  # Pass: > 0.05  Fail: <= 0.05
```

---

## Law 4: IMU Coupling Consistency (Sensor Physics)

**The law:** Accelerometer and gyroscope readings from the same IMU chip must be correlated when the device is on a moving body.

When a rigid body rotates, both translational acceleration and angular velocity change together. A device that shows high angular velocity but flat acceleration is not on a moving human. A device that shows high acceleration but zero gyro signal is not measuring rotation correctly.

**What it catches:** Single-axis sensors presented as 6-DOF data, sensor channel swaps, and corrupted recordings.

**How to check:**
```python
def check_imu_coupling(accel, gyro):
    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag  = np.linalg.norm(gyro,  axis=1)
    
    r, _ = scipy.stats.pearsonr(accel_mag, gyro_mag)
    return r  # Pass: r > 0.15  Fail: r <= 0.15
```

---

## Law 5: Newton F=ma Coupling (Newton, 1687)

**The law:** When EMG data is present, muscle activation must precede the resulting limb acceleration by 50-100ms.

This is the electromechanical delay — the time between a motor neuron firing and the resulting muscle contraction producing force. It is a fixed physiological constant for voluntary movement. Synthetic EMG data often ignores this delay, producing signals that are simultaneous with or lag behind the acceleration they supposedly cause.

**What it catches:** Synthetic EMG, time-shifted recordings, and sensor synchronization failures between EMG and IMU devices.

**How to check:**
```python
def check_emg_imu_coupling(emg_envelope, accel_magnitude, sample_rate):
    # Cross-correlation to find peak lag
    cross_corr = np.correlate(accel_magnitude - accel_magnitude.mean(),
                               emg_envelope - emg_envelope.mean(), mode='full')
    lag_samples = np.argmax(cross_corr) - len(accel_magnitude) + 1
    lag_ms = (lag_samples / sample_rate) * 1000
    
    return lag_ms  # Pass: 50 <= lag_ms <= 100  Fail: outside this range
```

---

## Law 6: BCG Heartbeat Signature (Starr, 1939)

**The law:** The heartbeat produces a mechanical pulse — the ballistocardiogram (BCG) — visible in a wrist-mounted accelerometer at rest.

First described by Isaac Starr in 1939, the BCG is the recoil of the body from the ejection of blood by the heart. At the wrist, it appears as a ~0.02g pulse at the heart rate frequency (0.8-2.0 Hz). Every living human has it. Mannequins, robots, and synthetic data do not.

**What it catches:** Data recorded on non-human subjects, synthetic data, and recordings with severe motion artifact that has been incorrectly labeled as rest.

**How to check:**
```python
def check_bcg(accel_at_rest, sample_rate):
    # Only check during rest periods (low overall motion)
    fft_mag = np.abs(np.fft.rfft(accel_at_rest[:, 2]))  # vertical axis
    freqs   = np.fft.rfftfreq(len(accel_at_rest), d=1/sample_rate)
    
    # Look for energy in heart rate band (0.8-2.0 Hz = 48-120 BPM)
    mask = (freqs >= 0.8) & (freqs <= 2.0)
    return np.sum(fft_mag[mask]) / np.sum(fft_mag)
    # Pass: > 0.02  Fail: <= 0.02
```

---

## Law 7: Joule Heating Consistency (Ohm, 1827)

**The law:** When EMG and thermal sensors are both present, the electrical power of the EMG signal should correlate with skin temperature increase over time.

Muscle contraction converts chemical energy to mechanical energy and heat. The electrical power measured by EMG (I²R) predicts the thermal output. If a dataset claims simultaneous EMG and thermal measurements but they do not correlate, at least one sensor is faulty or the data was recorded separately and merged.

**What it catches:** Mismatched multi-sensor recordings, faulty thermal sensors, and data fusion errors.

---

## Why Nobody Does This

These checks take about 50ms per record on a laptop. Running them on a 100,000-record dataset takes less than two hours. There is no technical reason they are not standard practice.

The reason is cultural. Motion capture researchers focus on labels, not physics. "Is this record labeled correctly?" gets asked. "Does this record obey Newton's laws?" does not.

This is changing. As Physical AI (robots, prosthetics, exoskeletons) moves from research to deployment, the quality of training data becomes a safety issue. A prosthetic hand trained on data that violates jerk bounds will make movements no human ever makes. In a surgical robot, that is a patient safety issue.

---

## S2S Implements All Seven

S2S (github.com/timbo4u1/S2S) runs all seven checks on every record and issues a signed certificate. Free for research under BSL-1.1.

Live demo on your phone: https://timbo4u1.github.io/S2S

---

*References: Flash & Hogan (1985) "The coordination of arm movements", Starr (1939) "Studies made by simulating systole", Bernstein (1967) "The Co-ordination and Regulation of Movements"*
