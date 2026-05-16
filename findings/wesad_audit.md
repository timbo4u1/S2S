# WESAD Audit Findings

## Setup
- Device: Empatica E4 wrist accelerometer
- Sampling rate: 32Hz
- Resolution: 64 LSB/g (8-bit ADC, ±2g range)
- Unit conversion: acc_ms2 = acc_raw / 64.0 * 9.81
- Windows: 128 samples = 4 seconds
- Subjects: S2-S11 (10 subjects, 20 windows each = 200 total)

## Results
| Tier | Count | % |
|---|---|---|
| GOLD | 0 | 0% |
| SILVER | 172 | 86% |
| BRONZE | 19 | 9% |
| REJECTED | 9 | 4% |

## Top law failures
| Law | Count | Explanation |
|---|---|---|
| resonance_frequency | 130 (65%) | 32Hz Nyquist=16Hz, tremor 8-12Hz barely resolvable — hardware limit, not data fault |
| sensor_freeze | 80 (40%) | E4 ADC quantization: small rest movements round to same integer value |
| spectral_flatness | 37 (18%) | Short windows at low Hz produce noisy PSD estimates |
| temporal_autocorrelation | 30 (15%) | Low Hz reduces lag-1 correlation reliability |
| cross_axis_cohesion | 1 (0.5%) | Near-zero — data is genuine biological |

## Law 14/15 (new)
- powerline_interference: 0 detections — E4 battery-powered, no mains coupling
- intra_window_splice: 0 detections — clean continuous recording

## Conclusions
1. 0% GOLD is structural — 32Hz + no gyro makes GOLD unreachable by design
2. 86% SILVER is the correct operating tier for WESAD wrist data
3. sensor_freeze at 40% reflects E4 ADC quantization at rest, not hardware fault
   — state-conditioned Law 13 helps but rest sessions are genuinely flat at this resolution
4. 4% REJECTED (9 windows) are genuine quality failures — sensor off-wrist or interference
5. WESAD chest ACC (700Hz, different segment) would reach GOLD — not audited here

## Recommendation
Use SILVER as passing threshold for WESAD wrist data.
resonance_frequency failures should not penalize score at hz < 40 — consider hz-gating this law.
