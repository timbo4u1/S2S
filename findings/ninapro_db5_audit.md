# NinaPro DB5 Audit Findings

## Unit mismatch
Data stored in g, not m/s². z-mean ≈ −0.634, max ≈ 0.966.
Multiply by 9.81 before S2S certification.

## Upsampling artifact
Data stored at 2000Hz upsampled from 200Hz.
10× sample repetition triggers Law 13 (sensor_freeze) at active threshold=10.
Corrected: use 200Hz timestamps.

## Results (corrected)
- 8–39% GOLD by subject, 0% REJECTED
- sensor_freeze still high: Delsys hardware aggressive filtering
  produces legitimate rest-state flatlines
- resonance_frequency second-highest failure: sensor mounting
  mechanical coupling signature

## Law 13 implication
State-conditioned Law 13 (rest threshold=25, active=10) correctly
handles Delsys flatlines in rest state without false rejection.
