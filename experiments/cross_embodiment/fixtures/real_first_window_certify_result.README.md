# Real first-window PhysicsEngine.certify() fixture

This fixture was exported from the actual S2S engine return value; it was not manually reconstructed.

## Files

- `real_first_window_certify_result.json` — exact JSON serialization of the `PhysicsEngine.certify()` return dict.
- `real_first_window_certify_input.json` — exact `imu_raw` argument passed to `certify()`.

## Generation

Command used (run from repository root):

```bash
python3 /tmp/export_first_window_fixture.py
```

The script is equivalent to:

```python
import math
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

# imu_raw is the deterministic fixture stored in real_first_window_certify_input.json
result = PhysicsEngine().certify(
    imu_raw=imu_raw,
    segment="forearm",
    device_id="audit-dev",
    session_id="audit-session",
)
json.dump(result, open("experiments/cross_embodiment/fixtures/real_first_window_certify_result.json", "w"), indent=2)
```

## Audit evidence

`result["law_details"]["inter_window_continuity"]` contains:

```json
{
  "law": "Inter_Window_Continuity",
  "equation": "|accel_delta| <= V_max * dt * scale",
  "physical_meaning": "Human limb cannot teleport between consecutive windows",
  "info": "FIRST_WINDOW_INITIALIZED",
  "status": "NOT_ASSESSED"
}
```

This is the genuine first-window engine output that caused the Milestone 2C adapter rejection.
