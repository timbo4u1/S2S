# Head-Tail-Fill Ablation Results

## Result: B < A at all augmentation ratios

| Experiment | F1 | vs baseline |
|---|---|---|
| A — real only (baseline) | 0.0637 | — |
| B — real + fills 20% | 0.0491 | −0.015 |
| B — real + fills 50% | 0.0615 | −0.002 |
| B — real + fills 100% | 0.0504 | −0.013 |

## Why it failed

Body = 236/256 samples (92%) is synthetic minimum-jerk.
Head + tail = only 20 samples (8%) real anchor.
The synthetic body dominates and erases gesture identity.
Min-jerk in acceleration space produces smooth but semantically empty signal.

## What this means for the approach

The concept is architecturally sound but the anchor ratio is wrong.
10 samples of real data cannot anchor 236 samples of synthetic body.
To preserve gesture identity, real anchor must dominate — not synthetic fill.

## Gate result

Gate 3 (B > A): FAIL
Per roadmap: stop head-tail-fill development.
Pivot to batch refinery mode.

## What would be needed to make this work

- Much shorter body: 10-20 synthetic samples max between longer real anchors
- Or: interpolation in learned feature space, not raw acceleration space
- Or: full trajectory model with gesture-specific dynamics (6+ months)

## Next step

Batch refinery — process entire datasets, output graded quality reports.
This is the proven S2S value: filtering, not generating.
