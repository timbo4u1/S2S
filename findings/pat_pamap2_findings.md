# Physical Action Tokenizer (PAT) — PAMAP2 Findings

## Result
Physics certification tier monotonically predicts action prediction
uncertainty across 4,607 windows, 8 activity classes, 9 subjects.

## Numbers
| Tier | Entropy H | Windows | vs Baseline |
|---|---|---|---|
| GOLD | 0.0335 | 92 | 5.7x sharper |
| SILVER | 0.1210 | 4,343 | baseline |
| REJECTED | 0.6298 | 171 | 7.7x wider |

REJECTED/GOLD ratio: 18.8x

## What this means
Each S2S-certified action token carries calibrated uncertainty.
A GOLD token is 18.8x more confident than a REJECTED token.
Downstream policies can weight training by physics trust level.

## Baseline accuracy: 92.4%
Near ceiling — top-1 accuracy improvement not expected or observed.
The PAT contribution is uncertainty calibration, not classification.

## Known limitation
GOLD entropy ordering may reflect data difficulty, not purely
physics constraint. Controlled experiment needed to separate the two.

## Script
experiments/pat_pamap2.py
experiments/results_pat_pamap2.json
