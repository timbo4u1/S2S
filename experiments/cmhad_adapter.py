#!/usr/bin/env python3
"""C-MHAD IMU adapter. Loads trials as recorded; never fills or rescales data. Does not run S2S."""
import glob, os, re
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/S2S_data/c-mhad/raw/subjects")
WIN, STRIDE = 256, 128


def load_trial(path):
    d = pd.read_csv(path, sep="\t", skiprows=[1, 2], usecols=range(7)).to_numpy(float)
    ts_ms = d[:, 0]
    dt = np.diff(ts_ms)
    step = float(np.median(dt))
    lost = np.round(dt / step).astype(int) - 1       # samples missing between neighbours
    return {
        "ts_ns": ((ts_ms - ts_ms[0]) * 1e6).astype(np.int64),
        "accel": d[:, 1:4],                          # m/s^2, as recorded
        "gyro": np.deg2rad(d[:, 4:7]),               # deg/s -> rad/s
        "gap_at": np.where(lost > 0)[0] + 1,         # first sample after each gap
        "gap_lost": lost[lost > 0],
        "step_ms": step,
    }


def trial_ids(path):
    s, t = re.search(r"inertial_sub(\d+)_tr(\d+)", os.path.basename(path)).groups()
    return int(s), int(t)


def trial_files():
    return sorted(glob.glob(ROOT + "/Subject*/InertialData/inertial_*.xls"), key=trial_ids)


def iter_windows(tr):
    for s in range(0, len(tr["ts_ns"]) - WIN + 1, STRIDE):
        e = s + WIN
        yield s, e, bool(((tr["gap_at"] > s) & (tr["gap_at"] < e)).any())


if __name__ == "__main__":
    n_tr = n_win = n_gap = 0
    for f in trial_files():
        tr = load_trial(f)
        assert (np.diff(tr["ts_ns"]) > 0).all(), f
        n_tr += 1
        for _, _, g in iter_windows(tr):
            n_win += 1
            n_gap += g
    print(f"trials={n_tr} windows={n_win} gap_windows={n_gap} clean_windows={n_win - n_gap}")
    tr = load_trial(trial_files()[0])
    gm = np.linalg.norm(tr["gyro"], axis=1)
    print("first trial: n=%d step=%.3f ms gaps=%d | gyro rad/s p50=%.3f p99=%.3f"
          % (len(tr["ts_ns"]), tr["step_ms"], len(tr["gap_at"]), *np.percentile(gm, [50, 99])))
