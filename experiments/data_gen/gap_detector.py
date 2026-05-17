"""
gap_detector.py — Gate 1 of head-tail-fill pipeline.

Scans NinaPro GOLD windows for fillable gaps between consecutive
certified windows of the same gesture class.

Training subjects: S1-S8 only.
Validation subjects: S9-S10 — never touched here.

Gate 1 target: >= 50 fillable gaps across >= 5 gesture classes.
If Gate 1 fails: stop, pivot to batch refinery mode.

Output: experiments/data_gen/gaps.json
"""

import sys, glob, json, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

try:
    import scipy.io as sio
    import numpy as np
    HAS_NP = True
except ImportError:
    print("ERROR: scipy and numpy required"); sys.exit(1)

# ── config ────────────────────────────────────────────────────────────────────
NINAPRO_DIR   = Path.home() / "ninapro_db5"
HZ            = 200
WIN           = 256                          # 1.28 seconds
N             = max(5, int(HZ * 0.05))       # 10 samples = 50ms head/tail
TRAIN_SUBJS   = list(range(1, 9))            # S1-S8 only
MAX_EUCL      = 2.0                          # m/s² vector distance
MIN_COS       = 0.85                         # direction continuity
MAX_DUR_MS    = 2000                         # max fill duration

# ── helpers ───────────────────────────────────────────────────────────────────
def make_ts(n, hz, offset=0):
    return [int(offset * 1e9 / hz) + int(i * 1e9 / hz) for i in range(n)]

def vec_mean(samples):
    n = len(samples)
    return [sum(s[k] for s in samples) / n for k in range(3)]

def euclidean(a, b):
    return math.sqrt(sum((a[k] - b[k])**2 for k in range(3)))

def cosine(a, b):
    dot  = sum(a[k]*b[k] for k in range(3))
    na   = math.sqrt(sum(x**2 for x in a)) + 1e-9
    nb   = math.sqrt(sum(x**2 for x in b)) + 1e-9
    return dot / (na * nb)

def certify_window(acc_list, offset=0):
    ts = make_ts(WIN, HZ, offset)
    pe = PhysicsEngine()
    return pe.certify({"timestamps_ns": ts,
                       "accel": acc_list,
                       "gyro":  [[0,0,0]]*WIN})

def is_fillable(tail, head, duration_ms):
    t_mean = vec_mean(tail)
    h_mean = vec_mean(head)
    euc    = euclidean(t_mean, h_mean)
    cos    = cosine(t_mean, h_mean)
    return {
        "ok":      euc < MAX_EUCL and cos > MIN_COS and duration_ms < MAX_DUR_MS,
        "euc":     round(euc, 4),
        "cos":     round(cos, 4),
        "dur_ms":  duration_ms,
        "fail":    ([] if euc < MAX_EUCL else ["euclidean"])
                 + ([] if cos > MIN_COS  else ["cosine"])
                 + ([] if duration_ms < MAX_DUR_MS else ["duration"])
    }

# ── main ─────────────────────────────────────────────────────────────────────
def run():
    files = sorted(glob.glob(str(NINAPRO_DIR / "s*" / "S*_E1_A1.mat")))
    train_files = [f for f in files
                   if any(f"s{i}/" in f.lower() or f"s{i}\\" in f.lower()
                          for i in TRAIN_SUBJS)]

    if not train_files:
        print(f"ERROR: no .mat files found in {NINAPRO_DIR}")
        sys.exit(1)

    print(f"Gap Detector — NinaPro DB5")
    print(f"Training files: {len(train_files)}")
    print(f"Win={WIN} samples, N={N} head/tail, HZ={HZ}")
    print(f"Criteria: euclidean<{MAX_EUCL}, cosine>{MIN_COS}, dur<{MAX_DUR_MS}ms")
    print()

    gaps       = []
    stats      = {"files": 0, "windows_certified": 0,
                  "gold": 0, "silver": 0, "pairs_tested": 0,
                  "rejected_euclidean": 0, "rejected_cosine": 0,
                  "rejected_duration": 0}
    by_gesture = {}

    for fpath in train_files:
        subj = Path(fpath).stem.split("_")[0]   # e.g. "S3"
        try:
            data = sio.loadmat(fpath)
        except Exception as e:
            print(f"  {subj}: load error {e}"); continue

        # get acc
        acc_raw = next((data[k] for k in ["acc","accel","ACC"]
                        if k in data and hasattr(data[k],'shape')), None)
        if acc_raw is None:
            print(f"  {subj}: no acc key"); continue

        a = acc_raw[:, :3].astype(float)   # first 3 cols = wrist IMU

        # get stimulus (gesture labels)
        stim_raw = next((data[k] for k in ["stimulus","restimulus"]
                         if k in data), None)
        if stim_raw is None:
            print(f"  {subj}: no stimulus key — skipping"); continue

        stim = stim_raw.flatten().astype(int)
        if len(stim) != len(a):
            stim = np.resize(stim, len(a))

        stats["files"] += 1
        n_total = len(a)
        file_gaps = 0

        # find gesture episodes (contiguous non-zero same label)
        i = 0
        while i < n_total:
            label = stim[i]
            if label == 0:          # rest — skip
                i += 1; continue

            # find end of this gesture episode
            j = i
            while j < n_total and stim[j] == label:
                j += 1

            episode_len = j - i
            if episode_len < WIN * 2:   # need at least 2 windows
                i = j; continue

            # window this episode, certify each window
            episode_acc  = a[i:j].tolist()
            ep_windows   = []
            w_offset     = 0

            while w_offset + WIN <= len(episode_acc):
                # skip first and last window of episode (session boundary rule)
                if w_offset == 0 or w_offset + WIN + WIN > len(episode_acc):
                    w_offset += WIN; continue

                chunk = episode_acc[w_offset:w_offset + WIN]
                r     = certify_window(chunk, offset=i + w_offset)
                tier  = r["tier"]
                stats["windows_certified"] += 1

                if tier in ("GOLD", "SILVER"):
                    if tier == "GOLD":   stats["gold"]   += 1
                    else:                stats["silver"] += 1
                    ep_windows.append({"acc": chunk, "tier": tier,
                                       "score": r.get("score", r.get("physical_law_score", 0)),
                                       "offset": w_offset})

                w_offset += WIN

            # test consecutive certified window pairs
            for k in range(len(ep_windows) - 1):
                wi = ep_windows[k]
                wj = ep_windows[k + 1]
                stats["pairs_tested"] += 1

                tail = wi["acc"][-N:]    # last N samples of window i
                head = wj["acc"][:N]     # first N samples of window j

                dur_ms = int((WIN / HZ) * 1000)   # body duration
                chk    = is_fillable(tail, head, dur_ms)

                for fail in chk["fail"]:
                    stats[f"rejected_{fail}"] = stats.get(f"rejected_{fail}", 0) + 1

                if chk["ok"]:
                    gap = {
                        "subject":            subj,
                        "gesture_label":      int(label),
                        "head_samples":       head,
                        "tail_samples":       tail,
                        "duration_ms":        dur_ms,
                        "euclidean_distance": chk["euc"],
                        "cosine_similarity":  chk["cos"],
                        "window_i_tier":      wi["tier"],
                        "window_j_tier":      wj["tier"],
                        "window_i_score":     wi.get("score", 0),
                        "window_j_score":     wj.get("score", 0),
                    }
                    gaps.append(gap)
                    by_gesture[int(label)] = by_gesture.get(int(label), 0) + 1
                    file_gaps += 1

            i = j

        print(f"  {subj}: {file_gaps} gaps found")

    # ── report ────────────────────────────────────────────────────────────────
    n_gesture_classes = len(by_gesture)
    print()
    print("=" * 52)
    print(f"GATE 1 RESULTS")
    print(f"  Total fillable gaps:   {len(gaps)}")
    print(f"  Gesture classes:       {n_gesture_classes}")
    print(f"  Windows certified:     {stats['windows_certified']}")
    print(f"    GOLD:                {stats['gold']}")
    print(f"    SILVER:              {stats['silver']}")
    print(f"  Pairs tested:          {stats['pairs_tested']}")
    print(f"  Rejected (euclidean):  {stats.get('rejected_euclidean',0)}")
    print(f"  Rejected (cosine):     {stats.get('rejected_cosine',0)}")
    print(f"  Rejected (duration):   {stats.get('rejected_duration',0)}")
    print()

    gate_gaps = len(gaps) >= 50
    gate_gest = n_gesture_classes >= 5
    print(f"  Gate 1 — gaps >= 50:   {'PASS' if gate_gaps else 'FAIL'} ({len(gaps)})")
    print(f"  Gate 1 — classes >= 5: {'PASS' if gate_gest else 'FAIL'} ({n_gesture_classes})")
    print()

    if gate_gaps and gate_gest:
        print("  GATE 1: PASS — proceed to minjerk_fill.py")
        out = Path(__file__).parent / "gaps.json"
        with open(out, "w") as f:
            json.dump({"meta": stats, "by_gesture": by_gesture,
                       "gaps": gaps}, f, indent=2)
        print(f"  Saved: {out} ({len(gaps)} gaps)")
        print()
        print("  Top gesture classes:")
        for lbl, cnt in sorted(by_gesture.items(), key=lambda x:-x[1])[:8]:
            print(f"    gesture {lbl:3d}: {cnt} gaps")
    else:
        print("  GATE 1: FAIL — pivot to batch refinery mode")
        print("  Do not proceed to minjerk_fill.py")

    print("=" * 52)

if __name__ == "__main__":
    run()
