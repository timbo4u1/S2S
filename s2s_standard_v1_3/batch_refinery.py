"""
S2S Batch Refinery — process entire dataset folders.

Usage:
    python3 -m s2s_standard_v1_3.batch_refinery --input /path/to/data --output report.csv
    python3 -m s2s_standard_v1_3.batch_refinery --input /path/to/data --segment forearm

Input:  folder containing CSV, MAT, or PKL sensor files
Output: refinery_report.csv  — one row per window
        refinery_summary.txt — dataset quality overview

This is what you run on your dataset before training.
"""

import sys, csv, json, glob, math, argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

HZ_DEFAULT  = 200
WIN         = 256
SEGMENT     = "forearm"

# ── loaders ──────────────────────────────────────────────────────────────────
def load_csv(fpath, hz=HZ_DEFAULT):
    """Load any CSV with numeric columns. Auto-detect accel columns."""
    import csv as _csv
    rows = []
    with open(fpath) as f:
        reader = _csv.reader(f)
        header = None
        for row in reader:
            try:
                vals = [float(v) for v in row]
                rows.append(vals)
            except ValueError:
                if header is None:
                    header = row
    if not rows:
        return None, hz
    arr = [[rows[i][j] for j in range(min(3, len(rows[i])))]
           for i in range(len(rows))]
    return arr, hz

def load_mat(fpath):
    try:
        import scipy.io as sio
        data = sio.loadmat(fpath)
        acc  = next((data[k] for k in ["acc","accel","ACC"]
                     if k in data and hasattr(data[k],'shape')), None)
        if acc is None:
            return None, HZ_DEFAULT
        return acc[:, :3].astype(float).tolist(), HZ_DEFAULT
    except Exception:
        return None, HZ_DEFAULT

def load_pkl(fpath):
    import pickle
    try:
        d = pickle.load(open(fpath,'rb'), encoding='latin1')
        # WESAD-style
        if isinstance(d, dict) and 'signal' in d:
            acc = d['signal']['wrist']['ACC'].astype(float) / 64.0 * 9.81
            return acc.tolist(), 32
        # generic dict with acc key
        for k in ["acc","accel","ACC"]:
            if k in d:
                arr = d[k]
                if HAS_NP:
                    import numpy as np
                    arr = np.array(arr)[:, :3].tolist()
                return arr, HZ_DEFAULT
        return None, HZ_DEFAULT
    except Exception:
        return None, HZ_DEFAULT

def load_file(fpath):
    ext = Path(fpath).suffix.lower()
    if ext == '.csv':  return load_csv(fpath)
    if ext == '.mat':  return load_mat(fpath)
    if ext == '.pkl':  return load_pkl(fpath)
    return None, HZ_DEFAULT

# ── certify windows ──────────────────────────────────────────────────────────
def certify_file_windows(fpath, segment=SEGMENT):
    acc, hz = load_file(fpath)
    if acc is None or len(acc) < WIN:
        return []

    results = []
    pe      = PhysicsEngine()
    n       = len(acc)
    win_idx = 0

    for start in range(0, n - WIN + 1, WIN):
        chunk = acc[start:start+WIN]
        ts    = [int(i * 1e9 / hz) for i in range(WIN)]
        r     = pe.certify({"timestamps_ns": ts,
                            "accel":         chunk,
                            "gyro":          [[0,0,0]]*WIN},
                           segment=segment)
        results.append({
            "file":        Path(fpath).name,
            "window":      win_idx,
            "start_sample":start,
            "end_sample":  start + WIN,
            "hz":          hz,
            "tier":        r["tier"],
            "score":       r.get("score", r.get("physical_law_score", 0)),
            "laws_passed": "|".join(r.get("laws_passed", [])),
            "laws_failed": "|".join(r.get("laws_failed", [])),
        })
        win_idx += 1

    return results

# ── summary ──────────────────────────────────────────────────────────────────
def build_summary(all_results, input_dir, segment, elapsed_s):
    total  = len(all_results)
    if total == 0:
        return "No windows processed."

    tiers  = {"GOLD":0,"SILVER":0,"BRONZE":0,"REJECTED":0}
    law_fails = {}
    scores = []

    for r in all_results:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
        scores.append(float(r["score"]))
        for law in r["laws_failed"].split("|"):
            if law:
                law_fails[law] = law_fails.get(law, 0) + 1

    usable  = tiers["GOLD"] + tiers["SILVER"]
    quality = int(100 * usable / total)
    mean_sc = sum(scores) / len(scores)

    if quality >= 85:
        verdict = "EXCELLENT — ready for robot training pipelines."
    elif quality >= 70:
        verdict = "GOOD — usable with minor preprocessing."
    elif quality >= 50:
        verdict = "FAIR — review flagged windows before training."
    else:
        verdict = "POOR — significant data quality issues detected."

    lines = [
        "=" * 56,
        "  S2S BATCH REFINERY REPORT",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 56,
        f"  Input:          {input_dir}",
        f"  Segment:        {segment}",
        f"  Windows:        {total}",
        f"  Processed in:   {elapsed_s:.1f}s",
        "",
        f"  DATASET QUALITY SCORE: {quality}%",
        f"  VERDICT: {verdict}",
        "",
        "  Tier distribution:",
        f"    GOLD:     {tiers['GOLD']:5d}  ({100*tiers['GOLD']//total}%)",
        f"    SILVER:   {tiers['SILVER']:5d}  ({100*tiers['SILVER']//total}%)",
        f"    BRONZE:   {tiers['BRONZE']:5d}  ({100*tiers['BRONZE']//total}%)",
        f"    REJECTED: {tiers['REJECTED']:5d}  ({100*tiers['REJECTED']//total}%)",
        f"    USABLE:   {usable:5d}  ({quality}%)",
        "",
        f"  Mean score:     {mean_sc:.1f}/100",
        "",
        "  Top failure modes:",
    ]
    for law, cnt in sorted(law_fails.items(), key=lambda x:-x[1])[:6]:
        pct = 100 * cnt // total
        lines.append(f"    {law:<30s} {cnt:5d} ({pct}%)")

    lines += [
        "",
        "  Recommendation:",
        f"    Use GOLD+SILVER windows for training ({usable}/{total}).",
        f"    Discard REJECTED windows ({tiers['REJECTED']}).",
        f"    Review BRONZE windows ({tiers['BRONZE']}) manually.",
        "=" * 56,
    ]
    return "\n".join(lines)

# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="S2S Batch Refinery")
    parser.add_argument("--input",   required=True, help="Input folder or file")
    parser.add_argument("--output",  default="refinery_report.csv")
    parser.add_argument("--segment", default="forearm")
    parser.add_argument("--ext",     default="csv,mat,pkl",
                        help="File extensions to process")
    args = parser.parse_args()

    input_path = Path(args.input)
    exts       = [f".{e.strip()}" for e in args.ext.split(",")]

    if input_path.is_file():
        files = [input_path]
    else:
        # deduplicate by filename — handles nested duplicate folders
        seen = set()
        files = []
        for f in sorted(input_path.rglob("*")):
            if f.suffix.lower() in exts and f.name not in seen:
                seen.add(f.name)
                files.append(f)

    if not files:
        print(f"No files found in {input_path} with extensions {exts}")
        sys.exit(1)

    print(f"S2S Batch Refinery")
    print(f"Input:   {input_path}")
    print(f"Files:   {len(files)}")
    print(f"Segment: {args.segment}")
    print()

    import time
    t0         = time.time()
    all_results = []

    for i, fpath in enumerate(sorted(files)):
        print(f"  [{i+1}/{len(files)}] {fpath.name}...", end=" ", flush=True)
        results = certify_file_windows(str(fpath), segment=args.segment)
        all_results.extend(results)
        tiers = {}
        for r in results:
            tiers[r["tier"]] = tiers.get(r["tier"],0) + 1
        print(f"{len(results)} windows — " +
              " ".join(f"{t}:{c}" for t,c in sorted(tiers.items())))

    elapsed = time.time() - t0

    # write CSV
    out_csv = Path(args.output)
    if all_results:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys())
            w.writeheader()
            w.writerows(all_results)
        print(f"\nReport: {out_csv} ({len(all_results)} rows)")

    # write summary
    summary = build_summary(all_results, input_path, args.segment, elapsed)
    print("\n" + summary)

    summary_path = out_csv.with_suffix(".txt")
    open(summary_path, "w").write(summary)
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    main()
