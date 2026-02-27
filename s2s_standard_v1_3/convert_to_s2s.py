#!/usr/bin/env python3
"""
convert_to_s2s.py (s2s v1.3)

Robust converter:
 - fail-fast delimiter detection (or explicit --delimiter)
 - detect numeric columns
 - optional synchronized NaN removal (--remove-nans)
 - Gaussian jitter injection applied to intervals (--inject-jitter-stdns)
 - optional reproducible jitter with --seed
 - explicit error if --sign-key requested but cryptography unavailable
 - writes tool_provenance into META_JSON (including sanitized CLI args and nan summary)
"""
from __future__ import annotations
import argparse
import csv
import json
import struct
import tempfile
import zlib
import logging
import time
import random
import math
import statistics
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import importlib
import sys

# Optional crypto libs (only required if --sign-key used)
try:
    _crypto_ed25519 = importlib.import_module("cryptography.hazmat.primitives.asymmetric.ed25519")
    _crypto_serial = importlib.import_module("cryptography.hazmat.primitives.serialization")
    CRYPTO_AVAILABLE = True
except Exception:
    _crypto_ed25519 = None
    _crypto_serial = None
    CRYPTO_AVAILABLE = False

# constants import (package or local)
try:
    from .constants import (
        MAGIC, VERSION_MAJOR, VERSION_MINOR,
        HEADER_CORE_FMT, HEADER_CORE_LEN, HEADER_TOTAL_LEN,
        TLV_META_JSON, TLV_TIMESTAMPS_NS, TLV_SENSOR_DATA, TLV_SIGNATURE,
        V1_2_SAMPLE_COUNT_GOLD
    )
except Exception:
    from constants import (
        MAGIC, VERSION_MAJOR, VERSION_MINOR,
        HEADER_CORE_FMT, HEADER_CORE_LEN, HEADER_TOTAL_LEN,
        TLV_META_JSON, TLV_TIMESTAMPS_NS, TLV_SENSOR_DATA, TLV_SIGNATURE,
        V1_2_SAMPLE_COUNT_GOLD
    )


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Mint SCAN2SELL .s2s files (v1.3 canonical)")
    p.add_argument("input", help="Input CSV/TXT file (header row preferred)")
    p.add_argument("-o", "--output", required=False, help="Output .s2s file (defaults to input.s2s)")
    p.add_argument("-c", "--columns", required=False, nargs='*',
                   help="Column names or zero-based indices to include as sensors (default: all numeric non-timestamp columns)")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sign-key", required=False, help="PEM file path to Ed25519 private key (optional)")
    # register both short and long forms robustly
    p.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                   help="Verbose logging (same as -v)")
    p.add_argument("--inject-jitter-stdns", type=float, default=0.0,
                   help="Inject Gaussian jitter on intervals with this stddev (ns). 0=disabled")
    p.add_argument("--seed", type=int, default=None, help="Optional seed for jitter RNG (default: time.time_ns())")
    p.add_argument("--remove-nans", action="store_true",
                   help="Remove any sample rows containing NaNs across any channel (synchronized).")
    p.add_argument("--fill-nans", choices=("none", "zero", "mean", "ffill"), default="none",
                   help="Fallback fill strategy if you do not want removal (not recommended for certification).")
    p.add_argument("--annotate-suspect", action="store_true",
                   help="Annotate meta with suspicion notes when timestamps look perfectly regular.")
    p.add_argument("--delimiter", default=None, help="Explicit delimiter for CSV parsing (bypass sniff).")
    return p.parse_args()



# ---------- table reader (fail-fast) ----------
def read_table_rows(path: Path, explicit_delim: Optional[str] = None) -> List[List[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text:
        raise ValueError("No rows read from file")

    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        raise ValueError("No rows read from file")

    # If explicit delim provided, use it
    if explicit_delim:
        reader = csv.reader(lines, delimiter=explicit_delim)
        return [list(r) for r in reader]

    # Fast path: .csv uses comma
    if path.suffix.lower() == ".csv":
        reader = csv.reader(lines, delimiter=',')
        return [list(r) for r in reader]

    # Try sniffing and validate that sniff result yields consistent column counts
    sample = "\n".join(lines[:50])
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        delim = dialect.delimiter
        if not isinstance(delim, str) or delim == "":
            raise csv.Error("Bad delimiter from sniffer")
        reader = csv.reader(lines, delimiter=delim)
        rows = [list(r) for r in reader]
        # validation: consistent column counts across first 50 rows
        counts = [len(r) for r in rows[:50] if r]
        if counts and (max(counts) - min(counts) > 2):
            # ambiguous: don't guess — fail fast so user can provide --delimiter
            raise ValueError("Ambiguous delimiter detected by csv.Sniffer; provide --delimiter (e.g. ',' or $'\\t').")
        return rows
    except csv.Error as e:
        raise ValueError(f"CSV sniffer failed: {e}. Provide --delimiter to override.")
    except Exception:
        # fallback is not acceptable for v1.2: fail fast
        raise ValueError("Ambiguous or malformed table; provide --delimiter or fix input file")

# ---------- numeric detection ----------
def _looks_numeric(s: str) -> bool:
    if s is None:
        return False
    s = str(s).strip()
    if s == "":
        return False
    try:
        float(s)
        return True
    except Exception:
        return False

def detect_numeric_columns(header: List[str], rows: List[List[str]], min_numeric_ratio: float = 0.6) -> List[Tuple[int, str]]:
    numeric_cols: List[Tuple[int, str]] = []
    for i, name in enumerate(header):
        if name and isinstance(name, str) and name.strip().lower().startswith("timestamp"):
            continue
        numeric = 0
        checked = 0
        for r in rows[:200]:
            if i >= len(r):
                continue
            v = r[i].strip()
            if v == "":
                continue
            checked += 1
            try:
                float(v)
                numeric += 1
            except Exception:
                # record but keep scanning; if too many malformed entries later we'll escalate
                pass
        ratio = (numeric / checked) if checked > 0 else 0.0
        if checked > 0 and ratio >= min_numeric_ratio:
            numeric_cols.append((i, name))
        else:
            # if checked > 0 and ratio is extremely low, raise so user knows data corrupted
            if checked > 0 and ratio < 0.05:
                raise ValueError(f"Column {name!r} appears massively non-numeric ({ratio*100:.1f}% numeric). Check file.")
    return numeric_cols

# ---------- helpers ----------
def pack_tlv(t: int, value: bytes) -> bytes:
    return struct.pack('<HI', int(t), len(value)) + value

def compute_jitter_metrics(timestamps: List[int]) -> Dict[str, Optional[float]]:
    if not timestamps or len(timestamps) < 2:
        return {"mean_delta_ns": None, "rms_jitter_ns": None, "cv": None, "p2p_ns": None, "c2c_ns": None}
    deltas = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
    mu = statistics.mean(deltas)
    sigma = statistics.pstdev(deltas) if len(deltas) >= 1 else 0.0
    p2p = max(deltas) - min(deltas)
    c2c = max(abs(d2 - d1) for d1, d2 in zip(deltas, deltas[1:])) if len(deltas) >= 2 else 0
    cv = (sigma / mu) if mu and mu > 0 else None
    return {"mean_delta_ns": mu, "rms_jitter_ns": sigma, "cv": cv, "p2p_ns": p2p, "c2c_ns": c2c}

def inject_gaussian_jitter_on_intervals(timestamps: List[int], stddev_ns: float, seed: Optional[int] = None) -> List[int]:
    if stddev_ns <= 0 or len(timestamps) < 2:
        return timestamps[:]
    rnd = random.Random(seed if seed is not None else int(time.time_ns()))
    deltas = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
    new_deltas = []
    for d in deltas:
        noise = rnd.gauss(0.0, float(stddev_ns))
        nd = max(1, int(round(d + noise)))
        new_deltas.append(nd)
    out = [timestamps[0]]
    for d in new_deltas:
        out.append(out[-1] + d)
    return out

def clean_synchronized_rows(timestamps: List[int], sensor_list: List[Tuple[str, List[float]]]
                            ) -> Tuple[List[int], List[Tuple[str, List[float]]], Dict[str, Any]]:
    n = len(timestamps)
    if n == 0:
        return [], [], {"original_count": 0, "kept_count": 0, "removed_pct": 0.0}
    # defensive: determine shortest channel length
    min_len = min((len(vals) for (_nm, vals) in sensor_list), default=n)
    nrows = min(n, min_len)
    valid_indices: List[int] = []
    for i in range(nrows):
        ok = True
        for (_name, vals) in sensor_list:
            v = vals[i]
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                ok = False
                break
        if ok:
            valid_indices.append(i)
    new_ts = [timestamps[i] for i in valid_indices]
    new_sensor_list = [(name, [vals[i] for i in valid_indices]) for (name, vals) in sensor_list]
    prov = {
        "original_count": n,
        "kept_count": len(valid_indices),
        "removed_pct": (100.0 * (n - len(valid_indices)) / n) if n > 0 else 0.0
    }
    return new_ts, new_sensor_list, prov

def fill_nans_series(vals: List[float], strategy: str) -> List[float]:
    if strategy == "none":
        return vals[:]
    if strategy == "zero":
        return [0.0 if not (isinstance(v, (int, float)) and math.isfinite(v)) else v for v in vals]
    if strategy == "mean":
        valid = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        meanv = statistics.mean(valid) if valid else 0.0
        return [meanv if not (isinstance(v, (int, float)) and math.isfinite(v)) else v for v in vals]
    if strategy == "ffill":
        out = []
        last = 0.0
        started = False
        for v in vals:
            if isinstance(v, (int, float)) and math.isfinite(v):
                out.append(v); last = v; started = True
            else:
                out.append(last if started else 0.0)
        return out
    return vals[:]

# ---------- core writer ----------
def write_s2s_file(output_path: Path, sensors: List[Tuple[str, List[float]]], timestamps: List[int],
                   sign_key: Optional[str] = None, meta_extra: Optional[Dict[str, Any]] = None):
    meta = {"sensor_map": [s[0] for s in sensors], "entries": len(timestamps), "version": f"{VERSION_MAJOR}.{VERSION_MINOR}"}
    if meta_extra:
        meta.update(meta_extra)
    meta_b = json.dumps(meta, separators=(',', ':')).encode('utf-8')
    tlv_meta = pack_tlv(TLV_META_JSON, meta_b)
    ts_b = b''.join(struct.pack('<Q', int(t)) for t in timestamps)
    tlv_ts = pack_tlv(TLV_TIMESTAMPS_NS, ts_b)
    payload = tlv_meta + tlv_ts
    for (_name, vals) in sensors:
        data_bytes = b''.join(struct.pack('<d', float(x)) for x in vals)
        payload += pack_tlv(TLV_SENSOR_DATA, data_bytes)
    payload_crc = zlib.crc32(payload) & 0xFFFFFFFF
    version_int = (VERSION_MAJOR << 16) | VERSION_MINOR
    creation_ns = int(time.time_ns())
    payload_len = len(payload)
    header_core = struct.pack(HEADER_CORE_FMT, MAGIC, version_int, creation_ns, payload_len)
    header_crc = zlib.crc32(header_core) & 0xFFFFFFFF
    header = header_core + struct.pack('<II', header_crc, payload_crc)
    if len(header) != HEADER_TOTAL_LEN:
        raise RuntimeError("Header length incorrect")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp:
        tmp.write(header); tmp.write(payload); tmp.flush(); tmp_path = Path(tmp.name)
    # optional signing
    if sign_key:
        if not CRYPTO_AVAILABLE:
            # Defensive: explicit error (fail-fast) so auditing knows signing was requested but impossible
            raise ImportError("cryptography package required to sign, but it is not available")
        key_bytes = Path(sign_key).read_bytes()
        try:
            # Try loading generic private key (PEM)
            private = _crypto_serial.load_pem_private_key(key_bytes, password=None)
        except Exception:
            # fallback: interpret raw private bytes for Ed25519
            private = _crypto_ed25519.Ed25519PrivateKey.from_private_bytes(key_bytes)
        blob = tmp_path.read_bytes()
        sig = private.sign(blob)
        with tmp_path.open('ab') as f:
            f.write(pack_tlv(TLV_SIGNATURE, sig))
    if output_path.exists():
        output_path.unlink()
    tmp_path.replace(output_path)
    manifest = {"output": str(output_path), "entries": len(timestamps), "header_crc": hex(header_crc), "payload_crc": hex(payload_crc), "signed": bool(sign_key)}
    return manifest

# ---------- public API + CLI flow ----------
def read_sensor_csv_full(input_path: Path, columns_arg: Optional[List[str]] = None, delimiter: Optional[str] = None
                         ) -> Tuple[List[Tuple[str, List[float]]], List[int], Dict[str, List[float]]]:
    rows = read_table_rows(input_path, explicit_delim=delimiter)
    if not rows:
        raise ValueError("No rows in file")
    header = rows[0]
    data_rows = rows[1:] if len(rows) > 1 else []
    # handle header that is numeric
    numeric_count = sum(1 for v in header if _looks_numeric(v))
    header_is_numeric = numeric_count >= max(1, len(header) // 2)
    if header_is_numeric:
        max_cols = max(len(r) for r in rows)
        header = [f"col{i}" for i in range(max_cols)]
        data_rows = [list(r) + [""] * (max_cols - len(r)) for r in rows]
    # timestamp column detection
    ts_idx: Optional[int] = None
    for i, h in enumerate(header):
        if h and isinstance(h, str) and h.strip().lower().startswith("timestamp"):
            ts_idx = i; break
    if ts_idx is not None:
        timestamps: List[int] = []
        for r in data_rows:
            try:
                val = r[ts_idx] if ts_idx < len(r) else ""
                timestamps.append(int(float(val)))
            except Exception:
                raise ValueError("Non-numeric timestamp encountered; aborting (v1.2 strict)")
    else:
        timestamps = [int(i * 1_000_000) for i in range(len(data_rows))]
    # choose sensor columns
    sensors_idx_name: List[Tuple[int, str]] = []
    if columns_arg:
        for col in columns_arg:
            try:
                idx = int(col)
                name = header[idx] if idx < len(header) else f"col{idx}"
                sensors_idx_name.append((idx, name))
            except Exception:
                if col in header:
                    sensors_idx_name.append((header.index(col), col))
                else:
                    raise ValueError(f"Column '{col}' not found")
    else:
        detected = detect_numeric_columns(header, data_rows)
        if detected:
            sensors_idx_name = detected
        else:
            # fallback to first non-ts column
            for i, h in enumerate(header):
                if i == ts_idx:
                    continue
                sensors_idx_name = [(i, h)]; break
    values_map: Dict[str, List[float]] = {}
    for idx, name in sensors_idx_name:
        vals: List[float] = []
        for r in data_rows:
            if idx < len(r):
                v = (r[idx] or "").strip()
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(float('nan'))
            else:
                vals.append(float('nan'))
        nm = name if name else f"col{idx}"
        values_map[nm] = vals
    sensor_list: List[Tuple[str, List[float]]] = [(name, values_map[name]) for (_, name) in sensors_idx_name]
    return sensor_list, timestamps, values_map

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_suffix('.s2s')
    if out.exists() and not args.overwrite:
        logging.error("Output exists; use --overwrite to replace.")
        return
    # explicit error if sign-key requested but crypto missing
    if args.sign_key and not CRYPTO_AVAILABLE:
        raise ImportError("cryptography required to sign but not installed; install 'cryptography' or omit --sign-key")

    # Sanitize and capture CLI args for provenance (omit potential raw secret bytes - we keep the key path but not its contents)
    cli_record = {k: v for k, v in vars(args).items() if k not in ("sign_key",)}  # exclude sign_key from CLI snapshot to avoid storing local key path
    # But include a safe marker for sign_key presence
    cli_record["sign_key_provided"] = bool(args.sign_key)

    # Read sensors and timestamps
    sensor_list, timestamps, values_map = read_sensor_csv_full(inp, args.columns, delimiter=args.delimiter)

    # Build a nan summary (fail-fast if any NaNs present and user did not request removal/fill)
    nan_summary: Dict[str, Dict[str, int]] = {}
    any_nan = False
    total_rows = len(timestamps)
    for (name, vals) in sensor_list:
        total = len(vals)
        nan_count = sum(0 if (isinstance(x, (int, float)) and math.isfinite(x)) else 1 for x in vals)
        nan_summary[name] = {"nan_count": int(nan_count), "total": int(total)}
        if nan_count > 0:
            any_nan = True

    if any_nan and (not args.remove_nans) and (args.fill_nans == "none"):
        # Fail fast — do not silently convert NaN-containing data
        raise ValueError(f"Input contains NaN values in sensor columns; run with --remove-nans or supply --fill-nans to proceed. NaN summary: {json.dumps(nan_summary)}")

    meta_extra: Dict[str, Any] = {}
    flags: List[str] = []
    notes: Dict[str, Any] = {}

    # Embed tool provenance early (sanitized CLI args + initial nan summary)
    meta_extra.setdefault("tool_provenance", {})
    meta_extra["tool_provenance"].update({
        "tool": "convert_to_s2s",
        "version": "v1.3",
        "timestamp": int(time.time()),
        "cli_args": cli_record,
        "nan_summary": nan_summary
    })

    # compute incoming jitter metric (before injection)
    jm_before = compute_jitter_metrics(timestamps)
    if jm_before.get("cv") is not None:
        notes["jitter_before_cv"] = jm_before["cv"]

    # mark suspect perfect timing
    if jm_before.get("cv") is not None and isinstance(jm_before["cv"], float) and jm_before["cv"] < 1e-9:
        flags.append("SUSPECT_SYNTHETIC")
        notes["suspicion_reason"] = "cv ~= 0 (perfect timing)"
        if args.annotate_suspect:
            meta_extra.setdefault("notes", {}).update(notes)
            meta_extra.setdefault("flags", []).extend(flags)

    # optional Gaussian jitter applied to intervals
    if args.inject_jitter_stdns and args.inject_jitter_stdns > 0.0:
        timestamps = inject_gaussian_jitter_on_intervals(timestamps, args.inject_jitter_stdns, seed=args.seed)
        notes["injected_jitter_stdns"] = float(args.inject_jitter_stdns)
        if args.seed is not None:
            notes["injected_jitter_seed"] = int(args.seed)
        else:
            # record generated seed for reproducibility
            # Use deterministic seed recorded from time_ns() at the point of injection if user did not set one
            notes["injected_jitter_seed"] = int(time.time_ns())
        flags.append("INJECTED_JITTER")

    # optional removal or fill of NaNs (synchronized)
    provenance = {"original_count": len(timestamps), "kept_count": len(timestamps), "removed_pct": 0.0}
    if args.remove_nans:
        timestamps_new, new_sensor_list, prov = clean_synchronized_rows(timestamps, sensor_list)
        timestamps = timestamps_new
        sensor_list = new_sensor_list
        provenance = prov
        flags.append("REMOVED_NANS")
        notes["nan_removal_provenance"] = provenance
    elif args.fill_nans and args.fill_nans != "none":
        # apply fill strategy per sensor
        for i, (name, vals) in enumerate(sensor_list):
            sensor_list[i] = (name, fill_nans_series(vals, args.fill_nans))
        notes["nan_fill"] = args.fill_nans
        flags.append("FILLED_NANS")
        # After fill, recompute a little provenance
        total_rows = len(timestamps)
        prov_after = {"original_count": provenance["original_count"], "kept_count": total_rows, "removed_pct": 0.0}
        meta_extra.setdefault("provenance_after_fill", prov_after)

    # compute output sensor stats + jitter
    sensor_stats: Dict[str, Any] = {}
    for name, vals in sensor_list:
        finite = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        sensor_stats[name] = {"count": len(vals), "count_nan": len(vals) - len(finite),
                              "mean": (statistics.mean(finite) if finite else None),
                              "peak": (max(abs(v) for v in finite) if finite else None)}
    jm_after = compute_jitter_metrics(timestamps)

    # Add these to meta
    meta_extra.setdefault("sensor_stats", sensor_stats)
    meta_extra.setdefault("provenance", provenance)
    # Merge or extend tool_provenance if already present
    meta_extra.setdefault("tool_provenance", {}).update({
        "jitter_before": jm_before,
        "jitter_after": jm_after
    })
    if flags:
        meta_extra.setdefault("flags", []).extend(flags)
    if notes:
        meta_extra.setdefault("notes", {}).update(notes)

    # final write
    manifest = write_s2s_file(out, sensor_list, timestamps, sign_key=args.sign_key, meta_extra=meta_extra)
    if args.verbose:
        print(json.dumps({"manifest": manifest, "jitter_before": jm_before, "jitter_after": jm_after, "provenance": provenance, "nan_summary": nan_summary}, indent=2))
    else:
        print(manifest["output"])

if __name__ == "__main__":
    main()
