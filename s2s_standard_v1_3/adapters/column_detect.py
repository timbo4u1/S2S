"""
Smart column detector for any sensor dataset.
No column names needed — detects sensor type from data statistics.
"""
import numpy as np
import re
from typing import Dict, List, Tuple


_UNIT_SCALES = {
    "accel": {
        "m/s2": 1.0,
        "g": 9.80665,
        "milli-g": 0.00980665,
    },
    "gyro": {
        "rad/s": 1.0,
        "deg/s": np.pi / 180.0,
        "mrad/s": 0.001,
        "mdps": np.pi / 180000.0,
    },
}


def _canonical_unit(unit, sensor):
    """Normalize a caller-supplied unit to a supported canonical unit."""
    if unit is None or sensor not in _UNIT_SCALES:
        return None

    value = str(unit).strip().lower()
    value = value.replace("²", "2").replace("^", "")
    value = re.sub(r"[\s_-]+", "", value)

    aliases = {
        "accel": {
            "m/s2": "m/s2",
            "m/sec2": "m/s2",
            "mps2": "m/s2",
            "ms2": "m/s2",
            "g": "g",
            "mg": "milli-g",
            "millig": "milli-g",
            "milligravity": "milli-g",
        },
        "gyro": {
            "rad/s": "rad/s",
            "rads": "rad/s",
            "radsec": "rad/s",
            "radian/s": "rad/s",
            "radians/s": "rad/s",
            "deg/s": "deg/s",
            "degree/s": "deg/s",
            "degrees/s": "deg/s",
            "dps": "deg/s",
            "degsec": "deg/s",
            "mrad/s": "mrad/s",
            "mrads": "mrad/s",
            "millirad/s": "mrad/s",
            "mdps": "mdps",
            "millideg/s": "mdps",
            "millidegree/s": "mdps",
            "millidegrees/s": "mdps",
        },
    }

    return aliases[sensor].get(value)


def _unit_from_header(header, sensor):
    """Recognize an explicit unit token in one column header."""
    value = str(header).strip().lower()
    value = value.replace("²", "2").replace("°", "deg")

    if sensor == "accel":
        if re.search(
            r"(?<![a-z0-9])(?:milli[\s_-]*g|mg)(?![a-z0-9])",
            value,
        ):
            return "milli-g"

        if re.search(
            r"m\s*/\s*s\s*\^?\s*2|mps2|msec2|m\s+s\s*\^\s*\(?\s*-?\s*2\s*\)?",
            value,
        ):
            return "m/s2"

        if re.search(r"(?<![a-z0-9])g(?![a-z0-9])", value):
            return "g"

    elif sensor == "gyro":
        if re.search(
            r"(?<![a-z0-9])(?:mdps|milli[\s_-]*degrees?[\s/_-]*s)"
            r"(?![a-z0-9])",
            value,
        ):
            return "mdps"

        if re.search(
            r"(?<![a-z0-9])(?:mrad[\s/_-]*s|millirad[\s/_-]*s)"
            r"(?![a-z0-9])",
            value,
        ):
            return "mrad/s"

        if re.search(
            r"(?<![a-z0-9])(?:dps|deg(?:ree)?s?[\s/_-]*s)"
            r"(?![a-z0-9])",
            value,
        ):
            return "deg/s"

        if re.search(
            r"(?<![a-z0-9])rad[\s/_-]*s(?![a-z0-9])",
            value,
        ):
            return "rad/s"

    return None


def _resolve_unit(explicit_unit, header_names, columns, sensor):
    """
    Resolve units from explicit caller input or consistent axis headers.

    Magnitude-based inference is deliberately not used.
    """
    if sensor not in _UNIT_SCALES:
        raise ValueError("Unknown sensor type: {}".format(sensor))

    if sensor == "gyro" and not columns:
        return {
            "unit": "not_present",
            "status": "not_detected",
            "source": None,
            "scale_to_si": None,
        }

    if len(columns) != 3:
        return {
            "unit": "unit_unknown",
            "status": "unit_unknown",
            "source": None,
            "scale_to_si": None,
        }

    if explicit_unit is not None:
        canonical = _canonical_unit(explicit_unit, sensor)
        if canonical is None:
            return {
                "unit": "unit_unknown",
                "status": "invalid_explicit",
                "source": "argument",
                "scale_to_si": None,
            }

        return {
            "unit": canonical,
            "status": "explicit",
            "source": "argument",
            "scale_to_si": float(_UNIT_SCALES[sensor][canonical]),
        }

    if header_names and all(
        0 <= column < len(header_names) for column in columns
    ):
        header_units = [
            _unit_from_header(header_names[column], sensor)
            for column in columns
        ]

        if (
            all(unit is not None for unit in header_units)
            and len(set(header_units)) == 1
        ):
            canonical = header_units[0]
            return {
                "unit": canonical,
                "status": "inferred_from_header",
                "source": "column_headers",
                "scale_to_si": float(_UNIT_SCALES[sensor][canonical]),
            }

    return {
        "unit": "unit_unknown",
        "status": "unit_unknown",
        "source": None,
        "scale_to_si": None,
    }


def detect_columns(data: np.ndarray, col_names: List[str] = None) -> Dict:
    """
    Detect sensor types from data statistics.
    Works even with no column names.
    
    Returns:
        {
          'accel': [col_indices],
          'gyro':  [col_indices], 
          'emg':   [col_indices],
          'ppg':   [col_indices],
          'timestamp': col_index or None,
          'confidence': 0-1
        }
    """
    result = {'accel':[], 'gyro':[], 'emg':[], 'ppg':[], 
              'timestamp': None, 'unknown': [], 'confidence': 0.0}
    
    n_cols = data.shape[1]
    votes = []
    
    for i in range(n_cols):
        col = data[:, i]
        # Remove NaN
        col = col[~np.isnan(col)]
        if len(col) < 10:
            continue
            
        median_abs = float(np.median(np.abs(col)))
        std = float(np.std(col))
        mean = float(np.mean(col))
        max_val = float(np.max(np.abs(col)))
        
        # Timestamp: monotonically increasing, huge values
        diffs = np.diff(col)
        is_monotonic = np.all(diffs >= 0)
        if is_monotonic and (median_abs > 100 or (median_abs > 0 and np.all(diffs >= 0) and np.all(diffs < 1e8))):
            result['timestamp'] = i
            votes.append((i, 'timestamp', 0.95))
            continue
        
        # Accelerometer: gravity ~9.81 m/s²
        # or milli-g: ~1000 (9810 milli-g), divide by 1000
        if 7.0 <= median_abs <= 12.0 and std < 5.0:
            result['accel'].append(i)
            votes.append((i, 'accel_ms2', 0.9))
        elif 800 <= median_abs <= 1200 and std < 500:
            result['accel'].append(i)
            votes.append((i, 'accel_millig', 0.85))
        
        # Gyroscope: near zero at rest, moderate range
        elif median_abs < 1.0 and std < 2.0 and max_val < 20.0:
            result['gyro'].append(i)
            votes.append((i, 'gyro_rads', 0.8))
        elif median_abs < 100 and std < 500 and max_val < 5000:
            result['gyro'].append(i)
            votes.append((i, 'gyro_scaled', 0.7))
        
        # EMG: zero-mean, high variance, spiky
        elif abs(mean) < 0.1 * std and std > 0.01 and max_val > 3 * std:
            result['emg'].append(i)
            votes.append((i, 'emg', 0.75))
        
        # PPG: positive values, periodic
        elif mean > 0 and std / mean < 0.3 and max_val < 10000:
            result['ppg'].append(i)
            votes.append((i, 'ppg', 0.6))
        
        else:
            result['unknown'].append(i)
    
    # Confidence: how many columns were classified
    classified = len(votes)
    total = n_cols
    result['confidence'] = round(classified / total, 2) if total > 0 else 0
    result['_votes'] = votes
    
    return result


def _sniff_delimiter(filepath, max_lines=20):
    """
    Detect the field delimiter from file contents using quote-aware CSV parsing.
    Returns one of ',', ';', '\\t', ' ', or None if undetermined.
    """
    import csv
    import io
    from collections import Counter

    candidates = [',', ';', '\\t']

    # Read sample lines for analysis
    sample_lines = []
    ws_lines = 0
    n_lines = 0

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                if n_lines > max_lines:
                    break
                sample_lines.append(line)
                if len(line.split()) > 1:
                    ws_lines += 1
    except (IOError, OSError):
        return None

    if n_lines == 0:
        return None

    # For each candidate delimiter, parse with csv.reader and analyze field counts
    best_delim = None
    best_score = 0.0

    for delim in candidates:
        try:
            content = '\n'.join(sample_lines)
            reader = csv.reader(io.StringIO(content), delimiter=delim)
            field_counts = [len(row) for row in reader]
        except Exception:
            continue

        if not field_counts:
            continue

        # Check for multi-column structure: EVERY row must have at least 2 fields
        # This ensures the delimiter produces consistent structure across all rows
        # (including header). Wrong delimiters often produce single-column rows
        # for some records (e.g., header without the competing delimiter).
        min_fields = min(field_counts)
        if min_fields < 2:
            continue

        # Find most common field count for consistency check
        most_common_count, most_common_freq = Counter(field_counts).most_common(1)[0]

        # Require 80% consistency (retain existing threshold)
        consistency = most_common_freq / len(field_counts)
        if consistency < 0.8:
            continue

        # Score: prefer more fields (minimum) and higher consistency
        # Use minimum field count to ensure robust multi-column structure
        score = min_fields * consistency

        if score > best_score:
            best_score = score
            best_delim = delim

    # Require minimum viable score (at least 2 fields minimum with consistency)
    if best_delim is not None and best_score >= 2.0:
        return best_delim

    # Fallback to whitespace
    if ws_lines >= n_lines * 0.8:
        return ' '

    return None


def _resolve_delimiter(filepath, delimiter=None):
    """
    Resolve the delimiter for file parsing.
    Returns (delimiter, error_message).
    If delimiter is explicitly provided, it is used as-is.
    If None, auto-detects from file contents.
    """
    if delimiter is not None:
        return delimiter, None

    detected = _sniff_delimiter(filepath)
    if detected is None:
        return None, (
            "Could not determine field delimiter from file contents. "
            "Please provide the delimiter explicitly."
        )
    return detected, None


def _numpy_delimiter(delim):
    """Convert internal delimiter to np.genfromtxt format."""
    if delim == ' ':
        return None  # numpy whitespace mode handles repeated spaces
    return delim


def _is_numeric_field(field):
    """
    Check if a field is numeric or a recognized numeric special value.

    Returns True for:
    - Standard numeric values (int, float)
    - Scientific notation (lowercase 'e' and uppercase 'E')
    - NaN, Inf, +Inf, -Inf (case-insensitive)
    - Empty fields (treated as missing numeric values)

    Returns False for genuine text (column names, labels, etc.).
    """
    field = field.strip().strip('"').strip("'")
    if not field:
        return True  # Empty field = missing numeric value
    try:
        float(field)
        return True
    except (ValueError, TypeError):
        return False


def _has_header_row(filepath, delimiter):
    """
    Determine if the first row is a header by parsing fields and checking
    if they are numeric. Returns True if header, False if numeric data.

    Uses the resolved delimiter for consistent field parsing.
    For whitespace-delimited data, splits on whitespace.
    For comma/semicolon/tab, splits on the delimiter character.
    """
    # Read first line
    with open(filepath) as f:
        first_line = f.readline().strip()

    if not first_line:
        return False

    # Parse fields based on delimiter
    if delimiter == ' ':
        fields = first_line.split()
    elif isinstance(delimiter, str) and len(delimiter) == 1:
        fields = first_line.split(delimiter)
    else:
        fields = first_line.split(str(delimiter))

    # A row is a header if ANY field is non-numeric
    # A row is numeric data if ALL fields are numeric or missing
    return not all(_is_numeric_field(f) for f in fields)


def detect_from_file(filepath: str, delimiter: str = None) -> Dict:
    """Auto-detect sensor columns from any CSV or space-delimited file."""
    delim, err = _resolve_delimiter(filepath, delimiter)
    if err:
        return {
            'accel': [], 'gyro': [], 'emg': [], 'ppg': [],
            'timestamp': None, 'unknown': [], 'confidence': 0.0,
            'file': filepath, 'shape': (0, 0), 'error': err,
        }

    np_delim = _numpy_delimiter(delim)

    # Read first 500 rows for speed
    # Use numeric-aware header detection to avoid discarding scientific notation
    _has_hdr = _has_header_row(filepath, delim)
    data = np.genfromtxt(filepath, delimiter=np_delim, max_rows=500,
                         skip_header=1 if _has_hdr else 0, ndmin=2)

    result = detect_columns(data)
    result['file'] = filepath
    result['shape'] = data.shape
    return result


def certify_file(filepath: str, segment: str = "forearm",
                 delimiter: str = None, max_windows: int = 100,
                 accel_unit: str = None, gyro_unit: str = None,
                 accel_columns: List[int] = None,
                 gyro_columns: List[int] = None) -> dict:
    """
    Certify any sensor file with zero configuration.
    Auto-detects accel, gyro, EMG columns from data statistics.
    
    Args:
        filepath: path to CSV or space-delimited sensor file
        segment:  body segment (forearm, upper_arm, hand, walking)
        delimiter: auto-detected if None
        max_windows: maximum windows to certify
        
    Returns:
        dict with tier, score, pass_rate, law_details, detected_columns
    
    Example:
        from s2s_standard_v1_3.adapters.column_detect import certify_file
        result = certify_file("my_sensor_data.csv", segment="forearm")
        print(result["tier"], result["pass_rate"])
    """
    import sys
    sys.path.insert(0, ".")
    from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

    # Resolve delimiter ONCE for consistent use across detection,
    # header parsing, data loading, and certification.
    _delim, _delim_err = _resolve_delimiter(filepath, delimiter)
    if _delim_err:
        return {
            "error": _delim_err,
            "tier": "N/A",
        }

    np_delim = _numpy_delimiter(_delim)

    # Pass resolved delimiter to detection so both use the same value.
    detection = detect_from_file(filepath, _delim)

    # Detect if first row is a header using numeric-aware detection
    # This ensures both detect_from_file and certify_file make identical decisions
    _has_header = _has_header_row(filepath, _delim)
    skip = 1 if _has_header else 0

    # Read column labels only when the file actually has a header.
    header_names = []
    if _has_header:
        # Re-read first line for header names
        with open(filepath) as _f:
            _first = _f.readline().strip()
        if _delim == " ":
            header_names = _first.split()
        elif isinstance(_delim, str) and len(_delim) == 1:
            import csv
            header_names = next(csv.reader([_first], delimiter=_delim))
        else:
            header_names = _first.split(str(_delim))

        header_names = [
            str(name).strip().strip('"').strip("'")
            for name in header_names
        ]

    # Streaming chunk reader — works on files of any size, low RAM usage
    # Reads chunk_size rows at a time instead of loading entire file
    chunk_size = max_windows * 256  # 256 samples per window
    data = np.genfromtxt(filepath,
                         delimiter=np_delim,
                         skip_header=skip,
                         max_rows=chunk_size,
                         ndmin=2)

    def _validate_columns(columns, label, allow_empty=False,
                          require_three=True):
        if not isinstance(columns, (list, tuple)):
            raise ValueError(
                "{}_columns must be a list or tuple".format(label)
            )

        if not columns and allow_empty:
            return []

        if require_three and len(columns) != 3:
            raise ValueError(
                "{} requires exactly three distinct axis columns".format(
                    label.capitalize()
                )
            )

        normalized = []

        for column in columns:
            if (
                isinstance(column, (bool, np.bool_))
                or not isinstance(column, (int, np.integer))
            ):
                raise ValueError(
                    "{} column indices must be integers".format(label)
                )

            column = int(column)

            if column < 0 or column >= data.shape[1]:
                raise ValueError(
                    "{} column index {} is outside the data width {}".format(
                        label, column, data.shape[1]
                    )
                )

            if column in normalized:
                raise ValueError(
                    "{} column indices must be distinct".format(label)
                )

            normalized.append(column)

        return normalized

    try:
        if accel_columns is None:
            detected_accel = list(detection.get("accel", []))

            if not detected_accel:
                return {
                    "error": "No accelerometer columns detected",
                    "detected": detection,
                    "tier": "N/A",
                }

            if len(detected_accel) != 3:
                return {
                    "error": (
                        "Auto-detection found {} accelerometer column(s). "
                        "Provide accel_columns=[x_column, y_column, "
                        "z_column] explicitly."
                    ).format(len(detected_accel)),
                    "detected": detection,
                    "tier": "N/A",
                }

            acc_cols = _validate_columns(
                detected_accel, "accel"
            )
        else:
            acc_cols = _validate_columns(
                accel_columns, "accel"
            )

        if gyro_columns is None:
            # Use automatic gyro detection only as a candidate.
            # An incomplete or ambiguous set remains unit_unknown.
            detected_gyro = list(detection.get("gyro", []))
            gyro_cols = _validate_columns(
                detected_gyro,
                "gyro",
                allow_empty=True,
                require_three=False,
            )
        else:
            gyro_cols = _validate_columns(
                gyro_columns,
                "gyro",
                allow_empty=True,
            )

    except ValueError as exc:
        return {
            "error": str(exc),
            "detected": detection,
            "tier": "N/A",
        }

    # Reject cross-sensor column overlap: a source column cannot be
    # assigned to both accelerometer and gyroscope axes.
    overlap = sorted(set(acc_cols) & set(gyro_cols))
    if overlap:
        return {
            "error": (
                "Accelerometer and gyroscope columns overlap: "
                f"{overlap}"
            ),
            "detected": detection,
            "tier": "N/A",
        }

    # Keep the original row sequence. Never remove acceleration rows
    # because a separate gyro channel is missing.
    acc_raw = np.asarray(data[:, acc_cols], dtype=float)
    accel_valid_rows = np.isfinite(acc_raw).all(axis=1)

    if not np.any(accel_valid_rows):
        return {
            "error": "No complete accelerometer rows found",
            "detected": detection,
            "tier": "N/A",
        }

    unit_metadata = {
        "accel": _resolve_unit(
            accel_unit, header_names, acc_cols, "accel"
        ),
        "gyro": _resolve_unit(
            gyro_unit, header_names, gyro_cols, "gyro"
        ),
    }

    invalid_explicit = [
        name for name, metadata in unit_metadata.items()
        if metadata["status"] == "invalid_explicit"
    ]
    if invalid_explicit:
        return {
            "error": (
                "Unsupported explicit unit for: "
                + ", ".join(invalid_explicit)
            ),
            "detected": detection,
            "tier": "N/A",
            "unit_metadata": unit_metadata,
        }

    accel_metadata = unit_metadata["accel"]
    if accel_metadata["status"] not in (
        "explicit", "inferred_from_header"
    ):
        return {
            "error": (
                "Accelerometer unit unknown. Pass accel_unit='m/s2', "
                "'g', or 'milli-g', or provide the same recognized unit "
                "in all three selected acceleration column headers."
            ),
            "detected": detection,
            "tier": "N/A",
            "unit_metadata": unit_metadata,
        }

    # Convert acceleration only after its unit has been established.
    acc = acc_raw * accel_metadata["scale_to_si"]

    # Unknown gyro units are not converted or passed to the physics engine.
    # Acceleration-only assessment remains possible.
    gyro = None
    gyro_metadata = unit_metadata["gyro"]

    if (
        len(gyro_cols) == 3
        and gyro_metadata["status"] in (
            "explicit", "inferred_from_header"
        )
    ):
        gyro_raw = np.asarray(data[:, gyro_cols], dtype=float)
        gyro = gyro_raw * gyro_metadata["scale_to_si"]

    engine = PhysicsEngine()
    window = 256
    step   = 256

    # Detect timestamp units from adjacent source rows.
    # Preserve the source clock; do not reset it to zero.
    ts_candidates = [
        v[0] for v in detection.get("_votes", [])
        if "timestamp" in v[1]
    ]
    ts_col = min(ts_candidates) if ts_candidates else None

    timestamp_scale = None
    timestamp_source = "synthesized_row_index"
    timestamp_unit = "synthesized"
    hz = 30.0

    if ts_col is not None:
        ts_sample = np.asarray(data[:500, ts_col], dtype=float)

        if len(ts_sample) >= 3:
            adjacent_valid = (
                np.isfinite(ts_sample[:-1]) &
                np.isfinite(ts_sample[1:])
            )
            diffs = np.diff(ts_sample)[adjacent_valid]
            positive_diffs = diffs[diffs > 0]

            if len(positive_diffs):
                med_diff = float(np.median(positive_diffs))

                if med_diff > 1e6:
                    timestamp_scale = 1.0
                    timestamp_unit = "nanoseconds"
                    hz = 1e9 / med_diff
                elif med_diff > 1000:
                    timestamp_scale = 1e3
                    timestamp_unit = "microseconds"
                    hz = 1e6 / med_diff
                elif med_diff > 1:
                    timestamp_scale = 1e6
                    timestamp_unit = "milliseconds"
                    hz = 1e3 / med_diff
                else:
                    timestamp_scale = 1e9
                    timestamp_unit = "seconds"
                    hz = 1.0 / med_diff

                timestamp_source = "source_column"

    hz = max(1.0, min(round(hz, 1), 10000.0))

    tiers = []
    scores = []
    skipped_invalid_accel = 0
    skipped_invalid_timestamps = 0
    gyro_window_counts = {
        "complete": 0,
        "partial": 0,
        "absent": 0,
        "invalid": 0,
        "unit_unknown": 0,
    }

    # +1 includes a final window when the input contains exactly
    # window samples remaining.
    n_starts = min(len(data) - window + 1, max_windows * step)

    for start in range(0, max(0, n_starts), step):
        end = start + window

        chunk_a_raw = acc[start:end]
        if len(chunk_a_raw) != window or not np.isfinite(chunk_a_raw).all():
            # Do not compress the stream by deleting invalid accel rows.
            skipped_invalid_accel += 1
            continue

        chunk_a = chunk_a_raw.tolist()

        if timestamp_source == "source_column":
            source_ts = np.asarray(data[start:end, ts_col], dtype=float)

            if not np.isfinite(source_ts).all():
                skipped_invalid_timestamps += 1
                continue

            timestamps_ns = np.rint(
                source_ts * timestamp_scale
            ).astype(np.int64).tolist()
        else:
            timestamps_ns = np.rint(
                np.arange(start, end, dtype=float) * (1e9 / hz)
            ).astype(np.int64).tolist()

        imu_raw = {
            "timestamps_ns": timestamps_ns,
            "accel": chunk_a,
        }

        gyro_samples_present = 0
        gyro_availability = "absent"

        if gyro_metadata["status"] == "unit_unknown":
            # Sensor columns exist, but their physical unit is unverified.
            gyro_availability = "unit_unknown"

        elif gyro is not None:
            chunk_g_raw = gyro[start:end]
            gyro_valid = np.isfinite(chunk_g_raw).all(axis=1)
            gyro_samples_present = int(gyro_valid.sum())

            if gyro_samples_present == window:
                imu_raw["gyro"] = chunk_g_raw.tolist()
                gyro_availability = "complete"
            elif gyro_samples_present > 0:
                # Omit the entire gyro stream for this window. Do not
                # fabricate values or pass a shortened, misaligned stream.
                gyro_availability = "partial"
            else:
                # Gyro columns exist with known units, but no valid samples.
                # This is distinct from 'absent' (no gyro columns at all).
                gyro_availability = "invalid"

        cert = engine.certify(imu_raw, segment=segment)

        cert["_gyro_availability"] = gyro_availability
        cert["_gyro_samples_present"] = gyro_samples_present
        cert["_gyro_samples_expected"] = window

        gyro_window_counts[gyro_availability] += 1
        tiers.append(cert["tier"])
        scores.append(cert["physical_law_score"])

    if not tiers:
        return {
            "error": "No windows processed",
            "tier": "N/A",
            "unit_metadata": unit_metadata,
        }

    from collections import Counter
    counts = Counter(tiers)
    certified = counts["GOLD"] + counts["SILVER"] + counts["BRONZE"]
    pass_rate = certified / len(tiers)

    # Overall tier by majority
    if counts["GOLD"] > len(tiers) * 0.5:
        summary = "GOLD"
    elif counts["REJECTED"] > len(tiers) * 0.3:
        summary = "REJECTED"
    elif pass_rate > 0.7:
        summary = "SILVER"
    else:
        summary = "BRONZE"

    return {
        "tier":             summary,
        "pass_rate":        round(pass_rate, 3),
        "mean_score":       round(sum(scores)/len(scores), 1),
        "total_windows":    len(tiers),
        "certified":        certified,
        "rejected":         counts["REJECTED"],
        "tier_counts":      dict(counts),
        "detected_columns": {
            "accel":      acc_cols,
            "gyro":       gyro_cols,
            "confidence": detection["confidence"]
        },
        "unit_metadata": unit_metadata,
        "detected_hz":      hz,
        "segment":          segment,
        "gyro_window_availability": gyro_window_counts,
        "windows_skipped_invalid_accel": skipped_invalid_accel,
        "gyro_columns_usable": gyro is not None,
        "timestamp_source": timestamp_source,
        "timestamp_unit": timestamp_unit,
        "timestamp_column": ts_col,
        "windows_skipped_invalid_timestamps": skipped_invalid_timestamps,
    }
