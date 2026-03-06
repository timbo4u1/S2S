#!/usr/bin/env python3
"""
s2s-certify  —  Physics certification for IMU CSV files
Usage:  s2s-certify yourfile.csv
"""
import sys, csv, json, os, time

TIER_COLORS = {
    "GOLD":     "\033[93m",   # yellow
    "SILVER":   "\033[96m",   # cyan
    "BRONZE":   "\033[33m",   # orange
    "REJECTED": "\033[91m",   # red
}
RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
DIM   = "\033[2m"

def find_columns(headers):
    """Auto-detect column names regardless of capitalisation or separator."""
    h = [c.lower().strip() for c in headers]
    def find(candidates):
        for c in candidates:
            for i, col in enumerate(h):
                if c in col:
                    return i
        return None

    t  = find(['time', 'ts', 'ns', 'ms', 'sec', 't'])
    ax = find(['acc_x', 'accel_x', 'ax', 'a_x', 'accelerometer_x', 'x_acc'])
    ay = find(['acc_y', 'accel_y', 'ay', 'a_y', 'accelerometer_y', 'y_acc'])
    az = find(['acc_z', 'accel_z', 'az', 'a_z', 'accelerometer_z', 'z_acc'])
    gx = find(['gyro_x', 'gyr_x', 'gx', 'g_x', 'gyroscope_x', 'x_gyro', 'wx'])
    gy = find(['gyro_y', 'gyr_y', 'gy', 'g_y', 'gyroscope_y', 'y_gyro', 'wy'])
    gz = find(['gyro_z', 'gyr_z', 'gz', 'g_z', 'gyroscope_z', 'z_gyro', 'wz'])
    return t, ax, ay, az, gx, gy, gz

def load_csv(path):
    with open(path, newline='') as f:
        # Sniff delimiter
        sample = f.read(2048); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        rows = list(reader)

    if not rows:
        print("Error: empty CSV"); sys.exit(1)

    headers = rows[0]
    t_i, ax_i, ay_i, az_i, gx_i, gy_i, gz_i = find_columns(headers)

    # If gyro not found, still run accel-only laws
    has_gyro = all(i is not None for i in [gx_i, gy_i, gz_i])
    has_accel = all(i is not None for i in [ax_i, ay_i, az_i])

    if not has_accel:
        print(f"\n{TIER_COLORS['REJECTED']}Could not find accelerometer columns.{RESET}")
        print(f"Columns found: {headers}")
        print("Expected columns like: acc_x, acc_y, acc_z  (or ax/ay/az)")
        sys.exit(1)

    timestamps, accel, gyro = [], [], []
    for row in rows[1:]:
        try:
            if t_i is not None:
                ts = float(row[t_i])
            else:
                ts = len(timestamps) * 0.01  # assume 100Hz
            timestamps.append(ts)
            accel.append([float(row[ax_i]), float(row[ay_i]), float(row[az_i])])
            if has_gyro:
                gyro.append([float(row[gx_i]), float(row[gy_i]), float(row[gz_i])])
        except (ValueError, IndexError):
            continue

    # Convert timestamps to nanoseconds if they look like seconds
    if timestamps and timestamps[-1] < 1e6:
        timestamps = [int(t * 1e9) for t in timestamps]
    else:
        timestamps = [int(t) for t in timestamps]

    imu = {
        "timestamps_ns": timestamps,
        "accel": accel,
    }
    if has_gyro:
        imu["gyro"] = gyro

    return imu, headers, has_gyro, len(accel)

def print_banner():
    print(f"""
{BOLD}╔══════════════════════════════════════════╗
║   S2S Physics Certification  v1.4.0      ║
║   github.com/timbo4u1/S2S                ║
╚══════════════════════════════════════════╝{RESET}""")

def print_result(result, filename, n_samples, has_gyro):
    tier  = result.get("physics_tier") or result.get("tier", "UNKNOWN")
    score = result.get("physics_score", 0)
    laws_passed = result.get("physics_laws_passed", [])
    laws_failed = result.get("physics_laws_failed", [])
    r     = result.get("imu_coupling_r", None)
    jerk  = result.get("jerk_p95_ms3", None)
    sig   = result.get("_signature", None)

    color = TIER_COLORS.get(tier, "\033[97m")

    print(f"\n{DIM}File:{RESET}    {filename}")
    print(f"{DIM}Samples:{RESET} {n_samples}  |  Gyro: {'✓' if has_gyro else '✗ (accel only)'}")
    print()
    print(f"  {BOLD}Result:  {color}{tier}{RESET}")
    print(f"  Score:   {BOLD}{score}/100{RESET}")
    if r is not None:
        print(f"  Coupling r:  {r:.3f}  {'✓' if r > 0.15 else '✗ low — synthetic-like'}")
    if jerk is not None:
        print(f"  Jerk P95:    {jerk:.1f} m/s³")
    print()

    if laws_passed:
        for law in laws_passed:
            print(f"  {GREEN}✓{RESET} {law.replace('_',' ')}")
    if laws_failed:
        for law in laws_failed:
            print(f"  {TIER_COLORS['REJECTED']}✗{RESET} {law.replace('_',' ')}")

    if sig:
        print(f"\n  {DIM}Ed25519 signed: {sig[:24]}…{RESET}")

    print()
    if tier == "GOLD":
        print(f"  {color}{BOLD}★ GOLD — all physics laws passed. Data is certified.{RESET}")
    elif tier == "SILVER":
        print(f"  {color}{BOLD}✓ SILVER — core laws passed. Suitable for training.{RESET}")
    elif tier == "BRONZE":
        print(f"  {color}~ BRONZE — marginal quality. Usable with caution.{RESET}")
    else:
        print(f"  {color}{BOLD}✗ REJECTED — physics violations detected.{RESET}")
        print(f"  {DIM}This data should not be used for training.{RESET}")
    print()

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h','--help'):
        print(__doc__)
        print("Example:")
        print("  s2s-certify my_imu_recording.csv")
        print("\nExpected CSV columns:")
        print("  timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z")
        print("  (column names are flexible — ax/ay/az, accel_x etc. all work)")
        sys.exit(0)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"File not found: {path}"); sys.exit(1)

    print_banner()
    print(f"  Certifying: {os.path.basename(path)}")
    print(f"  {DIM}Loading…{RESET}", end='\r')

    try:
        imu, headers, has_gyro, n = load_csv(path)
    except Exception as e:
        print(f"\nFailed to load CSV: {e}"); sys.exit(1)

    print(f"  Loaded {n} samples. Running physics checks…", end='\r')

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine
        t0 = time.time()
        result = PhysicsEngine().certify(imu)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.2f}s                        ")
        print_result(result, path, n, has_gyro)
    except Exception as e:
        print(f"\nCertification error: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
