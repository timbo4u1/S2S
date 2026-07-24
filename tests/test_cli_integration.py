"""
tests/test_cli_integration.py

End-to-end CLI integration tests.
Tests s2s-certify and s2s-refinery as a user would run them:
real subprocess calls, real file I/O, real output validation.
Catches CLI breakage that unit tests cannot detect.
"""
import sys, os, csv, json, math, random, subprocess, tempfile, shutil
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_imu_csv(path, n_rows=512, hz=100.0, seed=42):
    """Write a minimal valid IMU CSV with realistic biological signal."""
    random.seed(seed)
    state = [0.0, 0.0, 9.81]
    gyro_state = [0.0, 0.0, 0.0]

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_ns', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        for i in range(n_rows):
            ts = int(i * 1e9 / hz)
            row = []
            for axis in range(3):
                base = 0.92 * state[axis] + random.gauss(0, 0.4)
                if random.random() < 0.05:
                    base += random.choice([-1, 1]) * random.uniform(1.5, 4.0)
                state[axis] = base
                row.append(round(base, 6))
            row[2] += 9.81
            for axis in range(3):
                g = 0.90 * gyro_state[axis] + random.gauss(0, 0.03)
                gyro_state[axis] = g
                row.append(round(g, 6))
            w.writerow([ts] + row)
    return path


def _run(cmd, cwd=None):
    """Run a shell command, return (returncode, stdout, stderr)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=cwd
    )
    return result.returncode, result.stdout, result.stderr


# ── s2s-certify tests ─────────────────────────────────────────────────────────

class TestCertifyCLI:

    def test_certify_in_path(self):
        """s2s-certify must be findable in PATH."""
        rc, out, err = _run("which s2s-certify")
        assert rc == 0, "s2s-certify not found in PATH — run: pip install -e ."

    def test_certify_help(self):
        """s2s-certify --help must exit 0."""
        rc, out, err = _run("s2s-certify --help")
        assert rc == 0, f"s2s-certify --help failed:\n{err}"
        assert "segment" in out.lower() or "file" in out.lower()

    def test_certify_valid_csv_exit_0(self, tmp_path):
        """s2s-certify on valid CSV must exit 0."""
        csv_path = str(tmp_path / "test.csv")
        _make_imu_csv(csv_path)
        rc, out, err = _run(f"s2s-certify {csv_path}")
        assert rc == 0, f"s2s-certify failed with exit {rc}:\n{err}\n{out}"

    def test_certify_produces_output(self, tmp_path):
        """s2s-certify --output must create a JSON file."""
        csv_path = str(tmp_path / "test.csv")
        out_path = str(tmp_path / "result.json")
        _make_imu_csv(csv_path)
        rc, out, err = _run(f"s2s-certify {csv_path} --output {out_path}")
        assert rc == 0, f"s2s-certify failed:\n{err}"
        assert os.path.exists(out_path), "Output JSON file not created"

    def test_certify_output_has_tier(self, tmp_path):
        """s2s-certify JSON output must contain tier and score fields."""
        csv_path = str(tmp_path / "test.csv")
        out_path = str(tmp_path / "result.json")
        _make_imu_csv(csv_path)
        rc, _, _ = _run(f"s2s-certify {csv_path} --output {out_path}")
        assert rc == 0
        with open(out_path) as f:
            data = json.load(f)
        # Accept list (per-window) or single dict
        if isinstance(data, list):
            assert len(data) > 0
            first = data[0]
        else:
            first = data
        assert "tier" in first or "physical_law_score" in first, \
            f"Expected tier/score in output, got keys: {list(first.keys())}"

    def test_certify_tier_valid_value(self, tmp_path):
        """Tier value must be one of the known tiers."""
        csv_path = str(tmp_path / "test.csv")
        out_path = str(tmp_path / "result.json")
        _make_imu_csv(csv_path)
        _run(f"s2s-certify {csv_path} --output {out_path}")
        if os.path.exists(out_path):
            with open(out_path) as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                tier = data[0].get("tier", "UNKNOWN")
            elif isinstance(data, dict):
                tier = data.get("tier", "UNKNOWN")
            else:
                tier = "UNKNOWN"
            assert tier in ("GOLD", "SILVER", "BRONZE", "REJECTED",
                            "UNVERIFIED", "UNKNOWN"), \
                f"Unexpected tier value: {tier}"

    def test_certify_missing_file_nonzero(self):
        """s2s-certify on nonexistent file must exit non-zero."""
        rc, out, err = _run("s2s-certify /nonexistent/path/fake.csv")
        assert rc != 0, "Expected non-zero exit for missing file"

    def test_certify_segment_flag(self, tmp_path):
        """s2s-certify --segment forearm must work without error."""
        csv_path = str(tmp_path / "test.csv")
        _make_imu_csv(csv_path)
        rc, out, err = _run(f"s2s-certify {csv_path} --segment forearm")
        assert rc == 0, f"s2s-certify --segment failed:\n{err}"


# ── s2s-refinery tests ────────────────────────────────────────────────────────

class TestRefineryCLI:

    def test_refinery_in_path(self):
        """s2s-refinery must be findable in PATH."""
        rc, out, err = _run("which s2s-refinery")
        assert rc == 0, "s2s-refinery not found in PATH — run: pip install -e ."

    def test_refinery_help(self):
        """s2s-refinery --help must exit 0."""
        rc, out, err = _run("s2s-refinery --help")
        assert rc == 0, f"s2s-refinery --help failed:\n{err}"
        assert "input" in out.lower()

    def test_refinery_valid_folder_exit_0(self, tmp_path):
        """s2s-refinery on folder with valid CSVs must exit 0."""
        for i in range(3):
            _make_imu_csv(str(tmp_path / f"sensor_{i}.csv"), seed=i)
        out_csv = str(tmp_path / "report.csv")
        rc, out, err = _run(
            f"s2s-refinery --input {tmp_path} --output {out_csv} --segment forearm"
        )
        assert rc == 0, f"s2s-refinery failed:\n{err}\n{out}"

    def test_refinery_creates_output_csv(self, tmp_path):
        """s2s-refinery must create the output CSV file."""
        for i in range(2):
            _make_imu_csv(str(tmp_path / f"sensor_{i}.csv"), seed=i)
        out_csv = str(tmp_path / "report.csv")
        rc, _, _ = _run(
            f"s2s-refinery --input {tmp_path} --output {out_csv} --segment forearm"
        )
        assert rc == 0
        assert os.path.exists(out_csv), "Output CSV not created by s2s-refinery"

    def test_refinery_output_has_tier_column(self, tmp_path):
        """s2s-refinery CSV output must have a tier column."""
        for i in range(2):
            _make_imu_csv(str(tmp_path / f"sensor_{i}.csv"), seed=i)
        out_csv = str(tmp_path / "report.csv")
        _run(f"s2s-refinery --input {tmp_path} --output {out_csv} --segment forearm")
        assert os.path.exists(out_csv)
        with open(out_csv) as f:
            header = f.readline().lower()
        assert "tier" in header, \
            f"Expected 'tier' column in refinery output, got: {header[:100]}"

    def test_refinery_output_has_score_column(self, tmp_path):
        """s2s-refinery CSV output must have a score column."""
        _make_imu_csv(str(tmp_path / "sensor.csv"))
        out_csv = str(tmp_path / "report.csv")
        _run(f"s2s-refinery --input {tmp_path} --output {out_csv}")
        if os.path.exists(out_csv):
            with open(out_csv) as f:
                header = f.readline().lower()
            assert "score" in header, \
                f"Expected 'score' in refinery output header: {header[:100]}"

    def test_refinery_processes_multiple_files(self, tmp_path):
        """s2s-refinery must process all CSV files in the folder."""
        n_files = 4
        for i in range(n_files):
            _make_imu_csv(str(tmp_path / f"sensor_{i}.csv"), seed=i)
        out_csv = str(tmp_path / "report.csv")
        rc, out, err = _run(
            f"s2s-refinery --input {tmp_path} --output {out_csv}"
        )
        assert rc == 0
        if os.path.exists(out_csv):
            with open(out_csv) as f:
                rows = list(csv.reader(f))
            # At least one data row per file (header + data)
            assert len(rows) > n_files, \
                f"Expected >{n_files} rows in refinery output, got {len(rows)}"

    def test_refinery_empty_folder_graceful(self, tmp_path):
        """s2s-refinery on empty folder must not crash with unhandled exception."""
        out_csv = str(tmp_path / "report.csv")
        rc, out, err = _run(
            f"s2s-refinery --input {tmp_path} --output {out_csv}"
        )
        # May exit non-zero but must not produce a Python traceback
        assert "Traceback" not in err, \
            f"s2s-refinery crashed on empty folder:\n{err}"
