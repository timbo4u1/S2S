#!/usr/bin/env python3
"""
S2S Run All — calls existing adapters for every dataset.
Usage: python3 run_all.py
"""
import os, subprocess, sys

BASE   = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

JOBS = [
    {
        "name":   "EMG Amputee",
        "script": os.path.join(BASE, "amputee_adapter.py"),
        "args":   ["--input", os.path.expanduser("~/S2S_Project/EMG_Amputee"),
                   "--out",   os.path.expanduser("~/S2S/s2s_dataset")],
    },
    {
        "name":   "NinaproDB5",
        "script": os.path.join(BASE, "certify_ninapro_db5.py"),
        "args":   [],
    },
    {
        "name":   "S2S Dataset",
        "script": os.path.join(BASE, "certify_s2s_dataset.py"),
        "args":   [],
    },
]

print("\n" + "═"*50)
print("  S2S RUN ALL — Batch Certify All Datasets")
print("═"*50 + "\n")

for job in JOBS:
    print(f"▶ {job['name']} ...")
    cmd = [PYTHON, job["script"]] + job["args"]
    result = subprocess.run(cmd, cwd=BASE)
    status = "✅" if result.returncode == 0 else "❌"
    print(f"{status} {job['name']} done (exit={result.returncode})\n")

print("═"*50)
print("  All done.")
print("═"*50 + "\n")
