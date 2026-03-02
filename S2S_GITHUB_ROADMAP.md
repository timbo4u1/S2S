# S2S Development Roadmap
> Copy each section below as a **GitHub Issue**. Label with the milestone tag shown.
> When done → close the issue + check the box here.

---

## 🔴 MILESTONE 1 — Repo Hygiene (This Week)
*Goal: Make the repo look alive and trustworthy to any visitor.*

### Issue #1 — Push full commit history
**Label:** `housekeeping` `priority:critical`
```
Your local repo has 19 commits. GitHub shows 1.
This makes the project look abandoned before anyone reads the code.

Steps:
- [ ] git log --oneline  (confirm 19 local commits exist)
- [ ] git push origin main  (push full history)
- [ ] Verify GitHub shows all commits in history tab
```
**Can Claude help:** ✅ Yes — can write the exact git commands if anything goes wrong

---

### Issue #2 — Add missing local files
**Label:** `housekeeping`
```
Files confirmed locally but not on GitHub:
- [ ] qr.png
- [ ] wisdm_adapter.py
- [ ] train_classifier.py
- [ ] model/s2s_domain_classifier.json
- [ ] model/training_report.json
- [ ] docs/index.html
- [ ] docs/pose.html

Steps:
- [ ] git add .
- [ ] git commit -m "add: missing files from local — wisdm adapter, classifier, docs"
- [ ] git push origin main
```
**Can Claude help:** ✅ Yes

---

### Issue #3 — Add CONTEXT.md (AI assistant briefing file)
**Label:** `housekeeping` `documentation`
```
A file at root called CONTEXT.md that describes the full project
so any AI assistant (Claude, Copilot, etc.) can be briefed instantly
by pasting its contents — no re-explaining needed.

Contents should include:
- [ ] What S2S is in 3 sentences
- [ ] The 7 laws (names + what they check)
- [ ] Module list with one-line descriptions
- [ ] Current known weaknesses
- [ ] Roadmap link

See attached CONTEXT.md file generated alongside this roadmap.
```
**Can Claude help:** ✅ Yes — CONTEXT.md generated in this session, ready to commit

---

## 🟠 MILESTONE 2 — Tests & Trust (Week 1-2)
*Goal: Any technical reviewer can verify the laws work.*

### Issue #4 — Write pytest unit tests for all 7 laws
**Label:** `testing` `priority:high`
```
No tests = no trust. Each law needs a basic pass/fail test with known data.

Structure:
tests/
  test_newton_law.py
  test_rigid_body.py
  test_resonance.py
  test_jerk_bounds.py
  test_imu_consistency.py
  test_bcg_heartbeat.py
  test_joule_heating.py
  test_physics_engine.py   (integration: certify full sample)

Each test file needs:
- [ ] test_passes_real_data()   — known good iPhone IMU sample
- [ ] test_fails_synthetic()    — obviously fake data (zeros, random noise)
- [ ] test_edge_case()          — boundary value

Total: ~21 tests minimum
```
**Can Claude help:** ✅ Yes — can write all test files given the source code

---

### Issue #5 — Add GitHub Actions CI badge
**Label:** `testing` `devops`
```
Green CI badge = instant credibility signal to visitors.

Steps:
- [ ] Create .github/workflows/tests.yml
- [ ] Run: pytest tests/ on push to main
- [ ] Python versions matrix: 3.9, 3.10, 3.11
- [ ] Add badge to README: [![Tests](…)]

Template workflow provided in this issue body.
```
**Can Claude help:** ✅ Yes — can write the full workflow YAML

---

## 🟡 MILESTONE 3 — Fix the Classifier (Week 2-3)
*Goal: Raise domain classifier from 65.9% → target 85%+*

### Issue #6 — Retrain domain classifier using S2S physics vectors
**Label:** `ml` `priority:high`
```
Current accuracy: 65.9% (5-domain classifier)
Problem: likely trained on raw IMU, not S2S feature vectors

New approach:
- [ ] Use S2SFeatureExtractor to produce feature vectors per sample
      [pass1..pass7, score1..score7, overall_score] = 15 dims
- [ ] Train sklearn RandomForest + MLP on these features
- [ ] Compare vs raw IMU baseline
- [ ] Target: 85%+ accuracy
- [ ] Update model/training_report.json with new results
- [ ] Update README accuracy claim
```
**Can Claude help:** ✅ Yes — can write the full retrain script

---

### Issue #7 — Add honest accuracy caveat to README
**Label:** `documentation` `quick-win`
```
Currently README implies classifier is production-ready.
65.9% needs a clear caveat until Issue #6 is resolved.

Change line in README:
FROM: (implied production-ready)
TO:   "Domain classifier baseline: 65.9% (retraining in progress — see Issue #6)"

- [ ] Update README.md
- [ ] Commit and push
```
**Can Claude help:** ✅ Yes

---

## 🟢 MILESTONE 4 — ML Integration Layer (Week 3-5)
*Goal: Make S2S a plug-in for PyTorch/sklearn pipelines*

### Issue #8 — Build s2s_ml_interface module
**Label:** `feature` `ml` `priority:high`
```
Based on architecture designed in ML integration doc.

File: s2s_standard_v1_3/s2s_ml_interface.py

Classes to implement:
- [ ] S2SFeatureExtractor  — wraps PhysicsEngine, returns numpy array
- [ ] MotionDataset        — PyTorch Dataset using extractor
- [ ] S2SDataLoader        — convenience wrapper around DataLoader
- [ ] physics_loss()       — loss term: λ × (1 - physics_score/100)

Also:
- [ ] requirements-optional.txt listing: torch, numpy (not required, unlocks ML mode)
- [ ] Graceful fallback if torch not installed
- [ ] Example script: examples/train_with_s2s.py
```
**Can Claude help:** ✅ Yes — full implementation ready to write

---

### Issue #9 — Add NumPy fast-path to PhysicsEngine
**Label:** `performance` `ml`
```
Current: all math is pure Python (slow, no batching)
Goal: if numpy is installed, use vectorized operations

Steps:
- [ ] Detect numpy availability at import time
- [ ] Add _numpy_check() fast path in each law where possible
- [ ] Benchmark: pure Python vs numpy on 1000 samples
- [ ] Document speedup in README
```
**Can Claude help:** ✅ Yes

---

## 🔵 MILESTONE 5 — Visualization (Week 5-7)
*Goal: Make results shareable as screenshots/demos*

### Issue #10 — Build Streamlit certification dashboard
**Label:** `feature` `visualization`
```
A simple web UI showing:
- [ ] Upload CSV / paste IMU data
- [ ] Show certification tier (GOLD/SILVER/BRONZE/REJECTED) with color
- [ ] Bar chart: score per law (0-100)
- [ ] Timeline: which windows pass/fail
- [ ] Download certified JSON

File: dashboard/app.py
Run: streamlit run dashboard/app.py

Add to README: "## Live Demo" section with screenshot
```
**Can Claude help:** ✅ Yes — can write full Streamlit app

---

### Issue #11 — Add example output screenshots to README
**Label:** `documentation` `quick-win`
```
People share screenshots. The README has zero visuals beyond tables.

- [ ] Run certify() on UCI HAR sample
- [ ] Screenshot or render terminal output
- [ ] Add docs/screenshots/ folder
- [ ] Embed in README under "Real Results"
```
**Can Claude help:** ✅ Partial — can generate the code to produce output

---

## 🟣 MILESTONE 6 — Benchmark Experiment (Week 7-10)
*Goal: One reproducible result proving S2S improves ML*

### Issue #12 — Reproducible experiment: certified vs uncertified training
**Label:** `research` `ml` `priority:high`
```
This is the key academic/commercial proof-of-concept.

Experiment design:
- Dataset: UCI HAR (free, well-known)
- Model: simple MLP classifier (5 motion domains)
- Condition A: train on ALL data (uncertified)
- Condition B: train on GOLD+SILVER only (S2S certified)
- Condition C: train on all data + physics_loss term
- Metric: test accuracy, F1 per domain

Expected result: B or C > A (even 2-3% matters for a paper)

Deliverables:
- [ ] experiments/uci_har_benchmark.py  (reproducible script)
- [ ] experiments/results.json          (logged metrics)
- [ ] Section in README: "Benchmark Results"
```
**Can Claude help:** ✅ Yes — can write the full experiment script

---

### Issue #13 — PyPI package: pip install s2s-certify
**Label:** `distribution` `milestone:final`
```
Remove all friction from adoption.

Steps:
- [ ] Create pyproject.toml / setup.py
- [ ] Package name: s2s-certify
- [ ] Version: 1.3.0
- [ ] Test on TestPyPI first
- [ ] Publish to PyPI
- [ ] Update README install instructions
- [ ] Add PyPI badge to README

Note: BSL license allows research/education use — PyPI is fine.
```
**Can Claude help:** ✅ Yes — can write pyproject.toml and publish steps

---

## 📋 QUICK REFERENCE — Issue Priority Order

| # | Issue | Effort | Impact | Start |
|---|-------|--------|--------|-------|
| 1 | Push commit history | 5 min | 🔴 Critical | Now |
| 2 | Add missing files | 10 min | 🔴 Critical | Now |
| 3 | Add CONTEXT.md | 20 min | High | Now |
| 7 | Caveat classifier accuracy | 5 min | Medium | Now |
| 4 | Write pytest tests | 3-4 hrs | 🔴 Critical | Week 1 |
| 5 | GitHub Actions CI | 30 min | High | Week 1 |
| 6 | Retrain classifier | 2-3 hrs | High | Week 2 |
| 8 | ML interface module | 4-5 hrs | 🔴 Critical | Week 3 |
| 9 | NumPy fast-path | 2 hrs | Medium | Week 4 |
| 10 | Streamlit dashboard | 4-5 hrs | High | Week 5 |
| 11 | README screenshots | 1 hr | Medium | Week 5 |
| 12 | Benchmark experiment | 5-6 hrs | 🔴 Critical | Week 7 |
| 13 | PyPI package | 2 hrs | High | Week 10 |

---

## ✅ COMPLETED
*Move issues here when closed*

- [ ] *(none yet)*

---
*Roadmap generated: 2026-03-03. Update as milestones complete.*
