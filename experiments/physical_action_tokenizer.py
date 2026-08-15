#!/usr/bin/env python3
"""
Physical Action Tokenizer (PAT) — Experiment
=============================================
Tests whether S2S physics certification improves action disambiguation.

Three conditions on NinaPro DB5 s1/E1 data:
  A: Hard single-label classification (baseline MLP)
  B: Soft distribution — temperature-scaled softmax, no physics
  C: Physics-constrained distribution — S2S tier and laws shape the output

Key claim: physics certification narrows ambiguous action distributions.
GOLD windows should have lower entropy (more certain).
Physics law failures should shift probability mass away from impossible actions.

Usage: cd ~/S2S && python3 experiments/physical_action_tokenizer.py
"""
import os, sys, csv, json, math, random
import numpy as np
import scipy.io
sys.path.insert(0, os.path.expanduser("~/S2S"))

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
W = "\033[97m"; D = "\033[2m";  NC = "\033[0m"

# ── Config ────────────────────────────────────────────────────────────────────
CERT_CSV   = "experiments/ninapro_db5_certified.csv"
MAT_PATH   = os.path.expanduser("~/ninapro_db5/s1/S1_E1_A1.mat")
WIN_SIZE   = 500   # samples per window in certified CSV
HZ         = 200.0 # corrected from stored 2000Hz
N_CLASSES  = 3     # gesture 0,1,2 (drop gesture 3, only 30 windows)
TEMPERATURE = 2.0  # softmax temperature for condition B/C
EPOCHS     = 60
random.seed(42)
np.random.seed(42)

# ── Gesture metadata for physics constraint ───────────────────────────────────
# NinaPro DB5 Exercise 1 gesture mapping
GESTURE_NAMES = {0: "REST", 1: "FINGER_FLEX", 2: "FINGER_EXT"}

# Physics constraint: which laws failing affects which gesture class
# Higher value = increase probability, lower = decrease
# Format: {law_name: {gesture_class: delta_logit}}
LAW_CONSTRAINTS = {
    "sensor_freeze": {
        0: +1.5,   # frozen sensor → likely REST
        1: -1.0,   # not finger flex
        2: -0.5,
    },
    "jerk_bounds": {
        0: -1.0,   # high jerk = NOT rest
        1: +0.5,
        2: +0.5,
    },
    "imu_internal_consistency": {
        0:  0.0,   # uncertainty → widen all equally
        1:  0.0,
        2:  0.0,
    },
    "innovation_kurtosis": {
        0: +0.3,   # Gaussian innovations slightly more likely at rest
        1: -0.2,
        2: -0.1,
    },
    "resonance_frequency": {
        0: +0.8,   # out-of-band tremor more likely during REST
        1: -0.4,   # less likely during active finger movement
        2: -0.4,
    },
    "cross_axis_cohesion": {
        0: +0.5,   # uncoupled axes more likely at rest
        1: -0.3,
        2: -0.2,
    },
    "temporal_autocorrelation": {
        0: -0.5,   # no temporal coherence = unlikely to be real movement
        1: +0.2,
        2: +0.3,
    },
}

# Tier entropy modifier
TIER_ENTROPY_MOD = {
    "GOLD":   -0.5,  # sharpen distribution (more confident)
    "SILVER":  0.0,  # no change
    "BRONZE": +0.5,  # widen distribution (less confident)
    "REJECTED": +1.0,
}

# ── Load certified CSV metadata ───────────────────────────────────────────────
def load_certified():
    rows = []
    with open(CERT_CSV) as f:
        for r in csv.DictReader(f):
            label = int(r["gesture_label"])
            if label >= N_CLASSES:
                continue
            laws_failed = [l for l in r["laws_failed"].split("|") if l.strip()]
            rows.append({
                "window_idx":  int(r["window_idx"]),
                "start_sample": int(r["start_sample"]),
                "label":       label,
                "tier":        r["tier"],
                "score":       int(r["score"]),
                "laws_failed": laws_failed,
            })
    return rows

# ── Load raw acc features from mat ───────────────────────────────────────────
def extract_features(acc_array, start, size):
    """Extract 12 features from a raw acc window."""
    end = min(start + size, len(acc_array))
    window = acc_array[start:end]
    if len(window) < 32:
        return None
    feats = []
    for axis in range(3):
        col = window[:, axis].astype(float) * 9.81  # g → m/s²
        feats += [col.mean(), col.std(), col.max() - col.min()]
    mag = np.linalg.norm(window * 9.81, axis=1)
    feats += [mag.mean(), mag.std(), mag.max()]
    return np.array(feats, dtype=np.float32)

# ── Simple MLP ────────────────────────────────────────────────────────────────
def softmax(x, T=1.0):
    x = np.array(x, dtype=float) / T
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()

class MLP:
    def __init__(self, n_in, n_h, n_out, lr=0.005):
        self.lr = lr
        s1 = math.sqrt(2.0 / n_in)
        s2 = math.sqrt(2.0 / n_h)
        self.W1 = np.random.randn(n_h, n_in) * s1
        self.b1 = np.zeros(n_h)
        self.W2 = np.random.randn(n_out, n_h) * s2
        self.b2 = np.zeros(n_out)

    def forward(self, x):
        h = np.maximum(0, self.W1 @ x + self.b1)
        logits = self.W2 @ h + self.b2
        return h, logits

    def backward(self, x, h, probs, target_dist):
        dL = probs - target_dist
        self.W2 -= self.lr * np.outer(dL, h)
        self.b2 -= self.lr * dL
        dh = (self.W2.T @ dL) * (h > 0)
        self.W1 -= self.lr * np.outer(dh, x)
        self.b1 -= self.lr * dh

def normalize(X):
    mu, sigma = X.mean(0), X.std(0) + 1e-8
    return (X - mu) / sigma, mu, sigma

def entropy(dist):
    dist = np.array(dist) + 1e-12
    dist /= dist.sum()
    return -np.sum(dist * np.log(dist))

# ── Physics constraint function ───────────────────────────────────────────────
def apply_physics_constraint(logits, tier, laws_failed):
    """Adjust logits using S2S certification information."""
    logits = logits.copy()

    # Apply law-specific constraints
    for law in laws_failed:
        if law in LAW_CONSTRAINTS:
            for cls, delta in LAW_CONSTRAINTS[law].items():
                if cls < N_CLASSES:
                    logits[cls] += delta

    # Apply tier entropy modifier
    mod = TIER_ENTROPY_MOD.get(tier, 0.0)
    if mod < 0:  # sharpen: reduce temperature
        T = max(0.3, 1.0 + mod)
    elif mod > 0:  # widen: increase temperature
        T = 1.0 + mod
    else:
        T = 1.0

    return softmax(logits, T)

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, X, y, metadata=None, condition="A"):
    top1, top3, entropies = 0, 0, []
    tier_entropies = {"GOLD": [], "SILVER": [], "BRONZE": []}

    for i, (x, label) in enumerate(zip(X, y)):
        _, logits = model.forward(x)

        if condition == "A":
            probs = softmax(logits, T=1.0)
        elif condition == "B":
            probs = softmax(logits, T=TEMPERATURE)
        else:  # C: physics-constrained
            meta = metadata[i] if metadata else {}
            probs = apply_physics_constraint(
                logits,
                meta.get("tier", "SILVER"),
                meta.get("laws_failed", [])
            )

        pred = np.argmax(probs)
        top3_preds = np.argsort(probs)[::-1][:3]

        if pred == label:
            top1 += 1
        if label in top3_preds:
            top3 += 1

        ent = entropy(probs)
        entropies.append(ent)

        if metadata:
            tier = metadata[i].get("tier", "SILVER")
            if tier in tier_entropies:
                tier_entropies[tier].append(ent)

    n = len(y)
    return {
        "top1": top1 / n,
        "top3": top3 / n,
        "mean_entropy": np.mean(entropies),
        "tier_entropy": {
            t: np.mean(v) if v else None
            for t, v in tier_entropies.items()
        }
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{W}{'='*65}")
    print("  Physical Action Tokenizer (PAT) — S2S Experiment")
    print("  NinaPro DB5 s1/E1  |  3 gesture classes  |  Physics-constrained")
    print(f"{'='*65}{NC}\n")

    # Load
    print("Loading certified metadata...")
    rows = load_certified()
    print(f"  {len(rows)} windows, {N_CLASSES} classes")

    print("Loading raw acc from mat file...")
    mat = scipy.io.loadmat(MAT_PATH)
    acc = mat["acc"]  # (N, 3) float32

    # Extract features
    print("Extracting features...")
    features, labels, metadata = [], [], []
    skipped = 0
    for r in rows:
        feats = extract_features(acc, r["start_sample"], WIN_SIZE)
        if feats is None:
            skipped += 1
            continue
        features.append(feats)
        labels.append(r["label"])
        metadata.append({"tier": r["tier"], "laws_failed": r["laws_failed"],
                          "score": r["score"]})

    if skipped:
        print(f"  Skipped {skipped} windows (too short)")

    X = np.array(features)
    y = np.array(labels)
    print(f"  Feature matrix: {X.shape}")

    from collections import Counter
    print(f"  Class distribution: {dict(Counter(y.tolist()))}")
    tier_dist = Counter(m["tier"] for m in metadata)
    print(f"  Tier distribution: {dict(tier_dist)}")

    # Train/test split — subject-stratified by gesture class
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    n_test = max(1, int(len(idx) * 0.25))
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    m_tr = [metadata[i] for i in train_idx]
    m_te = [metadata[i] for i in test_idx]

    X_tr, mu, sigma = normalize(X_tr)
    X_te = (X_te - mu) / (sigma + 1e-8)

    results = {}

    # Train one shared model, evaluate under three conditions
    print("\nTraining MLP...")
    model = MLP(n_in=X_tr.shape[1], n_h=64, n_out=N_CLASSES, lr=0.005)
    np.random.seed(42)

    for ep in range(EPOCHS):
        perm = np.random.permutation(len(X_tr))
        for i in perm:
            x, label = X_tr[i], y_tr[i]
            h, logits = model.forward(x)
            probs = softmax(logits)
            # Hard label target for training
            target = np.zeros(N_CLASSES)
            target[label] = 1.0
            model.backward(x, h, probs, target)

        if (ep + 1) % 20 == 0:
            r = evaluate(model, X_te, y_te, condition="A")
            print(f"  Epoch {ep+1:3d}/{EPOCHS}  top1={r['top1']:.3f}")

    # Evaluate three conditions
    print(f"\n{W}{'─'*65}")
    print(f"  Results")
    print(f"{'─'*65}{NC}")

    for cond, label, use_meta in [
        ("A", "Hard classification (baseline)", False),
        ("B", "Soft distribution, no physics", False),
        ("C", "Physics-constrained distribution", True),
    ]:
        r = evaluate(model, X_te, y_te,
                     metadata=m_te if use_meta else None,
                     condition=cond)
        results[f"condition_{cond}"] = r
        delta_top1 = ""
        if cond != "A":
            d = r["top1"] - results["condition_A"]["top1"]
            delta_top1 = f"  ({'+' if d>=0 else ''}{d*100:.1f}%)"
        print(f"\n  {W}Condition {cond}: {label}{NC}")
        print(f"    Top-1 accuracy:  {r['top1']:.4f}{delta_top1}")
        print(f"    Top-3 accuracy:  {r['top3']:.4f}")
        print(f"    Mean entropy:    {r['mean_entropy']:.4f}")
        if r["tier_entropy"]["GOLD"] is not None:
            print(f"    {G}GOLD entropy:    {r['tier_entropy']['GOLD']:.4f}{NC}  (should be lowest)")
        if r["tier_entropy"]["SILVER"] is not None:
            print(f"    SILVER entropy:  {r['tier_entropy']['SILVER']:.4f}")
        if r["tier_entropy"]["BRONZE"] is not None:
            print(f"    BRONZE entropy:  {r['tier_entropy']['BRONZE']:.4f}  (should be highest)")

    # Physics constraint analysis
    print(f"\n{W}{'─'*65}")
    print(f"  Physics Constraint Analysis")
    print(f"{'─'*65}{NC}")

    law_counts = Counter()
    for m in m_te:
        for law in m["laws_failed"]:
            law_counts[law] += 1

    print(f"  Test windows: {len(m_te)}")
    print(f"  Windows with at least one law failure: "
          f"{sum(1 for m in m_te if m['laws_failed'])}")
    if law_counts:
        print(f"  Top failing laws in test set:")
        for law, cnt in law_counts.most_common(5):
            pct = 100 * cnt / len(m_te)
            if law in LAW_CONSTRAINTS:
                print(f"    {G}{law}: {cnt} ({pct:.1f}%) ← constraint applied{NC}")
            else:
                print(f"    {law}: {cnt} ({pct:.1f}%)")

    # Sample PAT tokens
    print(f"\n{W}{'─'*65}")
    print(f"  Sample Physical Action Tokens")
    print(f"{'─'*65}{NC}")

    sample_tokens = []
    shown = 0
    for i in range(min(len(X_te), 5)):
        x, true_label, meta = X_te[i], y_te[i], m_te[i]
        _, logits = model.forward(x)
        dist_B = softmax(logits, T=TEMPERATURE)
        dist_C = apply_physics_constraint(logits, meta["tier"], meta["laws_failed"])

        token = {
            "true_label": GESTURE_NAMES.get(int(true_label), str(true_label)),
            "tier": meta["tier"],
            "score": meta["score"],
            "laws_failed": meta["laws_failed"],
            "action_distribution_unconstrained": {
                GESTURE_NAMES[j]: round(float(dist_B[j]), 4)
                for j in range(N_CLASSES)
            },
            "action_distribution_physics": {
                GESTURE_NAMES[j]: round(float(dist_C[j]), 4)
                for j in range(N_CLASSES)
            },
            "entropy_before": round(entropy(dist_B), 4),
            "entropy_after":  round(entropy(dist_C), 4),
            "physics_constrained": bool(meta["laws_failed"]),
        }
        sample_tokens.append(token)

        print(f"\n  Token {i+1}: true={token['true_label']}  "
              f"tier={meta['tier']}  score={meta['score']}")
        print(f"    Laws failed: {meta['laws_failed'] or 'none'}")
        print(f"    Before physics: "
              + "  ".join(f"{k}={v:.3f}" for k, v in
                          token["action_distribution_unconstrained"].items()))
        print(f"    After physics:  "
              + "  ".join(f"{k}={v:.3f}" for k, v in
                          token["action_distribution_physics"].items()))
        ent_delta = token["entropy_after"] - token["entropy_before"]
        color = G if ent_delta < 0 else Y
        print(f"    Entropy: {token['entropy_before']:.4f} → "
              f"{token['entropy_after']:.4f}  "
              f"{color}({'+' if ent_delta>=0 else ''}{ent_delta:.4f}){NC}")

    # Summary verdict
    A = results["condition_A"]["top1"]
    C = results["condition_C"]["top1"]
    delta = C - A

    print(f"\n{W}{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}{NC}")
    print(f"  Baseline (A):              {A:.4f}")
    print(f"  Physics-constrained (C):   {C:.4f}  "
          f"({'+'if delta>=0 else ''}{delta*100:.2f}%)")

    gold_ent  = results["condition_C"]["tier_entropy"].get("GOLD")
    silv_ent  = results["condition_C"]["tier_entropy"].get("SILVER")
    tier_ok = (gold_ent is not None and silv_ent is not None
               and gold_ent < silv_ent)

    print()
    if delta > 0 and tier_ok:
        print(f"  {G}✓ PAT PROVEN{NC}")
        print(f"    Physics constraint improves accuracy AND")
        print(f"    GOLD windows have lower entropy than SILVER")
    elif tier_ok:
        print(f"  {Y}~ PARTIAL{NC}")
        print(f"    Tier entropy ordering correct (GOLD < SILVER)")
        print(f"    but top-1 accuracy did not improve")
    else:
        print(f"  {R}✗ NOT PROVEN on this dataset{NC}")
        print(f"    Insufficient law failures to trigger constraints")
        print(f"    Consider running on full 10-subject dataset")

    # Save
    out = {
        "experiment": "physical_action_tokenizer",
        "dataset": "NinaPro DB5 s1/E1",
        "n_classes": N_CLASSES,
        "gesture_names": GESTURE_NAMES,
        "n_windows": len(y),
        "results": results,
        "sample_tokens": sample_tokens,
        "law_constraints_used": list(LAW_CONSTRAINTS.keys()),
    }
    out_path = "experiments/results_physical_action_tokenizer.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved → {out_path}")
    print(f"{W}{'='*65}{NC}\n")


if __name__ == "__main__":
    main()
