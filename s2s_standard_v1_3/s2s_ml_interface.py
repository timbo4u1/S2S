"""
S2S ML Interface — v1.0
Bridges the S2S physics engine with PyTorch and sklearn ML pipelines.

Install optional dependencies:
    pip install torch numpy scikit-learn

Usage:
    from s2s_standard_v1_3.s2s_ml_interface import S2SFeatureExtractor, physics_loss

    extractor = S2SFeatureExtractor()
    features  = extractor(imu_raw)          # numpy array, shape (15,)
    loss      = physics_loss(scores, 0.1)   # physics regularization term
"""

import math
import sys

# ── Optional dependency detection ─────────────────────────────────────────────

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH = True
except ImportError:
    _TORCH = False

from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine


# ── FEATURE EXTRACTOR ─────────────────────────────────────────────────────────

class S2SFeatureExtractor:
    """
    Converts raw IMU data into a fixed-length ML feature vector using
    the S2S physics engine. No physics code is re-implemented here.

    Feature vector (15 dims):
        [pass_1..pass_7]       — 7 binary floats (0.0 or 1.0) per law
        [score_1..score_7]     — 7 float scores per law (0.0–1.0)
        [overall_score]        — 1 float (0.0–1.0)

    Example:
        extractor = S2SFeatureExtractor()
        features  = extractor(imu_raw={"timestamps_ns": [...], "accel": [...], "gyro": [...]})
        # features.shape == (15,)  if numpy installed
        # features is a list of 15 floats otherwise
    """

    # S2S returns these law names in order
    LAW_NAMES = [
        "Newton F=ma",
        "rigid_body_kinematics",
        "resonance_frequency",
        "jerk_bounds",
        "imu_consistency",
        "BCG heartbeat",
        "Joule heating",
    ]

    def __init__(self, physics_engine=None, segment="forearm", normalize_scores=True):
        """
        Args:
            physics_engine:   PhysicsEngine instance (created fresh if None)
            segment:          Body segment string passed to certify()
            normalize_scores: If True, divide raw scores by 100 → range [0,1]
        """
        self.engine = physics_engine or PhysicsEngine()
        self.segment = segment
        self.normalize = normalize_scores

    def __call__(self, imu_raw):
        """
        Run physics certification and return feature vector.

        Args:
            imu_raw: dict with keys timestamps_ns, accel, gyro

        Returns:
            numpy array of shape (15,) if numpy available, else list of 15 floats
        """
        result = self.engine.certify(imu_raw=imu_raw, segment=self.segment)
        return self._build_feature_vector(result)

    def _build_feature_vector(self, result):
        """Turn a certify() result dict into a flat numeric vector."""
        laws_passed = result.get('laws_passed', [])

        # 7 binary pass/fail flags
        pass_flags = []
        for name in self.LAW_NAMES:
            passed = name in laws_passed
            pass_flags.append(1.0 if passed else 0.0)

        # 7 per-law scores — try to read from result, fall back to pass_flag * overall
        overall_raw = float(result.get('physical_law_score', 0))
        scores = []
        for i, name in enumerate(self.LAW_NAMES):
            # Try dedicated per-law score key
            score = result.get(name, {})
            if isinstance(score, dict):
                val = float(score.get('score', pass_flags[i] * overall_raw))
            elif isinstance(score, (int, float)):
                val = float(score)
            else:
                val = pass_flags[i] * overall_raw
            scores.append(val)

        # Overall score
        overall = overall_raw / 100.0 if self.normalize else overall_raw

        # Normalize per-law scores
        if self.normalize:
            scores = [s / 100.0 for s in scores]

        feature_list = pass_flags + scores + [overall]  # 7 + 7 + 1 = 15

        if _NUMPY:
            return np.array(feature_list, dtype=np.float32)
        return feature_list

    @property
    def feature_dim(self):
        """Dimension of the output feature vector (always 15)."""
        return 15

    @property
    def feature_names(self):
        """Human-readable names for each feature dimension."""
        names = [f"pass_{n.replace(' ', '_')}" for n in self.LAW_NAMES]
        names += [f"score_{n.replace(' ', '_')}" for n in self.LAW_NAMES]
        names += ["overall_score"]
        return names


# ── PYTORCH DATASET ───────────────────────────────────────────────────────────

if _TORCH:
    class MotionDataset(Dataset):
        """
        PyTorch Dataset that applies S2S feature extraction to each sample.

        Example:
            dataset = MotionDataset(imu_list, labels)
            loader  = DataLoader(dataset, batch_size=32, shuffle=True)

            for features, labels in loader:
                outputs = model(features)   # features shape: [batch, 15]
        """

        def __init__(self, imu_data_list, labels, extractor=None,
                     segment="forearm", precompute=False):
            """
            Args:
                imu_data_list: list of dicts, each with timestamps_ns/accel/gyro
                labels:        list of integer class labels
                extractor:     S2SFeatureExtractor (created fresh if None)
                segment:       body segment for certification
                precompute:    if True, extract all features upfront (faster training,
                               higher memory). If False, extract on-the-fly.
            """
            assert len(imu_data_list) == len(labels), \
                "imu_data_list and labels must have same length"

            self.imu_data_list = imu_data_list
            self.labels = labels
            self.extractor = extractor or S2SFeatureExtractor(segment=segment)

            self._features = None
            if precompute:
                print(f"Precomputing S2S features for {len(imu_data_list)} samples...")
                self._features = [self.extractor(d) for d in imu_data_list]
                print("Done.")

        def __len__(self):
            return len(self.imu_data_list)

        def __getitem__(self, idx):
            if self._features is not None:
                features = self._features[idx]
            else:
                features = self.extractor(self.imu_data_list[idx])

            label = self.labels[idx]

            if _NUMPY and isinstance(features, np.ndarray):
                feat_tensor = torch.from_numpy(features)
            else:
                feat_tensor = torch.tensor(features, dtype=torch.float32)

            return feat_tensor, torch.tensor(label, dtype=torch.long)

    class S2SDataLoader(DataLoader):
        """
        Convenience wrapper: DataLoader pre-configured for S2S MotionDataset.

        Example:
            loader = S2SDataLoader(imu_list, labels, batch_size=32)
            for features, labels in loader:
                ...
        """
        def __init__(self, imu_data_list, labels, batch_size=32,
                     shuffle=True, num_workers=0, **kwargs):
            dataset = MotionDataset(imu_data_list, labels)
            super().__init__(dataset, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers, **kwargs)

else:
    # Stub classes so imports don't fail if torch is not installed
    class MotionDataset:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for MotionDataset. "
                "Install with: pip install torch"
            )

    class S2SDataLoader:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required for S2SDataLoader. "
                "Install with: pip install torch"
            )


# ── PHYSICS-INFORMED LOSS ─────────────────────────────────────────────────────

def physics_loss(physics_scores, lambda_phys=0.1):
    """
    Physics-informed regularization loss term.

    Penalizes batches where physics scores are low.
    Add to your task loss:
        total_loss = task_loss + physics_loss(scores, lambda_phys=0.1)

    Formula: λ × mean(1 - physics_score/100)

    Args:
        physics_scores: list of floats OR torch.Tensor, values in [0, 100]
        lambda_phys:    weighting coefficient (default 0.1)

    Returns:
        torch.Tensor scalar if torch available, else float
    """
    if _TORCH and isinstance(physics_scores, torch.Tensor):
        penalty = torch.mean(1.0 - physics_scores / 100.0)
        return lambda_phys * penalty

    if _NUMPY and isinstance(physics_scores, np.ndarray):
        penalty = float(np.mean(1.0 - physics_scores / 100.0))
        return lambda_phys * penalty

    # Pure Python fallback
    n = len(physics_scores)
    if n == 0:
        return 0.0
    penalty = sum(1.0 - s / 100.0 for s in physics_scores) / n
    return lambda_phys * penalty


def certify_batch(imu_batch, segment="forearm", engine=None):
    """
    Run PhysicsEngine.certify() on a list of IMU samples.
    Returns (feature_matrix, scores, tiers).

    Args:
        imu_batch: list of imu_raw dicts
        segment:   body segment
        engine:    PhysicsEngine instance (created fresh if None)

    Returns:
        features: numpy array shape (n, 15) or list of lists
        scores:   list of overall_score floats
        tiers:    list of tier strings
    """
    eng = engine or PhysicsEngine()
    extractor = S2SFeatureExtractor(physics_engine=eng, segment=segment)

    features_list = []
    scores = []
    tiers = []

    for imu_raw in imu_batch:
        result = eng.certify(imu_raw=imu_raw, segment=segment)
        features_list.append(extractor._build_feature_vector(result))
        scores.append(float(result.get('physical_law_score', 0)))
        tiers.append(result.get('tier', 'REJECTED'))

    if _NUMPY:
        return np.array(features_list, dtype=np.float32), scores, tiers
    return features_list, scores, tiers


# ── EXAMPLE TRAINING LOOP ─────────────────────────────────────────────────────

def example_training_loop():
    """
    Minimal example of using S2S features with PyTorch.
    Run: python3 -c "from s2s_standard_v1_3.s2s_ml_interface import example_training_loop; example_training_loop()"
    """
    if not _TORCH:
        print("PyTorch not installed. Install with: pip install torch")
        return

    import torch.nn as nn

    print("S2S ML Interface — Example Training Loop")
    print("=" * 45)

    # --- 1. Build toy dataset ---
    import math
    def fake_imu(label, n=100, hz=100):
        dt = 1.0 / hz
        scale = 1.0 + label * 0.5
        return {
            "timestamps_ns": [int(i * dt * 1e9) for i in range(n)],
            "accel": [[scale * math.sin(2 * math.pi * 2 * i * dt),
                       scale * math.cos(2 * math.pi * 1.5 * i * dt),
                       9.81] for i in range(n)],
            "gyro":  [[0.1 * scale * math.sin(2 * math.pi * 2 * i * dt),
                       0.1 * scale * math.cos(2 * math.pi * 1.5 * i * dt),
                       0.0] for i in range(n)],
        }

    n_samples = 50
    imu_list = [fake_imu(i % 3) for i in range(n_samples)]
    labels   = [i % 3 for i in range(n_samples)]

    # --- 2. Dataset + DataLoader ---
    dataset = MotionDataset(imu_list, labels, precompute=True)
    loader  = DataLoader(dataset, batch_size=10, shuffle=True)

    # --- 3. Simple MLP classifier ---
    feature_dim  = S2SFeatureExtractor().feature_dim  # 15
    num_classes  = 3
    model = nn.Sequential(
        nn.Linear(feature_dim, 32),
        nn.ReLU(),
        nn.Linear(32, num_classes),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # --- 4. Training loop with physics loss ---
    for epoch in range(5):
        total_loss = 0.0
        for feat_batch, label_batch in loader:
            outputs   = model(feat_batch)
            task_loss = criterion(outputs, label_batch)

            # Physics regularization
            scores = feat_batch[:, -1] * 100.0  # last feature = overall_score
            phys   = physics_loss(scores, lambda_phys=0.1)

            loss = task_loss + phys
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1}/5  loss={total_loss/len(loader):.4f}")

    print("\nDone. S2S ML interface working correctly.")
    print(f"Feature names: {S2SFeatureExtractor().feature_names}")


if __name__ == "__main__":
    example_training_loop()
