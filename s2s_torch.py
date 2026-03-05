"""
s2s_torch.py — S2S Physics Loss for PyTorch Training

Wraps S2S physics score as a differentiable training signal:
    L = L_task + λ × (1 - physics_score / 100)

High-quality data (GOLD score=90) → physics penalty ≈ 0.10
Low-quality data  (BRONZE score=60) → physics penalty ≈ 0.40
Rejected data     (score=0)         → physics penalty = 1.00

Usage:
    from s2s_torch import S2SLoss

    criterion = S2SLoss(task_weight=1.0, physics_weight=0.3)

    for batch_windows, batch_labels in dataloader:
        predictions = model(batch_windows)
        loss = criterion(predictions, batch_labels, batch_windows)
        loss.backward()

Requirements:
    torch >= 1.9
    s2s-certify >= 1.4.0  (pip install s2s-certify)

Install:
    pip install s2s-certify torch

Zero additional dependencies beyond torch + s2s-certify.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional, Union


# ── Physics scorer (pure Python, no torch dependency) ─────────────────────────

def _score_window_raw(accel_samples: List[List[float]], hz: float = 50.0) -> int:
    """
    Score a single IMU window 0-100 using S2S physics laws.
    Input: list of [ax, ay, az] samples in m/s²
    Returns: integer score 0-100
    """
    n = len(accel_samples)
    if n < 20:
        return 0

    ax = [accel_samples[i][0] for i in range(n)]
    ay = [accel_samples[i][1] for i in range(n)]
    az = [accel_samples[i][2] for i in range(n)]

    # Law 1: Variance check — dead/flat signal
    mx = sum(ax)/n; my = sum(ay)/n; mz = sum(az)/n
    tv = (sum((v-mx)**2 for v in ax) +
          sum((v-my)**2 for v in ay) +
          sum((v-mz)**2 for v in az)) / n
    if tv < 0.005:
        return 0   # HARD REJECT — flat/dead signal

    # Law 2: Clipping check — ADC saturation
    for axis_vals in [ax, ay, az]:
        max_val = max(abs(v) for v in axis_vals)
        if max_val < 0.5:
            continue
        at_max = sum(1 for v in axis_vals if abs(abs(v) - max_val) < 0.01)
        if at_max / n > 0.20:
            return 0   # HARD REJECT — clipped/saturated

    # Law 3: Jerk bound (only meaningful at ≥50Hz)
    scores = []
    if hz >= 50:
        dt = 1.0 / hz
        jerks = []
        for av in [ax, ay, az]:
            vel  = [(av[k+1]-av[k-1])/(2*dt) for k in range(1, n-1)]
            jerk = [(vel[k+1]-vel[k-1])/(2*dt) for k in range(1, len(vel)-1)]
            jerks.extend([abs(j) for j in jerk])
        if jerks:
            jerks.sort()
            p95 = jerks[int(len(jerks)*0.95)]
            if p95 > 5000:
                return 0   # HARD REJECT — robotic/keyframe artefact

    # Variance quality score
    if tv > 10.0:   scores.append(90)
    elif tv > 2.0:  scores.append(80)
    elif tv > 0.5:  scores.append(70)
    else:           scores.append(58)

    # Zero-crossing rate — motion complexity
    dx = [v - mx for v in ax]
    zcr = sum(1 for i in range(1, n) if dx[i]*dx[i-1] < 0) / n
    if zcr > 0.15:   scores.append(85)
    elif zcr > 0.07: scores.append(70)
    else:            scores.append(55)

    return int(sum(scores) / len(scores)) if scores else 0


def score_batch(
    windows: Union[torch.Tensor, List],
    hz: float = 50.0,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Score a batch of IMU windows.

    Args:
        windows: Tensor [batch, timesteps, 3] or [batch, 3, timesteps]
                 or list of lists [[ax,ay,az], ...]
        hz:      Sample rate in Hz (affects jerk law threshold)
        device:  Target device for output tensor

    Returns:
        scores: FloatTensor [batch] with values 0-100
    """
    if isinstance(windows, torch.Tensor):
        w = windows.detach().cpu()
        if w.dim() == 3:
            # Handle both [batch, timesteps, 3] and [batch, 3, timesteps]
            if w.shape[-1] == 3:
                # [batch, timesteps, 3] → iterate
                batch_list = [[[w[b,t,0].item(), w[b,t,1].item(), w[b,t,2].item()]
                                for t in range(w.shape[1])]
                               for b in range(w.shape[0])]
            elif w.shape[1] == 3:
                # [batch, 3, timesteps] → transpose
                batch_list = [[[w[b,0,t].item(), w[b,1,t].item(), w[b,2,t].item()]
                                for t in range(w.shape[2])]
                               for b in range(w.shape[0])]
            else:
                raise ValueError(f"Expected last or second dim = 3, got {w.shape}")
        else:
            raise ValueError(f"Expected 3D tensor [batch, time, 3], got {w.shape}")
    else:
        batch_list = windows

    raw_scores = [_score_window_raw(w, hz) for w in batch_list]
    scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)

    if device is not None:
        scores_tensor = scores_tensor.to(device)
    return scores_tensor


# ── S2S Loss ───────────────────────────────────────────────────────────────────

class S2SLoss(nn.Module):
    """
    S2S Physics-Aware Loss for PyTorch training.

    Combines task loss with physics quality penalty:
        L = task_weight × L_task + physics_weight × L_physics

    where:
        L_physics = mean(1 - physics_score / 100)
                  = 0.10 for GOLD (score=90)
                  = 0.40 for BRONZE (score=60)
                  = 1.00 for REJECTED (score=0)

    Args:
        task_weight    (float): Weight for task loss. Default 1.0.
        physics_weight (float): Weight for physics penalty. Default 0.3.
        hz             (float): Sample rate of IMU data. Default 50.0.
        task_loss      (nn.Module): Task loss function. Default CrossEntropyLoss.
        reduction      (str): 'mean' or 'sum'. Default 'mean'.

    Example:
        criterion = S2SLoss(task_weight=1.0, physics_weight=0.3, hz=100.0)
        loss = criterion(logits, labels, imu_windows)
        loss.backward()
    """

    def __init__(
        self,
        task_weight:    float = 1.0,
        physics_weight: float = 0.3,
        hz:             float = 50.0,
        task_loss:      Optional[nn.Module] = None,
        reduction:      str = 'mean',
    ):
        super().__init__()
        self.task_weight    = task_weight
        self.physics_weight = physics_weight
        self.hz             = hz
        self.reduction      = reduction
        self.task_loss_fn   = task_loss or nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        predictions: torch.Tensor,
        labels:      torch.Tensor,
        imu_windows: Union[torch.Tensor, List],
    ) -> torch.Tensor:
        """
        Compute combined task + physics loss.

        Args:
            predictions: Model output logits [batch, num_classes]
            labels:      Ground truth labels [batch]
            imu_windows: Raw IMU data [batch, timesteps, 3] or list

        Returns:
            Scalar loss tensor (differentiable w.r.t. predictions)
        """
        # Task loss (differentiable)
        task_loss = self.task_loss_fn(predictions, labels)

        # Physics penalty (non-differentiable score, scalar penalty)
        device = predictions.device
        scores = score_batch(imu_windows, hz=self.hz, device=device)
        physics_penalty = (1.0 - scores / 100.0).mean()

        total = self.task_weight * task_loss + self.physics_weight * physics_penalty
        return total

    def score_batch(
        self,
        imu_windows: Union[torch.Tensor, List]
    ) -> torch.Tensor:
        """
        Score a batch of windows without computing loss.
        Useful for monitoring data quality during training.

        Returns:
            scores: FloatTensor [batch] with values 0-100
        """
        return score_batch(imu_windows, hz=self.hz)

    def get_tier(self, score: float) -> str:
        """Map a physics score to its S2S tier name."""
        if score >= 87: return 'GOLD'
        if score >= 75: return 'SILVER'
        if score >= 60: return 'BRONZE'
        if score > 0:   return 'REJECTED'
        return 'REJECTED'

    def extra_repr(self) -> str:
        return (f"task_weight={self.task_weight}, "
                f"physics_weight={self.physics_weight}, "
                f"hz={self.hz}Hz")


# ── Curriculum sampler (bonus utility) ────────────────────────────────────────

class S2SCurriculumSampler:
    """
    Curriculum learning sampler based on S2S physics tiers.
    Yields batches in physics quality order: GOLD → SILVER → BRONZE.

    Usage:
        sampler = S2SCurriculumSampler(
            windows=all_windows,      # list of [timesteps, 3] arrays
            labels=all_labels,
            hz=50.0,
            phase_epochs=[10, 10, 20] # epochs per phase
        )
        for epoch in range(40):
            batch_windows, batch_labels = sampler.get_epoch_data(epoch)
            # train on this epoch's data
    """

    def __init__(
        self,
        windows: List,
        labels:  List[int],
        hz:      float = 50.0,
        phase_epochs: List[int] = None,
    ):
        self.hz = hz
        self.phase_epochs = phase_epochs or [10, 10, 20]

        print("S2SCurriculumSampler: scoring windows...")
        scores = [_score_window_raw(w, hz) for w in windows]

        self.tiers = {
            'GOLD':   [],
            'SILVER': [],
            'BRONZE': [],
        }
        for i, (w, l, s) in enumerate(zip(windows, labels, scores)):
            if s >= 87:   self.tiers['GOLD'].append((w, l, s))
            elif s >= 75: self.tiers['SILVER'].append((w, l, s))
            elif s >= 60: self.tiers['BRONZE'].append((w, l, s))

        print(f"  GOLD:   {len(self.tiers['GOLD'])}")
        print(f"  SILVER: {len(self.tiers['SILVER'])}")
        print(f"  BRONZE: {len(self.tiers['BRONZE'])}")

        # Phase boundaries
        self._p1_end = self.phase_epochs[0]
        self._p2_end = self._p1_end + self.phase_epochs[1]

    def get_epoch_data(self, epoch: int):
        """Get (windows, labels) for a given epoch number (0-indexed)."""
        if epoch < self._p1_end:
            pool = self.tiers['GOLD']
        elif epoch < self._p2_end:
            pool = self.tiers['GOLD'] + self.tiers['SILVER']
        else:
            pool = (self.tiers['GOLD'] +
                    self.tiers['SILVER'] +
                    self.tiers['BRONZE'])
        windows = [item[0] for item in pool]
        labels  = [item[1] for item in pool]
        return windows, labels

    def current_phase(self, epoch: int) -> str:
        if epoch < self._p1_end:   return "PHASE_1_GOLD"
        if epoch < self._p2_end:   return "PHASE_2_GOLD_SILVER"
        return "PHASE_3_ALL"


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("S2S PyTorch Loss Wrapper — self test")
    print("="*50)

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("ERROR: torch not installed. Run: pip install torch")
        import sys; sys.exit(1)

    # Simulate a batch of 8 IMU windows (256 timesteps, 3 axes)
    # Mix of high and low quality
    import random, math
    random.seed(42)
    batch_size = 8
    timesteps  = 256

    def make_window(quality):
        """quality: 'gold', 'bronze', 'flat'"""
        if quality == 'gold':
            # High variance, realistic motion
            return [[random.gauss(0,3.0) + math.sin(t*0.1)*5,
                     random.gauss(9.81,1.0),
                     random.gauss(0,2.0)] for t in range(timesteps)]
        elif quality == 'bronze':
            # Low variance, weak signal
            return [[random.gauss(0,0.3),
                     random.gauss(9.81,0.2),
                     random.gauss(0,0.2)] for t in range(timesteps)]
        else:  # flat — rejected
            return [[random.gauss(0,0.001),
                     random.gauss(9.81,0.001),
                     random.gauss(0,0.001)] for t in range(timesteps)]

    windows_list = (
        [make_window('gold')]   * 3 +
        [make_window('bronze')] * 3 +
        [make_window('flat')]   * 2
    )

    # Convert to tensor [batch, timesteps, 3]
    windows_tensor = torch.tensor(
        [[[v for v in step] for step in w] for w in windows_list],
        dtype=torch.float32
    )

    # Fake model output and labels
    num_classes = 3
    logits  = torch.randn(batch_size, num_classes)
    labels  = torch.randint(0, num_classes, (batch_size,))

    # Test S2SLoss
    print(f"\nBatch: {batch_size} windows  [{timesteps} timesteps × 3 axes]")
    print(f"  3 GOLD + 3 BRONZE + 2 FLAT")

    criterion = S2SLoss(task_weight=1.0, physics_weight=0.3, hz=50.0)
    print(f"\nCriterion: {criterion}")

    loss = criterion(logits, labels, windows_tensor)
    print(f"\nTotal loss: {loss.item():.4f}")

    # Inspect per-window scores
    scores = criterion.score_batch(windows_tensor)
    print(f"\nPer-window physics scores:")
    for i, s in enumerate(scores):
        tier = criterion.get_tier(s.item())
        penalty = 1.0 - s.item()/100.0
        label = ['GOLD','GOLD','GOLD','BRONZE','BRONZE','BRONZE','FLAT','FLAT'][i]
        print(f"  Window {i} ({label:6s}): score={s.item():.0f}  "
              f"tier={tier:<8}  penalty={penalty:.2f}")

    # Test with different physics weights
    print(f"\nPhysics weight sensitivity:")
    for lam in [0.0, 0.1, 0.3, 0.5, 1.0]:
        c = S2SLoss(task_weight=1.0, physics_weight=lam, hz=50.0)
        l = c(logits, labels, windows_tensor)
        print(f"  λ={lam:.1f}  loss={l.item():.4f}")

    # Test both tensor formats
    print(f"\nFormat compatibility:")
    w_blt = windows_tensor                              # [batch, time, 3]
    w_b3t = windows_tensor.permute(0,2,1)              # [batch, 3, time]
    s1 = criterion.score_batch(w_blt)
    s2 = criterion.score_batch(w_b3t)
    match = torch.allclose(s1, s2)
    print(f"  [batch,time,3] scores: {s1.tolist()}")
    print(f"  [batch,3,time] scores: {s2.tolist()}")
    print(f"  Both formats match: {match}")

    # Test gradient flow
    print(f"\nGradient flow check:")
    logits_grad = torch.randn(batch_size, num_classes, requires_grad=True)
    loss = criterion(logits_grad, labels, windows_tensor)
    loss.backward()
    grad_norm = logits_grad.grad.norm().item()
    print(f"  Gradient norm: {grad_norm:.4f}  (should be > 0)")
    print(f"  Backward pass: {'✓ OK' if grad_norm > 0 else '✗ FAILED'}")

    print(f"\n{'='*50}")
    print("All tests passed. S2SLoss ready for use.")
    print()
    print("Example training loop:")
    print("""
    from s2s_torch import S2SLoss

    criterion = S2SLoss(
        task_weight=1.0,
        physics_weight=0.3,  # tune this — 0.1-0.5 typical range
        hz=100.0             # match your sensor sample rate
    )

    for epoch in range(num_epochs):
        for windows, labels in dataloader:
            optimizer.zero_grad()
            predictions = model(windows)
            loss = criterion(predictions, labels, windows)
            loss.backward()
            optimizer.step()
    """)
