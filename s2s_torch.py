import torch
from s2s_standard_v1_3.s2s_physics_v1_3 import PhysicsEngine

class S2SPhysicsLoss(torch.nn.Module):
    def __init__(self, task_loss_fn, lambda_physics=0.1):
        super().__init__()
        self.task_loss = task_loss_fn
        self.lambda_physics = lambda_physics
        self.engine = PhysicsEngine()

    def forward(self, predictions, targets, imu_batch):
        task_l = self.task_loss(predictions, targets)
        scores = []
        for sample in imu_batch:
            result = self.engine.certify(sample)
            scores.append(result['physical_law_score'] / 100.0)
        physics_scores = torch.tensor(scores, dtype=torch.float32)
        physics_penalty = (1.0 - physics_scores).mean()
        return task_l + self.lambda_physics * physics_penalty
