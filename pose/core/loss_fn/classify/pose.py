import torch 
import torch.nn as nn 
import torch.nn.functional as F

class VarifocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # pred_score: Logits
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        
        with torch.autocast(device_type=pred_score.device.type, enabled=False):
            loss = (
                F.binary_cross_entropy_with_logits(
                    pred_score.float(), 
                    gt_score.float(), 
                    reduction="none"
                ) * weight
            ).sum()
        return loss