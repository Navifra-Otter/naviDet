import torch
import torch.nn as nn

class KeypointLoss(nn.Module):
    def __init__(self, sigmas: torch.Tensor) -> None:
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor) -> torch.Tensor:
        area = area.unsqueeze(-1)  # (N, 1)
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

class ImprovedKeypointLoss(nn.Module):
    """
    개선된 Keypoint Loss
    - OKS 기반 유지
    - Wing Loss 스타일 gradient 개선
    - 더 안정적인 정규화
    """
    def __init__(self, sigmas: torch.Tensor, omega: float = 10.0) -> None:
        super().__init__()
        self.sigmas = sigmas
        self.omega = omega
        
    def forward(
        self, 
        pred_kpts: torch.Tensor,  # (N, nkpts, 2)
        gt_kpts: torch.Tensor,     # (N, nkpts, 2)
        kpt_mask: torch.Tensor,    # (N, nkpts)
        area: torch.Tensor         # (N,)
    ) -> torch.Tensor:
        
        area = area.unsqueeze(-1).clamp(min=1.0)  # 최소값 보장
        
        # Euclidean distance squared
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + \
            (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        
        # OKS normalization
        variance = (2 * self.sigmas).pow(2) * area * 2
        e = d / (variance + 1e-6)  # 더 큰 epsilon
        
        # Wing Loss 스타일: 작은 오차에 더 집중
        threshold = 2.0
        loss = torch.where(
            e < threshold,
            self.omega * torch.log(1 + e / self.omega + 0.0 * e),
            e - threshold + self.omega * torch.log(1 + threshold / self.omega + 0.0 * e)
        )
        
        # Visibility masking
        loss = loss * kpt_mask
        
        # 정규화: visible keypoint 개수로
        num_visible = kpt_mask.sum().clamp(min=1.0)
        return loss.sum() / num_visible


class RobustKeypointLoss(nn.Module):
    """
    더욱 강건한 버전 - Adaptive Wing Loss + OKS
    """
    def __init__(
        self, 
        sigmas: torch.Tensor,
        alpha: float = 2.1,
        omega: float = 14,
        epsilon: float = 1,
        theta: float = 0.5
    ) -> None:
        super().__init__()
        self.sigmas = sigmas
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.theta = theta
        
    def forward(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor
    ) -> torch.Tensor:
        
        area = area.unsqueeze(-1).clamp(min=1.0)
        
        # L2 distance
        diff = torch.sqrt(
            (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + 
            (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2) + 1e-8
        )
        
        # OKS normalization
        sigma_area = self.sigmas * torch.sqrt(area)
        diff_normalized = diff / (sigma_area + 1e-6)
        
        # Adaptive Wing Loss
        C = self.omega * (1 - torch.log(torch.tensor(1 + self.omega / self.epsilon)))
        
        loss = torch.where(
            diff_normalized < self.theta,
            self.omega * torch.log(1 + torch.pow(diff_normalized / self.epsilon, self.alpha - diff_normalized)),
            diff_normalized - C
        )
        
        loss = loss * kpt_mask
        num_visible = kpt_mask.sum().clamp(min=1.0)
        
        return loss.sum() / num_visible
    
class MultiScaleKeypointLoss(nn.Module):
    """
    Multi-scale 일관성을 고려한 Loss
    """
    def __init__(self, sigmas: torch.Tensor, scales: list = [0.5, 1.0, 2.0]):
        super().__init__()
        self.sigmas = sigmas
        self.scales = scales
        self.base_loss = ImprovedKeypointLoss(sigmas)
        
    def forward(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor
    ) -> torch.Tensor:
        
        total_loss = 0
        
        # Multi-scale evaluation
        for scale in self.scales:
            scaled_pred = pred_kpts * scale
            scaled_gt = gt_kpts * scale
            scaled_area = area * (scale ** 2)
            
            loss = self.base_loss(scaled_pred, scaled_gt, kpt_mask, scaled_area)
            total_loss += loss
            
        return total_loss / len(self.scales)
    
class HybridKeypointLoss(nn.Module):
    """
    여러 Loss의 장점을 결합
    - OKS Loss (metric consistency)
    - L1 Loss (stable gradient)
    - Smoothness regularization
    """
    def __init__(
        self, 
        sigmas: torch.Tensor,
        lambda_oks: float = 1.0,
        lambda_l1: float = 0.5,
        lambda_smooth: float = 0.1
    ):
        super().__init__()
        self.sigmas = sigmas
        self.lambda_oks = lambda_oks
        self.lambda_l1 = lambda_l1
        self.lambda_smooth = lambda_smooth
        
        self.oks_loss = ImprovedKeypointLoss(sigmas)
        
    def forward(
        self,
        pred_kpts: torch.Tensor,
        gt_kpts: torch.Tensor,
        kpt_mask: torch.Tensor,
        area: torch.Tensor
    ) -> tuple:
        
        # 1. OKS-based loss
        loss_oks = self.oks_loss(pred_kpts, gt_kpts, kpt_mask, area)
        
        # 2. L1 loss (더 안정적인 gradient)
        diff = torch.abs(pred_kpts - gt_kpts).sum(dim=-1)
        loss_l1 = (diff * kpt_mask).sum() / kpt_mask.sum().clamp(min=1.0)
        
        # 3. Smoothness regularization (인접 keypoint 간)
        if pred_kpts.shape[1] > 1:
            kpt_diff = pred_kpts[:, 1:] - pred_kpts[:, :-1]
            loss_smooth = kpt_diff.pow(2).sum() / pred_kpts.shape[0]
        else:
            loss_smooth = torch.tensor(0.0, device=pred_kpts.device)
        
        # Combined loss
        total_loss = (
            self.lambda_oks * loss_oks +
            self.lambda_l1 * loss_l1 +
            self.lambda_smooth * loss_smooth
        )
        
        return total_loss