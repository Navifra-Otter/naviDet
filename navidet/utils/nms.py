"""박스 변환 + 간단한 NMS (torchvision 의존 없이)."""

from __future__ import annotations

import torch


def xywh2xyxy(b: torch.Tensor) -> torch.Tensor:
    x, y, w, h = b.unbind(-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], -1)


def nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5,
        topk: int = 50) -> list[int]:
    """greedy NMS. 반환: 살아남은 인덱스 리스트."""
    if boxes_xyxy.numel() == 0:
        return []
    x1, y1, x2, y2 = boxes_xyxy.unbind(1)
    area = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0 and len(keep) < topk:
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.max(x1[i], x1[rest]); yy1 = torch.max(y1[i], y1[rest])
        xx2 = torch.min(x2[i], x2[rest]); yy2 = torch.min(y2[i], y2[rest])
        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep
