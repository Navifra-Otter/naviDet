    
import torch

def bbox_iou(box1, box2, CIoU=True, eps=1e-7):
    """
    box1, box2: (N, 4) -> (cx, cy, w, h) format
    """
    # xywh -> xyxy
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # Intersection
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    w1, h1 = box1[:, 2], box1[:, 3]
    w2, h2 = box2[:, 2], box2[:, 3]
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    
    if CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        c2 = cw ** 2 + ch ** 2 + eps
        rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4
        v = (4 / (3.141592 ** 2)) * torch.pow(torch.atan(w2 / (h2+eps)) - torch.atan(w1 / (h1+eps)), 2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        ciou = iou - (rho2 / c2 + v * alpha)
        return torch.nan_to_num(ciou, nan=0.0)
    return torch.nan_to_num(iou, nan=0.0)

def point2box_xywh(points, visibility):
    """
    Visibility를 고려한 안정적인 box 추정
    points: (N, nkpts, 2)
    visibility: (N, nkpts)
    """
    valid_mask = visibility > 0
    
    boxes = []
    for i in range(points.shape[0]):
        visible_points = points[i][valid_mask[i]]
        
        if visible_points.shape[0] < 2:
            visible_points = points[i]
        
        # Outlier rejection: percentile 기반
        if visible_points.shape[0] > 4:
            x_sorted, _ = torch.sort(visible_points[:, 0])
            y_sorted, _ = torch.sort(visible_points[:, 1])
            
            trim = max(1, int(visible_points.shape[0] * 0.1))
            x_min = x_sorted[trim]
            x_max = x_sorted[-trim-1]
            y_min = y_sorted[trim]
            y_max = y_sorted[-trim-1]
        else:
            x_min = visible_points[:, 0].min()
            x_max = visible_points[:, 0].max()
            y_min = visible_points[:, 1].min()
            y_max = visible_points[:, 1].max()
        
        w = (x_max - x_min).clamp(min=1.0)
        h = (y_max - y_min).clamp(min=1.0)
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        
        boxes.append(torch.stack([cx, cy, w, h]))
    
    return torch.stack(boxes)