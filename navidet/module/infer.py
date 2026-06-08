"""
추론 + 시각화 공용 헬퍼.

단일 샘플에 대해 6DoF를 추론하고, 예측(청록)·GT(초록) 쿼드와 좌표축을 그린 PIL
이미지와 발행용 메시지를 돌려준다. tools/predict.py(배치 시각화)와 학습 루프의
epoch별 검증 시각화가 함께 사용한다.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageDraw

from ..messaging.publisher import build_message
from ..utils.nms import nms, xywh2xyxy
from ..utils.visualize import draw_axes, draw_quad
from .pose_label import canonical_quad


def tensor_to_pil(img: torch.Tensor) -> Image.Image:
    arr = (img.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def euler_deg(R: np.ndarray) -> tuple[float, float, float]:
    """회전행렬 → (roll, pitch, yaw) 도 (XYZ Tait-Bryan)."""
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return tuple(float(np.degrees(a)) for a in (roll, pitch, yaw))


def render_gt(sample, class_names=None) -> Image.Image:
    """모델 없이 GT(초록 쿼드 + 좌표축)만 그린 PIL 이미지. (한 번만 저장용)"""
    pil = tensor_to_pil(sample["img"])
    draw = ImageDraw.Draw(pil)
    Kc = sample["K"].numpy()
    if sample["gt_labels"].shape[0] > 0:
        gtR = sample["gt_rot"].numpy(); gtt = sample["gt_trans"].numpy()
        gts = sample["gt_size"].numpy()
        for k in range(gtR.shape[0]):
            quad = (canonical_quad(gts[k, 0], gts[k, 1]) @ gtR[k].T) + gtt[k]
            draw_quad(draw, quad, Kc, color=(0, 255, 0), w=1, label=False)
            draw_axes(draw, gtR[k], gtt[k], Kc, length=0.2, w=2)
    return pil


@torch.no_grad()
def render_sample(model, sample, class_names=None, conf=0.25, iou=0.5,
                  draw_gt=True, use_sensor_depth=False):
    """
    sample: PoseDataset.__getitem__ 결과 dict.
    use_sensor_depth: True면 translation을 sample["gt_depth"](센서 depth)로 복원.
    반환: (PIL.Image, [PoseMessage]).
    """
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    model.head.return_raw = False                    # 디코딩된 텐서 출력 보장

    nc = model.nc
    rd = model.head.rot_dim
    img = sample["img"].unsqueeze(0).to(device)
    K = sample["K"].unsqueeze(0).to(device)
    H, W = img.shape[-2:]

    out = model(img)
    det = out["det"][0]                              # [4+nc+rd+3, A]
    boxes_xywh = det[:4].T
    scores = det[4:4 + nc]
    rot_raw = det[4 + nc:4 + nc + rd].T
    size = det[4 + nc + rd:].T
    cf, cls = scores.max(0)

    pil = tensor_to_pil(sample["img"])
    draw = ImageDraw.Draw(pil)
    Kc = sample["K"].numpy()
    msgs = []

    mask = cf > conf
    if mask.any():
        keep = nms(xywh2xyxy(boxes_xywh[mask]), cf[mask], iou)
        sel = mask.nonzero(as_tuple=False).squeeze(1)[keep]
        if len(sel):
            dm = (sample["gt_depth"].unsqueeze(0).to(device)
                  if use_sensor_depth and "gt_depth" in sample else None)
            pose = model.decode_pose(out, K, (H, W),
                                     boxes_xywh[sel].unsqueeze(0),
                                     rot_raw[sel].unsqueeze(0), size[sel].unsqueeze(0),
                                     depth_map=dm)
            R = pose["R"][0].cpu().numpy(); t = pose["t"][0].cpu().numpy()
            sz = pose["size"][0].cpu().numpy()
            for k in range(len(sel)):
                quad = (canonical_quad(sz[k, 0], sz[k, 1]) @ R[k].T) + t[k]
                # 6DoF는 전면 쿼드+좌표축으로 표현 (2D 검출박스는 쿼드와 겹쳐 중복 → 생략)
                draw_quad(draw, quad, Kc, color=(0, 230, 230), w=1, label=False)
                draw_axes(draw, R[k], t[k], Kc, length=0.2, w=2)
                bb = xywh2xyxy(boxes_xywh[sel[k]]).cpu().numpy()   # 텍스트 위치용
                ci = int(cls[sel[k]]); c = float(cf[sel[k]])
                name = class_names[ci] if class_names and ci < len(class_names) else str(ci)
                # 클래스/score는 박스 위, R·t 값은 박스 아래에 표기
                draw.text((bb[0], bb[1] - 12), f"{name} {c:.2f}", fill=(0, 230, 230))
                rpy = euler_deg(R[k])
                lines = [
                    f"t=({t[k,0]:.2f},{t[k,1]:.2f},{t[k,2]:.2f})m",
                    f"R rpy=({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})deg",
                ]
                for li, txt in enumerate(lines):
                    draw.text((bb[0], bb[3] + 2 + li * 11), txt, fill=(0, 230, 230))
                msgs.append(build_message(ci, c, R[k], t[k], sz[k]))

    if draw_gt and sample["gt_labels"].shape[0] > 0:
        gtR = sample["gt_rot"].numpy(); gtt = sample["gt_trans"].numpy()
        gts = sample["gt_size"].numpy()
        for k in range(gtR.shape[0]):
            quad = (canonical_quad(gts[k, 0], gts[k, 1]) @ gtR[k].T) + gtt[k]
            draw_quad(draw, quad, Kc, color=(0, 255, 0), w=1, label=False)

    model.train(was_training)
    return pil, msgs
