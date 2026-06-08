"""
시각화 유틸: 3D pose 재투영 그리기 + 학습 곡선 plot.
"""

from __future__ import annotations

import numpy as np
from PIL import ImageDraw

import matplotlib
matplotlib.use("Agg")            # 헤드리스 환경 대비
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- #
#  3D pose 재투영 그리기
# ----------------------------------------------------------------------------- #
def project(P3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """[N,3] 카메라좌표 → [N,2] 픽셀."""
    p = (K @ P3d.T).T
    return p[:, :2] / p[:, 2:3]


def draw_axes(draw: ImageDraw.ImageDraw, R, t, K, length=0.15, w=4):
    """오브젝트 좌표축(X=빨강, Y=초록, Z=파랑)을 재투영해 그린다."""
    origin = project(t[None], K)[0]
    axes = np.stack([t + R[:, i] * length for i in range(3)])
    ends = project(axes, K)
    for end, color in zip(ends, [(255, 60, 60), (60, 255, 60), (80, 120, 255)]):
        draw.line([tuple(origin), tuple(end)], fill=color, width=w)


def draw_quad(draw, kpts3d, K, color=(255, 255, 0), w=3, label=True):
    """전면 4코너 쿼드를 재투영해 그린다 (코너 번호 0=TR 시계방향)."""
    uv = project(kpts3d, K)
    for i in range(4):
        draw.line([tuple(uv[i]), tuple(uv[(i + 1) % 4])], fill=color, width=w)
    if label:
        for i, c in enumerate(uv):
            draw.ellipse([c[0]-5, c[1]-5, c[0]+5, c[1]+5], outline=(255, 255, 255), width=2)
            draw.text((c[0]+6, c[1]+6), str(i), fill=(255, 255, 255))


# ----------------------------------------------------------------------------- #
#  학습 곡선
# ----------------------------------------------------------------------------- #
TRAIN_TERMS = ["total", "box", "cls", "rot", "size", "depth", "trans"]
VAL_TERMS = [("rot_deg", "val rot (deg)"), ("trans_mm", "val trans (mm)"),
             ("width_mm", "val width (mm)"), ("recall", "val recall")]


def plot_history(history: list[dict], out_path: str):
    """
    history: [{"epoch","lr","train":{손실...},"val":{pose 지표...}|None}, ...] → PNG.
    위 2행: train 손실 항목 + lr / 아래 1행: 검증 pose 지표(회전·translation·width·recall).
    """
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    has_val = history[0].get("val") is not None
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    flat = list(axes.flat)

    for ax, term in zip(flat[:7], TRAIN_TERMS):           # train 손실 (파랑)
        ax.plot(epochs, [h["train"].get(term) for h in history], "-o", ms=3, color="tab:blue")
        ax.set_title("train " + term); ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3)
    flat[7].plot(epochs, [h["lr"] for h in history], "-o", ms=3, color="tab:green")
    flat[7].set_title("lr"); flat[7].set_xlabel("epoch"); flat[7].grid(True, alpha=0.3)

    for ax, (key, title) in zip(flat[8:12], VAL_TERMS):   # 검증 pose 지표 (빨강)
        if has_val:
            ax.plot(epochs, [(h["val"] or {}).get(key) for h in history], "-s", ms=3, color="tab:red")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3)

    fig.suptitle("YOLO6DoF training", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
