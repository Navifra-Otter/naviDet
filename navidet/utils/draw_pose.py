"""Draw pose predictions (boxes + keypoints) on an image.

Used by `engine/_pose_engine.evaluate()` to dump a small random sample of
validation predictions to `<output_dir>/results/epoch_<N>/`.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


_FONT = None


def _font():
    global _FONT
    if _FONT is not None:
        return _FONT
    try:
        _FONT = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        _FONT = ImageFont.load_default()
    return _FONT


def _palette(n: int) -> list[tuple[int, int, int]]:
    """Distinct RGB colors via golden-ratio HSV."""
    out = []
    for i in range(max(n, 1)):
        h = (i * 0.61803398875) % 1.0
        s, v = 0.8, 0.95
        # HSV → RGB
        i6 = int(h * 6)
        f = h * 6 - i6
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        rgb = [
            (v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q),
        ][i6 % 6]
        out.append(tuple(int(c * 255) for c in rgb))
    return out


def draw_pose_prediction(
    image: Image.Image,
    keypoints: np.ndarray,        # (M, K*3) flat [x0,y0,v0,x1,y1,v1,...]
    scores: np.ndarray,           # (M,)
    labels: np.ndarray,           # (M,)
    *,
    num_keypoints: int,
    score_thresh: float = 0.3,
    skeleton: Iterable[tuple[int, int]] | None = None,
    boxes: np.ndarray | None = None,  # optional (M, 4) xyxy
) -> Image.Image:
    """Returns a copy of `image` with predictions overlaid.

    Skips instances below `score_thresh`. Draws a colored circle per keypoint
    and (optionally) connects keypoints along `skeleton` edges. If `boxes` is
    provided, draws bounding boxes with score / class label.
    """
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _font()

    if keypoints.size == 0:
        return out

    keep = scores >= score_thresh
    if not keep.any():
        return out
    kp = keypoints[keep].reshape(-1, num_keypoints, 3)
    sc = scores[keep]
    lb = labels[keep]
    bx = boxes[keep] if boxes is not None else None

    colors = _palette(len(kp))
    for inst_idx, (inst_kp, s, l, color) in enumerate(zip(kp, sc, lb, colors)):
        if bx is not None:
            x1, y1, x2, y2 = bx[inst_idx].tolist()
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            tag = f"#{int(l)}  {float(s):.2f}"
            tw = draw.textlength(tag, font=font) if hasattr(draw, "textlength") else 60
            draw.rectangle([x1, y1 - 16, x1 + tw + 6, y1], fill=color)
            draw.text((x1 + 3, y1 - 15), tag, fill="white", font=font)

        # skeleton edges
        if skeleton is not None:
            for a, b in skeleton:
                if a < num_keypoints and b < num_keypoints \
                        and inst_kp[a, 2] > 0 and inst_kp[b, 2] > 0:
                    draw.line(
                        [tuple(inst_kp[a, :2]), tuple(inst_kp[b, :2])],
                        fill=color, width=2,
                    )

        # keypoint dots + indices
        for k_idx, (x, y, v) in enumerate(inst_kp):
            if v <= 0:
                continue
            r = 4
            draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline="black")
            draw.text((x + 5, y - 7), str(k_idx), fill="white", font=font)

    # header strip with sample count
    header = f"{len(kp)} instance(s)  thr={score_thresh:.2f}"
    draw.rectangle([0, 0, 320, 18], fill=(0, 0, 0))
    draw.text((4, 1), header, fill="white", font=font)
    return out


# Skeletons for common K values; users can extend.
SKELETON_4_CORNERS = ((0, 1), (1, 2), (2, 3), (3, 0))  # closed quad (pallets, etc.)
SKELETON_17_COCO = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6),
)


def default_skeleton(num_keypoints: int):
    if num_keypoints == 4:
        return SKELETON_4_CORNERS
    if num_keypoints == 17:
        return SKELETON_17_COCO
    return None
