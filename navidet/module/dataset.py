"""
학습용 PyTorch Dataset — task에 따라 두 종류의 GT를 생성한다.

task="6dof" (기본):
  - RGB(1920x1080)를 letterbox로 imgsz 정사각으로 리사이즈
  - YOLO 4-keypoint 라벨 + depth로 6DoF(R,t,size)를 '원본 픽셀/원본 K'에서 계산
    (R,t,size는 메트릭이라 리사이즈 불변)
  - bbox와 intrinsics K만 letterbox 변환에 맞춰 조정
  - dense depth GT를 depth-head 해상도(imgsz/2)로 제공
  반환 타깃은 Pose6DoFLoss가 기대하는 형식과 동일.

task="pose"/"detect" (2D 멀티태스크):
  - RGB만 사용(depth/RANSAC 불필요). YOLO 라벨에서 bbox + keypoint를 직접 읽어
    letterbox 픽셀 좌표로 변환 → gt_bboxes, gt_kpts 생성.
  반환 타깃은 MultiTaskLoss가 기대하는 형식(gt_labels/gt_bboxes/mask_gt/gt_kpts).
"""

from __future__ import annotations

import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..utils.camera import load_orbbec_ini
from .pose_label import keypoints_to_pose, parse_yolo_kpt_label


def letterbox_rgb(img: Image.Image, new: int, color=114):
    """비율 유지 리사이즈 + 패딩. 반환: np uint8[new,new,3], r, (left, top)."""
    w, h = img.size
    r = min(new / h, new / w)
    nw, nh = round(w * r), round(h * r)
    resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    left, top = (new - nw) // 2, (new - nh) // 2
    canvas[top:top + nh, left:left + nw] = np.asarray(resized)
    return canvas, r, left, top


def letterbox_depth(depth_m: np.ndarray, new: int, r: float, left: int, top: int):
    """depth(meter)를 RGB와 동일 변환으로 letterbox (NEAREST, 패딩=0)."""
    h, w = depth_m.shape
    nw, nh = round(w * r), round(h * r)
    resized = np.asarray(
        Image.fromarray(depth_m.astype(np.float32), mode="F").resize((nw, nh), Image.NEAREST))
    canvas = np.zeros((new, new), dtype=np.float32)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


class PoseDataset(Dataset):
    def __init__(self, root: str, ini: str, imgsz: int = 640,
                 depth_scale: float = 0.001, num_kpts: int = 4,
                 rmse_thresh: float = 0.05, limit: int = 0,
                 use_cache: bool = True, cache_name: str = "pose_labels",
                 pose_mode: str = "full", task: str = "6dof",
                 kpt_shape: tuple[int, int] = (4, 3)):
        self.root = root
        self.imgsz = imgsz
        self.dep_size = imgsz // 2          # depth head 출력 해상도와 일치
        self.depth_scale = depth_scale
        self.num_kpts = num_kpts
        self.rmse_thresh = rmse_thresh
        self.pose_mode = pose_mode          # "face"(권장)|"full"|"bottom"|"corners" — 즉석계산 경로
        # task: "6dof"(R,t,size,depth GT) | "pose"/"detect"(2D 멀티태스크 GT)
        self.task = task
        self.is_2d = task in ("pose", "detect", "segment")
        self.kpt_dim = kpt_shape[1]         # 2(xy) / 3(xy+vis)

        intr = load_orbbec_ini(ini)
        self.K0 = intr.K                    # 원본 color intrinsics
        self.orig_wh = (intr.width, intr.height)

        # build_labels로 미리 구운 6DoF GT(npz)가 있으면 로드만 (반복 RANSAC 제거)
        cache_dir = os.path.join(root, cache_name)
        self.cache_dir = cache_dir if (use_cache and os.path.isdir(cache_dir)) else None
        if self.cache_dir:
            print(f"[dataset] 6DoF GT 캐시 사용: {cache_dir}")

        self.labels = sorted(glob(os.path.join(root, "labels", "*.txt")))
        if limit:
            self.labels = self.labels[:limit]

    def __len__(self):
        return len(self.labels)

    def _gt_original(self, stem, lp, depth_m):
        """원본 공간 6DoF GT(cls, box_xyxy, R, t, size) 리스트 반환.
        캐시(npz)가 있으면 로드만, 없으면 keypoint+depth로 즉석 계산."""
        cls_l, box_l, R_l, t_l, sz_l = [], [], [], [], []

        npz = os.path.join(self.cache_dir, stem + ".npz") if self.cache_dir else None
        if npz and os.path.exists(npz):                     # ── 캐시 로드 (RANSAC 생략) ──
            data = np.load(npz)
            keep = data["valid"] & (data["rmse"] <= self.rmse_thresh)
            return (list(data["cls"][keep]), list(data["bbox_xyxy"][keep]),
                    list(data["R"][keep]), list(data["t"][keep]), list(data["size"][keep]))

        # ── 캐시 없음: 즉석 계산 ──
        for line in open(lp).read().strip().split("\n"):
            if not line.strip():
                continue
            cls, bbox_n, kpts_n = parse_yolo_kpt_label(line, self.num_kpts)
            pose = keypoints_to_pose(cls, bbox_n, kpts_n, depth_m, self.K0,
                                     self.orig_wh, mode=self.pose_mode)
            if (not pose.valid) or pose.fit_rmse > self.rmse_thresh:
                continue
            cls_l.append(pose.cls); box_l.append(pose.bbox_xyxy)
            R_l.append(pose.R); t_l.append(pose.t); sz_l.append(pose.size)
        return cls_l, box_l, R_l, t_l, sz_l

    def _gt_pose_original(self, lp):
        """원본 픽셀 공간 2D GT 반환 — depth/RANSAC 불필요.
        반환: cls_l, box_l(xyxy px), kpt_l(각 [nk, kpt_dim] px, D=3이면 vis 포함).
        라벨이 'cls cx cy w h kx1 ky1 [v1] ...' (정규화)인 표준 YOLO-pose 포맷."""
        W, H = self.orig_wh
        cls_l, box_l, kpt_l = [], [], []
        for line in open(lp).read().strip().split("\n"):
            if not line.strip():
                continue
            v = list(map(float, line.split()))
            cls = int(v[0])
            cx, cy, bw, bh = v[1:5]                       # 정규화 (cx,cy,w,h)
            x1 = (cx - bw / 2) * W; y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W; y2 = (cy + bh / 2) * H
            rest = v[5:]
            stride = len(rest) // self.num_kpts           # 2(xy) 또는 3(xyv)
            kp = np.array(rest, dtype=np.float64).reshape(self.num_kpts, stride)
            out = np.zeros((self.num_kpts, self.kpt_dim), dtype=np.float64)
            out[:, 0] = kp[:, 0] * W                       # 픽셀 x
            out[:, 1] = kp[:, 1] * H                       # 픽셀 y
            if self.kpt_dim == 3:
                # 라벨에 visibility가 있으면 사용, 없으면 1.0(가시)로 간주
                out[:, 2] = kp[:, 2] if stride >= 3 else 1.0
            cls_l.append(cls)
            box_l.append(np.array([x1, y1, x2, y2], dtype=np.float64))
            kpt_l.append(out)
        return cls_l, box_l, kpt_l

    def _getitem_2d(self, idx):
        """task in {pose,detect}: RGB + 2D GT(bbox/keypoint)만 생성."""
        lp = self.labels[idx]
        stem = os.path.splitext(os.path.basename(lp))[0]
        img = Image.open(os.path.join(self.root, "images", stem + ".png")).convert("RGB")

        cls_l, box_l, kpt_l = self._gt_pose_original(lp)
        rgb, r, left, top = letterbox_rgb(img, self.imgsz)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        N = len(cls_l)
        if N:
            boxes = np.stack(box_l) * r
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            kpts = np.stack(kpt_l)                          # [N, nk, D]
            kpts[..., 0] = kpts[..., 0] * r + left
            kpts[..., 1] = kpts[..., 1] * r + top          # vis 채널(D=3)은 변환 없음
            gt_bboxes = torch.from_numpy(boxes).float()
            gt_labels = torch.tensor(cls_l).float().view(N, 1)
            gt_kpts = torch.from_numpy(kpts).float()
        else:
            gt_bboxes = torch.zeros(0, 4)
            gt_labels = torch.zeros(0, 1)
            gt_kpts = torch.zeros(0, self.num_kpts, self.kpt_dim)

        return {"img": x, "gt_labels": gt_labels, "gt_bboxes": gt_bboxes,
                "gt_kpts": gt_kpts, "stem": stem}

    def __getitem__(self, idx):
        if self.is_2d:
            return self._getitem_2d(idx)
        lp = self.labels[idx]
        stem = os.path.splitext(os.path.basename(lp))[0]
        img = Image.open(os.path.join(self.root, "images", stem + ".png")).convert("RGB")
        depth_m = np.asarray(
            Image.open(os.path.join(self.root, "depth", stem + ".png"))
        ).astype(np.float32) * self.depth_scale

        # 1) 원본 공간 6DoF GT (캐시 로드 또는 즉석 계산)
        cls_l, box_l, R_l, t_l, sz_l = self._gt_original(stem, lp, depth_m)

        # 2) letterbox 변환 (이미지/깊이/K/bbox)
        rgb, r, left, top = letterbox_rgb(img, self.imgsz)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0     # [3,H,W]

        dep = letterbox_depth(depth_m, self.imgsz, r, left, top)
        dep_small = dep[::2, ::2][:self.dep_size, :self.dep_size]      # imgsz/2
        gt_depth = torch.from_numpy(dep_small)[None]                   # [1,hd,wd]
        depth_mask = (gt_depth > 0).float()

        K = self.K0.copy()
        K[0] *= r; K[1] *= r                                          # fx,fy,cx,cy * r
        K[0, 2] += left; K[1, 2] += top                               # principal point 이동
        K = torch.from_numpy(K).float()

        N = len(cls_l)
        if N:
            boxes = np.stack(box_l) * r
            boxes[:, [0, 2]] += left
            boxes[:, [1, 3]] += top
            gt_bboxes = torch.from_numpy(boxes).float()
            gt_labels = torch.tensor(cls_l).float().view(N, 1)
            gt_rot = torch.from_numpy(np.stack(R_l)).float()
            gt_trans = torch.from_numpy(np.stack(t_l)).float()
            gt_size = torch.from_numpy(np.stack(sz_l)).float()
        else:
            gt_bboxes = torch.zeros(0, 4)
            gt_labels = torch.zeros(0, 1)
            gt_rot = torch.zeros(0, 3, 3)
            gt_trans = torch.zeros(0, 3)
            gt_size = torch.zeros(0, 3)

        return {
            "img": x, "K": K, "gt_depth": gt_depth, "depth_mask": depth_mask,
            "gt_labels": gt_labels, "gt_bboxes": gt_bboxes, "gt_rot": gt_rot,
            "gt_trans": gt_trans, "gt_size": gt_size, "stem": stem,
        }


def collate_fn(batch):
    """가변 N 오브젝트를 배치 최대 M으로 zero-padding.
    샘플 키로 task를 판별 — gt_kpts 있으면 2D(pose) 배치, 아니면 6dof 배치."""
    B = len(batch)
    M = max(max(s["gt_labels"].shape[0] for s in batch), 1)   # 빈 배치 방지
    imgs = torch.stack([s["img"] for s in batch])
    img_size = (imgs.shape[2], imgs.shape[3])

    def pad(key, shape):
        out = torch.zeros(B, M, *shape)
        for i, s in enumerate(batch):
            n = s[key].shape[0]
            if n:
                out[i, :n] = s[key]
        return out

    mask_gt = torch.zeros(B, M, 1)
    for i, s in enumerate(batch):
        mask_gt[i, :s["gt_labels"].shape[0]] = 1.0

    # ── 2D 멀티태스크(pose/detect) 배치 ──
    if "gt_kpts" in batch[0]:
        nk, D = batch[0]["gt_kpts"].shape[1:]
        targets = {
            "gt_labels": pad("gt_labels", (1,)),
            "gt_bboxes": pad("gt_bboxes", (4,)),
            "gt_kpts": pad("gt_kpts", (nk, D)),
            "mask_gt": mask_gt,
            "img_size": img_size,
        }
        return imgs, targets

    # ── 6DoF 배치 ──
    K = torch.stack([s["K"] for s in batch])
    gt_depth = torch.stack([s["gt_depth"] for s in batch])
    depth_mask = torch.stack([s["depth_mask"] for s in batch])
    targets = {
        "gt_labels": pad("gt_labels", (1,)),
        "gt_bboxes": pad("gt_bboxes", (4,)),
        "gt_rot": pad("gt_rot", (3, 3)),
        "gt_trans": pad("gt_trans", (3,)),
        "gt_size": pad("gt_size", (3,)),
        "mask_gt": mask_gt,
        "K": K, "img_size": img_size,
        "gt_depth": gt_depth, "depth_mask": depth_mask,
    }
    return imgs, targets
