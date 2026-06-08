"""
YOLO6DoF 실시간 추론 래퍼 (최적화판).

naviEYE pose predictor와 동일 인터페이스로, BGR np.ndarray 프레임 하나를 받아
preprocess → forward → postprocess를 수행한다.

속도 최적화
  - 전처리를 GPU에서 수행 (PIL letterbox 제거): 업로드 후 F.interpolate + pad
  - fp16 autocast forward (Edge 배포 표준, forward 시간 단축)
  - NMS 전 top-k 프리필터로 후처리 비용 절감
`predictor.model`(nn.Module)을 노출해 forward 시간 후킹이 가능하다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..core.model import YOLO6DoF
from ..utils.camera import resolve_intrinsics
from ..utils.nms import nms, xywh2xyxy


class YOLO6DoFPredictor:
    def __init__(self, ckpt_path: str, ini: str | None = None,
                 device: str | None = None, conf: float = 0.25, iou: float = 0.5,
                 half: bool | None = None, pre_topk: int = 300, fuse: bool = True,
                 intrinsics=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.nc = ck["nc"]
        self.rd = 6 if ck["rot_repr"] == "6d" else 4
        self.imgsz = ck["imgsz"]
        self.conf, self.iou, self.pre_topk = conf, iou, pre_topk
        self.half = (self.device.type == "cuda") if half is None else half

        self.model = YOLO6DoF(nc=self.nc, scale=ck["scale"], rot_repr=ck["rot_repr"],
                              light_head=ck.get("light_head", True)).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        if fuse:
            self.model.fuse()                            # Conv+BN 융합 (추론 가속)

        # intrinsics: 파일(.ini) / K행렬 / dict / 라이브 Orbbec 파이프라인 등 무엇이든.
        # intrinsics 인자가 우선, 없으면 ini 경로. 둘 다 없으면 첫 프레임에서 근사.
        self.K0 = self.native = None
        if intrinsics is not None:
            self.set_intrinsics(intrinsics)
        elif ini:
            self.set_intrinsics(ini)

    def set_intrinsics(self, source, width: int | None = None,
                       height: int | None = None) -> None:
        """런타임에 intrinsics 설정/갱신 (예: 카메라 스트림 시작 후 SDK 에서 직접).

          pred.set_intrinsics(pipeline)          # 라이브 Orbbec Pipeline
          pred.set_intrinsics(K, 1920, 1080)     # 3x3 K 행렬 + 해상도
          pred.set_intrinsics("cam.ini")         # .ini 파일
        """
        intr = resolve_intrinsics(source, width=width, height=height)
        self.K0 = intr.K
        self.native = (intr.width, intr.height)

    # ------------------------------------------------------------------ #
    def _preprocess(self, frame_bgr: np.ndarray):
        """BGR np.ndarray → (입력 텐서[1,3,S,S], letterbox 보정 K[1,3,3]). 전부 GPU."""
        H0, W0 = frame_bgr.shape[:2]
        S = self.imgsz
        # 업로드(uint8) 후 GPU에서 BGR→RGB, 정규화
        t = torch.from_numpy(frame_bgr).to(self.device, non_blocking=True)   # [H,W,3] uint8 BGR
        t = t.permute(2, 0, 1).flip(0).float().div_(255.0).unsqueeze(0)       # [1,3,H,W] RGB

        r = min(S / H0, S / W0)
        nh, nw = round(H0 * r), round(W0 * r)
        t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        canvas = torch.full((1, 3, S, S), 114 / 255.0, device=self.device, dtype=t.dtype)
        top, left = (S - nh) // 2, (S - nw) // 2
        canvas[:, :, top:top + nh, left:left + nw] = t

        if self.K0 is not None:                          # 원본 intrinsics → 프레임 크기 스케일
            K = self.K0.copy()
            K[0] *= W0 / self.native[0]
            K[1] *= H0 / self.native[1]
        else:                                            # intrinsics 미제공 → 근사
            f = float(max(W0, H0))
            K = np.array([[f, 0, W0 / 2], [0, f, H0 / 2], [0, 0, 1]], dtype=np.float64)
        K[0] *= r; K[1] *= r; K[0, 2] += left; K[1, 2] += top    # letterbox 보정
        return canvas, torch.from_numpy(K).float().unsqueeze(0).to(self.device)

    @torch.no_grad()
    def _postprocess(self, out: dict, K: torch.Tensor):
        det = out["det"][0].float()                      # [4+nc+rd+3, A]
        boxes = det[:4].T
        scores = det[4:4 + self.nc]
        conf, cls = scores.max(0)

        # NMS 전 top-k 프리필터 (8400개 정렬·NMS 비용 절감)
        mask = conf > self.conf
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            return []
        if idx.numel() > self.pre_topk:
            top = conf[idx].topk(self.pre_topk).indices
            idx = idx[top]

        keep = nms(xywh2xyxy(boxes[idx]), conf[idx], self.iou)
        sel = idx[keep]
        rot = det[4 + self.nc:4 + self.nc + self.rd].T[sel]
        size = det[4 + self.nc + self.rd:].T[sel]
        pose = self.model.decode_pose(out, K, (self.imgsz, self.imgsz),
                                      boxes[sel].unsqueeze(0), rot.unsqueeze(0),
                                      size.unsqueeze(0))
        R = pose["R"][0].cpu().numpy(); t = pose["t"][0].cpu().numpy()
        sz = pose["size"][0].cpu().numpy()
        c = conf[sel].cpu().numpy(); cl = cls[sel].cpu().numpy()
        return [{"cls": int(cl[k]), "conf": float(c[k]), "R": R[k], "t": t[k], "size": sz[k]}
                for k in range(len(sel))]

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray):
        x, K = self._preprocess(frame_bgr)
        with torch.autocast(device_type=self.device.type, enabled=self.half):
            out = self.model(x)                          # forward (후킹 지점)
        # autocast 출력(fp16) → 후처리(grid_sample 등)는 fp32로
        out = {"det": out["det"].float(), "depth": out["depth"].float()}
        return self._postprocess(out, K)
