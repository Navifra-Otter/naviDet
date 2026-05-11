"""YOLO-format pose dataset.

Two output modes:

  * **EC mode** (`ec_format=True`, default when `transforms` is supplied) —
    `__getitem__` returns `(PIL.Image, target_dict)` shaped exactly like
    `CocoDetection`, so EdgeCrafter's transforms (Mosaic / Resize / ColorJitter
    / RandomZoomOut / Normalize / ConvertBoxes / …) and `BatchImageCollateFunction`
    work without modification.

  * **Legacy mode** (`ec_format=False`, no `transforms`) — preserves the
    historical `(uint8 CHW tensor, flat label tensor)` output for the `--simple`
    Trainer path.
"""

import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from tqdm import tqdm

from navidet.core.registry import register
from navidet.utils import MASTER_RANK, colored_msg, printS


@register()
class YoloPoseDataset(Dataset):
    # Tell the EC registry to build `transforms` (a Compose) before passing it
    # in — otherwise we'd receive a raw dict spec.
    __inject__ = ["transforms"]

    """YOLO-format pose dataset (EC-pipeline ready).

    Args:
        img_dir / label_dir: standard YOLO layout — images and `.txt` labels.
        img_size:        used only in legacy mode for the in-dataset resize.
        nkpts:           number of keypoints per instance.
        transforms:      EC-style transform (e.g. Compose); when provided the
                         dataset emits the EC-compatible (PIL, target_dict).
        ec_format:       force EC output regardless of `transforms`. Defaults to
                         True when transforms is given.
        cache_ram:       cache decoded uint8 numpy arrays in RAM (legacy only).
    """

    def __init__(
        self,
        img_dir,
        label_dir,
        img_size: int = 640,
        nkpts: int = 4,
        transforms=None,
        transform=None,  # legacy alias
        ec_format: bool | None = None,
        cache_ram: bool = False,
    ):
        if transforms is None:
            transforms = transform
        self._transforms = transforms
        self._ec_format = (transforms is not None) if ec_format is None else ec_format

        if not os.path.isdir(img_dir):
            raise FileNotFoundError(
                f"YoloPoseDataset: image directory not found: {img_dir!r}. "
                f"Check `train_dataloader.dataset.img_dir` (or `valid_*`) in your YAML."
            )
        if not os.path.isdir(label_dir):
            raise FileNotFoundError(
                f"YoloPoseDataset: label directory not found: {label_dir!r}."
            )

        # Case-insensitive match for common extensions.
        exts = ("jpg", "jpeg", "png", "bmp", "webp")
        self.img_files = sorted({
            p for ext in exts
            for pat in (f"*.{ext}", f"*.{ext.upper()}")
            for p in glob.glob(os.path.join(img_dir, pat))
        })
        if not self.img_files:
            raise RuntimeError(
                f"YoloPoseDataset: no images found in {img_dir!r} "
                f"(extensions tried: {exts}). Verify the path and image format."
            )
        self.label_dir = label_dir
        self.img_size = img_size
        self.nkpts = nkpts
        self.cache_ram = cache_ram and not self._ec_format
        self._ram_cache = {} if self.cache_ram else None

        # epoch counter for transforms with curriculum (e.g. Mosaic stop_epoch)
        self.epoch = -1

        # Pre-parse labels into flat numpy arrays so __getitem__ stays fast.
        # Columns: [img_idx_placeholder, cls, cx, cy, w, h, k0x, k0y, k0v, ...]
        # All coords are YOLO-normalized [0, 1].
        printS(f"Loading labels for {len(self.img_files)} images...")
        self.cached_labels = []
        pbar = tqdm(self.img_files,
                    desc=f" {colored_msg('[SYSTEMS]', 'blue')} Parsing Labels") \
            if MASTER_RANK else self.img_files
        for img_path in pbar:
            label_path = os.path.join(
                self.label_dir,
                os.path.basename(img_path).rsplit('.', 1)[0] + ".txt")
            targets = np.zeros((0, 6 + self.nkpts * 3))
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                try:
                    labels = np.loadtxt(label_path, ndmin=2)
                    if labels.shape[0] > 0:
                        ncols = labels.shape[1]
                        expected_xy = 5 + self.nkpts * 2
                        expected_xyv = 5 + self.nkpts * 3
                        if ncols == expected_xy:
                            kpts = labels[:, 5:].reshape(-1, self.nkpts, 2)
                            vis = np.ones((len(labels), self.nkpts, 1))
                            kpts = np.concatenate([kpts, vis], axis=2)\
                                .reshape(-1, self.nkpts * 3)
                        elif ncols == expected_xyv:
                            kpts = labels[:, 5:]
                        else:
                            printS(f"Skipping {label_path}: expected "
                                   f"{expected_xy} or {expected_xyv} cols, got {ncols}")
                            self.cached_labels.append(targets)
                            continue
                        boxes = labels[:, 1:5].copy()
                        cls = labels[:, 0:1].copy()
                        zeros = np.zeros((len(labels), 1))
                        targets = np.concatenate([zeros, cls, boxes, kpts], axis=1)
                except Exception:
                    pass
            self.cached_labels.append(targets)

    def __len__(self) -> int:
        return len(self.img_files)

    # ------------------------------------------------------------------ EC
    def set_epoch(self, epoch: int) -> None:
        """Hook expected by EC transforms with curriculum (Mosaic, etc.)."""
        self.epoch = int(epoch)

    def _ec_target(self, idx: int, w: int, h: int) -> dict:
        """Convert the cached flat YOLO labels to an EC-style target dict."""
        flat = self.cached_labels[idx]
        n = flat.shape[0]

        if n == 0:
            empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
            return {
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "idx": torch.tensor([idx], dtype=torch.int64),
                "boxes": tv_tensors.BoundingBoxes(
                    empty_boxes, format=tv_tensors.BoundingBoxFormat.XYXY,
                    canvas_size=(h, w)),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "keypoints": torch.zeros((0, self.nkpts, 3), dtype=torch.float32),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "orig_size": torch.tensor([w, h], dtype=torch.int64),
                "size": torch.tensor([w, h], dtype=torch.int64),
            }

        cls = torch.from_numpy(flat[:, 1]).long()
        # YOLO boxes are normalized CXCYWH. Denormalize, then convert to XYXY
        # which is what EC's CocoEvaluator and ConvertBoxes transform expect.
        cxcywh = torch.from_numpy(flat[:, 2:6]).float()
        cxcywh[:, [0, 2]] *= w
        cxcywh[:, [1, 3]] *= h
        boxes_xyxy = torch.empty_like(cxcywh)
        boxes_xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        boxes_xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        boxes_xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        boxes_xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2
        area = cxcywh[:, 2] * cxcywh[:, 3]

        kpts = torch.from_numpy(flat[:, 6:].reshape(n, self.nkpts, 3)).float()
        kpts[..., 0] *= w
        kpts[..., 1] *= h

        return {
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "idx": torch.tensor([idx], dtype=torch.int64),
            "boxes": tv_tensors.BoundingBoxes(
                boxes_xyxy, format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(h, w)),
            "labels": cls,
            "keypoints": kpts,
            "area": area,
            "iscrowd": torch.zeros((n,), dtype=torch.int64),
            # EC convention: (w, h) order — DETRPosePostProcessor scales
            # predicted keypoints with these as multipliers along (x, y).
            "orig_size": torch.tensor([w, h], dtype=torch.int64),
            "size": torch.tensor([w, h], dtype=torch.int64),
        }

    # ----------------------------------------------------------- load_item
    def load_item(self, idx):
        """Pre-transforms (PIL, target_dict) — needed by CocoEvaluator's
        `get_coco_api_from_dataset` to scan ground truths once at startup.

        Keypoints stay in 3D `(N, K, 3)` (interleaved x,y,v) here because
        EC's transforms (Resize / Mosaic / RandomZoomOut / …) operate on this
        layout. The flat `(N, 3K)` xyxyvv conversion happens AFTER transforms
        in __getitem__ for the model's consumption.
        """
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        target = self._ec_target(idx, img.width, img.height)
        return img, target

    # ----------------------------------------- 3D (N,K,3) → (N,3K) xyxyvv
    @staticmethod
    def _to_xyxyvv(kpts3d: torch.Tensor) -> torch.Tensor:
        """`(N, K, 3)` interleaved (x,y,v) → `(N, 3K)` flat as xyxyvv.

        Layout: [x0, y0, x1, y1, …, xK-1, yK-1, v0, v1, …, vK-1]. This is what
        prepare_for_cdn / DETRPoseHungarianMatcher / DETRPoseCriterion expect
        (`Z = kp[:, 0:2K]`, `V = kp[:, 2K:3K]`).
        """
        n = kpts3d.shape[0]
        if n == 0:
            return kpts3d.new_zeros((0, kpts3d.shape[1] * kpts3d.shape[2] if kpts3d.ndim == 3 else 0))
        xy = kpts3d[..., :2].reshape(n, -1)  # x0,y0,x1,y1,...
        v = kpts3d[..., 2]                    # v0,v1,...
        return torch.cat([xy, v], dim=-1)

    # ------------------------------------------------------------- __getitem__
    def __getitem__(self, idx):
        if self._ec_format:
            img, target = self.load_item(idx)
            if self._transforms is not None:
                if hasattr(self._transforms, "set_epoch"):
                    self._transforms.set_epoch(self.epoch)
                img, target = self._transforms(img, target)
            # Sanity: torchvision v2 transforms that drop boxes (RandomIoUCrop,
            # SanitizeBoundingBoxes, …) don't know about our `keypoints`
            # field, so they break the row alignment. Detect and fail loud.
            kpts3d = target.get("keypoints")
            n_box = target["boxes"].shape[0]
            if kpts3d is not None and kpts3d.shape[0] != n_box:
                raise RuntimeError(
                    f"YoloPoseDataset: post-transform target has "
                    f"{n_box} boxes vs {kpts3d.shape[0]} keypoint rows. "
                    "A geometric transform dropped boxes without filtering "
                    "the custom `keypoints` field. Remove RandomIoUCrop / "
                    "SanitizeBoundingBoxes / RandomZoomOut / RandomHorizontalFlip "
                    "from this YAML's transforms (see ecpose_s_pallet.yaml)."
                )
            # Final layout conversion expected by EC pose model & criterion.
            # Normalize xy to [0, 1] using the post-transform canvas size
            # (transforms have already adjusted the image to its final size).
            if kpts3d is not None and kpts3d.ndim == 3:
                # Image is now a tensor (after ConvertPILImage) shape (C, H, W).
                if isinstance(img, torch.Tensor) and img.ndim == 3:
                    H, W = img.shape[-2], img.shape[-1]
                else:
                    W, H = (img.size if hasattr(img, "size")
                            else (target["boxes"].canvas_size[1],
                                  target["boxes"].canvas_size[0]))
                kpts3d = kpts3d.clone()
                kpts3d[..., 0] /= max(W, 1)
                kpts3d[..., 1] /= max(H, 1)
                target["keypoints"] = self._to_xyxyvv(kpts3d)
            return img, target

        # Legacy uint8-CHW path used by --simple Trainer.
        if self._ram_cache is not None and idx in self._ram_cache:
            img = self._ram_cache[idx]
        else:
            img_path = self.img_files[idx]
            arr = cv2.imread(img_path)
            if arr is None:
                arr = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            if arr.shape[0] != self.img_size or arr.shape[1] != self.img_size:
                arr = cv2.resize(arr, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_LINEAR)
            arr = arr[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(arr)
            if self._ram_cache is not None:
                self._ram_cache[idx] = img

        img_tensor = torch.from_numpy(img)
        labels_out = torch.from_numpy(self.cached_labels[idx]).float()
        return img_tensor, labels_out

    # ----------------------------------------------------- legacy collate_fn
    @staticmethod
    def collate_fn(batch):
        """Used only by the legacy / --simple path. EC mode uses
        BatchImageCollateFunction registered in navidet.data.dataloader."""
        imgs, labels = zip(*batch)
        imgs = torch.stack(imgs, 0)
        new_labels = []
        for i, label in enumerate(labels):
            if label.shape[0] > 0:
                l = label.clone()
                l[:, 0] = i
                new_labels.append(l)
        labels = torch.cat(new_labels, 0) if new_labels \
            else torch.zeros((0, 6 + 4 * 3))
        return imgs, labels
