"""
COCO keypoint mAP evaluation.

Replicates the metric used in EdgeCrafter/ecpose
(engine/data/dataset/coco_eval.py + engine/solver/pose_engine.py::evaluate),
which runs faster_coco_eval's COCOeval_faster on keypoints (OKS-based AP/AR).

The user's repo stores labels in YOLO format (normalized cls + xywh + kpts),
so we build an in-memory COCO GT json from the dataset's cached labels and
decode the model's raw head output into COCO-style predictions.
"""

import contextlib
import copy
import os

import numpy as np
import torch


def _build_coco_gt(dataset, ncls, nkpts, img_size, sigmas=None):
    """Build a COCO GT object from a YoloPoseDataset.

    All coordinates are emitted in the input-resolution pixel space
    (img_size x img_size), which matches the space the model decodes into.
    """
    from faster_coco_eval import COCO

    if sigmas is None:
        sigmas = [0.05] * nkpts

    images, annotations, categories = [], [], []
    kp_names = [f"kp{i}" for i in range(nkpts)]
    skeleton = []

    for cls_id in range(ncls):
        categories.append({
            "id": int(cls_id),
            "name": f"class_{cls_id}",
            "supercategory": "object",
            "keypoints": kp_names,
            "skeleton": skeleton,
        })

    ann_id = 1
    for idx in range(len(dataset)):
        image_id = idx
        images.append({
            "id": image_id,
            "file_name": os.path.basename(dataset.img_files[idx]),
            "width": img_size,
            "height": img_size,
        })

        targets = dataset.cached_labels[idx]
        if targets is None or len(targets) == 0:
            continue

        for row in targets:
            cls = int(row[1])
            cx, cy, w, h = row[2:6].tolist()
            cx, cy, w, h = cx * img_size, cy * img_size, w * img_size, h * img_size
            x = cx - w / 2.0
            y = cy - h / 2.0
            area = float(max(w * h, 1.0))

            kpt_flat = row[6:6 + nkpts * 3]
            kpts_xyv = []
            num_kpts = 0
            for k in range(nkpts):
                kx = float(kpt_flat[k * 3]) * img_size
                ky = float(kpt_flat[k * 3 + 1]) * img_size
                kv = int(kpt_flat[k * 3 + 2])
                if kv > 0:
                    num_kpts += 1
                kpts_xyv.extend([kx, ky, kv])

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": area,
                "iscrowd": 0,
                "num_keypoints": num_kpts,
                "keypoints": kpts_xyv,
            })
            ann_id += 1

    coco_dict = {
        "info": {"description": "yolo->coco GT (in-memory)"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    coco = COCO()
    coco.dataset = coco_dict
    coco.createIndex()

    for cat in coco.dataset["categories"]:
        cat["sigmas"] = list(sigmas)

    return coco


def _decode_pose_predictions(raw_out, ncls, kpt_shape, num_select=100):
    """Convert PoseHead inference output to per-image COCO predictions.

    raw_out: (B, ncls + nkpts*ndim, A) — first ncls channels are sigmoid'd
             class logits, the remaining are decoded keypoints (x, y[, v])
             in input-resolution pixel space.

    Returns a list (length B) of dicts with keys:
        scores:    (num_select,)
        labels:    (num_select,)            COCO category_id
        keypoints: (num_select, nkpts, 3)   (x, y, v) in input-pixel space
    """
    nkpts, ndim = kpt_shape
    B, C, A = raw_out.shape
    assert C == ncls + nkpts * ndim, f"unexpected channels {C} vs {ncls + nkpts*ndim}"

    cls_scores = raw_out[:, :ncls, :]                 # (B, ncls, A)
    kpt_pred = raw_out[:, ncls:, :]                   # (B, nkpts*ndim, A)
    kpt_pred = kpt_pred.view(B, nkpts, ndim, A)       # (B, nkpts, ndim, A)

    flat = cls_scores.reshape(B, ncls * A)
    k = min(num_select, flat.shape[1])
    topk_vals, topk_idx = torch.topk(flat, k, dim=1)  # (B, k)
    cls_ids = topk_idx % ncls
    anchor_ids = topk_idx // ncls

    results = []
    for b in range(B):
        a_idx = anchor_ids[b]                                       # (k,)
        kpts_b = kpt_pred[b].permute(2, 0, 1)                       # (A, nkpts, ndim)
        sel = kpts_b.index_select(0, a_idx)                         # (k, nkpts, ndim)
        if ndim == 2:
            vis = torch.ones_like(sel[..., :1])
            sel = torch.cat([sel, vis], dim=-1)
        results.append({
            "scores": topk_vals[b].detach().cpu(),
            "labels": cls_ids[b].detach().cpu(),
            "keypoints": sel.detach().cpu(),
        })
    return results


class CocoKeypointEvaluator:
    """Mirror of ecpose CocoEvaluator (single-process slice)."""

    def __init__(self, dataset, ncls, kpt_shape, img_size,
                 num_select=100, sigmas=None, score_thresh=0.0):
        from faster_coco_eval import COCOeval_faster

        self.ncls = ncls
        self.kpt_shape = kpt_shape          # (nkpts, ndim)
        self.img_size = img_size
        self.num_select = num_select
        self.score_thresh = score_thresh

        self.coco_gt = _build_coco_gt(
            dataset, ncls=ncls, nkpts=kpt_shape[0],
            img_size=img_size, sigmas=sigmas,
        )
        self._COCOeval_faster = COCOeval_faster
        self._reset()

    def _reset(self):
        self.coco_eval = self._COCOeval_faster(
            self.coco_gt, iouType="keypoints",
            print_function=print, separate_eval=True,
        )
        self.coco_eval.params.kpt_oks_sigmas = np.array(
            self.coco_gt.dataset["categories"][0]["sigmas"], dtype=np.float64
        )
        self.img_ids = []
        self.eval_imgs = []
        self._results = []

    def reset(self):
        self._reset()

    @torch.no_grad()
    def update_from_raw(self, raw_out, image_ids):
        """raw_out: model head output (B, C, A). image_ids: list[int] length B."""
        per_image = _decode_pose_predictions(
            raw_out, self.ncls, self.kpt_shape, num_select=self.num_select,
        )
        coco_preds = []
        for img_id, pred in zip(image_ids, per_image):
            scores = pred["scores"]
            labels = pred["labels"]
            kpts = pred["keypoints"]                      # (k, nkpts, 3)
            mask = scores > self.score_thresh
            if mask.any():
                scores = scores[mask]
                labels = labels[mask]
                kpts = kpts[mask]
            for s, l, k in zip(scores.tolist(), labels.tolist(), kpts):
                coco_preds.append({
                    "image_id": int(img_id),
                    "category_id": int(l),
                    "keypoints": k.flatten().tolist(),
                    "score": float(s),
                })
        if coco_preds:
            self._results.extend(coco_preds)
            self.img_ids.extend([int(i) for i in image_ids])

    def summarize(self):
        """Run full COCO keypoint accumulate/summarize. Returns dict of metrics."""
        if len(self._results) == 0:
            return {f"AP": 0.0, f"AP50": 0.0, f"AP75": 0.0, f"AR": 0.0}

        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_dt = self.coco_gt.loadRes(self._results)

        coco_eval = self._COCOeval_faster(
            self.coco_gt, coco_dt, iouType="keypoints",
            print_function=print, separate_eval=False,
        )
        coco_eval.params.kpt_oks_sigmas = self.coco_eval.params.kpt_oks_sigmas
        coco_eval.params.imgIds = sorted(set(self.img_ids))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats.tolist()
        # COCO keypoints stat order:
        # [AP, AP50, AP75, AP_M, AP_L, AR, AR50, AR75, AR_M, AR_L]
        keys = ["AP", "AP50", "AP75", "AP_M", "AP_L",
                "AR", "AR50", "AR75", "AR_M", "AR_L"]
        return {k: float(v) for k, v in zip(keys, stats)}
