"""
navidet 학습 (config 기반). model.task로 6DoF / 2D 멀티태스크(pose)를 선택한다.

사용 예
-------
  python -m navidet.tools.train --config navidet/config/default_6dof.yaml   # 6DoF
  python -m navidet.tools.train --config navidet/config/default_pose.yaml   # pose
  python -m navidet.tools.train --set train.epochs=50 data.imgsz=512
"""

from __future__ import annotations

import argparse
import math
import os

import torch
import yaml
from torch.utils.data import DataLoader

from navidet.core.model import YOLO6DoF, YOLOPose, TASK_PRESET
from navidet.module.dataset import PoseDataset, collate_fn
from navidet.module.loss import Pose6DoFLoss
from navidet.module.loss_mt import MultiTaskLoss
from navidet.module.trainer import EpochReporter, Progress
from navidet.utils.config import load_config


# task별 epoch 손실 집계 키
LOSS_KEYS = {
    "6dof": ["box", "dfl", "obj", "cls", "rot", "size", "depth", "trans", "total"],
    "pose": ["box", "dfl", "cls", "seg", "kpt", "vis", "total"],
}


def move_targets(targets, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in targets.items()}


def run_epoch(model, loss_fn, loader, device, optimizer=None, scaler=None, amp=False,
              epoch=0, total_epochs=1, task="6dof"):
    train = optimizer is not None
    model.train(train)
    model.head.return_raw = not train          # eval에서도 loss용 raw dict
    keys = LOSS_KEYS.get(task, LOSS_KEYS["pose"])
    agg = {k: 0.0 for k in keys}
    n = n_skip = 0
    prog = Progress(epoch, total_epochs, len(loader)) if train else None
    for it, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = move_targets(targets, device)
        with torch.set_grad_enabled(train), torch.autocast(
                device_type=device.type, enabled=amp):
            out = model(imgs)
            loss, items = loss_fn(out, targets)

        if train:
            if not torch.isfinite(loss):       # NaN/Inf 가드
                optimizer.zero_grad(set_to_none=True)
                n_skip += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

        for k, v in items.items():
            if k in agg:
                agg[k] += v
        n += 1
        if prog:
            prog.step(it, total=agg["total"] / n)
    if prog:
        prog.close()
    if n_skip:
        print(f"  [warn] {n_skip} non-finite batch(es) skipped")
    return {k: v / max(n, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML 설정 경로 (기본 default.yaml)")
    ap.add_argument("--set", nargs="*", default=[], help="override 예: train.lr=1e-3")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    d, m, tr_cfg = cfg.data, cfg.model, cfg.train
    task = m.get("task", "6dof")
    is_2d = task in TASK_PRESET            # detect/segment/pose

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(tr_cfg.out, exist_ok=True)

    # 최종(override 반영된) config를 화면 표시 + runs 폴더에 복사 저장
    cfg_dump = yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False)
    print("=" * 60)
    print(f"device : {device}")
    print(f"config (saved to {tr_cfg.out}/config.yaml):")
    print("-" * 60)
    print(cfg_dump.rstrip())
    print("=" * 60)
    with open(os.path.join(tr_cfg.out, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(cfg_dump)

    kpt_shape = tuple(m.get("kpt_shape", (4, 3)))
    ds_kw = dict(ini=d.ini, imgsz=d.imgsz, depth_scale=d.depth_scale,
                 rmse_thresh=d.rmse_thresh, limit=tr_cfg.limit,
                 pose_mode=d.get("pose_mode", "full"), task=task, kpt_shape=kpt_shape,
                 use_cache=d.get("use_cache", True))
    train_ds = PoseDataset(d.train_root, **ds_kw)
    train_loader = DataLoader(train_ds, batch_size=tr_cfg.batch, shuffle=True,
                              num_workers=tr_cfg.workers, collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    print(f"train frames: {len(train_ds)}")

    val_ds = None
    if d.get("val_root"):
        val_ds = PoseDataset(d.val_root, **ds_kw)
        print(f"val frames: {len(val_ds)}")

    # 검증 시각화할 고정 샘플 인덱스 (save_interval 시점마다 epoch 폴더에 저장)
    viz_num = tr_cfg.get("val_viz_num", 0) if val_ds is not None else 0
    viz_idx = list(range(min(viz_num, len(val_ds)))) if viz_num else []

    light_head = m.get("light_head", True)
    if is_2d:
        tasks = TASK_PRESET[task]
        nm = m.get("nm", 32)
        model = YOLOPose(nc=d.nc, scale=m.scale, tasks=tasks, nm=nm,
                         kpt_shape=kpt_shape).to(device)
        loss_fn = MultiTaskLoss(nc=d.nc, strides=model.head.strides, tasks=tasks,
                                kpt_shape=kpt_shape, nm=nm,
                                weights=cfg.loss.to_dict()).to(device)
    else:
        model = YOLO6DoF(nc=d.nc, scale=m.scale, rot_repr=m.rot_repr,
                         light_head=light_head).to(device)
        loss_fn = Pose6DoFLoss(nc=d.nc, rot_repr=m.rot_repr,
                               weights=cfg.loss.to_dict()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr_cfg.lr,
                                  weight_decay=tr_cfg.weight_decay)

    def lr_lambda(ep):
        warm = 3
        if ep < warm:
            return (ep + 1) / warm
        prog = (ep - warm) / max(1, tr_cfg.epochs - warm)
        return 0.5 * (1 + math.cos(math.pi * prog)) * 0.99 + 0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(enabled=tr_cfg.amp and device.type == "cuda")

    reporter = EpochReporter(tr_cfg.out, tr_cfg.epochs, tr_cfg.save_interval,
                             val_ds=val_ds, viz_idx=viz_idx, class_names=list(d.class_names),
                             conf=cfg.predict.conf, iou=cfg.predict.iou, task=task)
    meta = dict(nc=d.nc, scale=m.scale, rot_repr=m.rot_repr, imgsz=d.imgsz,
                light_head=light_head, task=task, kpt_shape=list(kpt_shape),
                nm=m.get("nm", 32), cfg=cfg.to_dict())
    best = float("inf")
    for ep in range(tr_cfg.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        tr = run_epoch(model, loss_fn, train_loader, device, optimizer,
                       scaler if tr_cfg.amp else None, amp=tr_cfg.amp,
                       epoch=ep, total_epochs=tr_cfg.epochs, task=task)
        scheduler.step()
        make_ckpt = lambda e=ep: {"model": model.state_dict(), "epoch": e, **meta}
        best = reporter.step(ep, lr_now, model, tr, make_ckpt)

    print(f"done. best={best:.3f}  ckpt={tr_cfg.out}  curves={tr_cfg.out}/curves.png")


if __name__ == "__main__":
    main()
