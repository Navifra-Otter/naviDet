"""
DINOv3 → Student 지식 증류 학습. model.task로 6DoF / 2D 멀티태스크(pose)를 선택한다.

Total Loss = α(epoch)·Distillation + β(epoch)·Task
  - Distillation: Student neck 특징을 FeatureProjector로 DINOv3 차원에 매핑한 뒤,
                  DINOv3의 '순수 patch 공간 특징'(register/cls 제거)과 정렬 (cosine/MSE)
                  → backbone/neck 공유 구조라 task와 무관하게 동일하게 동작.
  - Task        : task="6dof" → Pose6DoFLoss(Depth + 6D Head)
                  task in {pose,detect,segment} → MultiTaskLoss

사용 예
-------
  # 6DoF 증류 (기본 config)
  python -m navidet.tools.train_distill --config navidet/config/default_6dof.yaml
  # 2D pose 증류
  python -m navidet.tools.train_distill --config navidet/config/default_pose.yaml
  # teacher_ckpt 비우면 Mock teacher로 파이프라인 스모크
  python -m navidet.tools.train_distill --set train.epochs=2 train.limit=8
"""

from __future__ import annotations

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from navidet.core.model import YOLO6DoF, YOLOPose, TASK_PRESET
from navidet.module.dataset import PoseDataset, collate_fn
from navidet.module.distill import (DINOv3Teacher, FeatureProjector, MockDINOv3,
                                     feature_distillation_loss, loss_weights)
from navidet.module.loss import Pose6DoFLoss
from navidet.module.loss_mt import MultiTaskLoss
from navidet.module.trainer import EpochReporter, Progress
from navidet.utils.config import load_config


# task별 epoch 손실 집계 키 (train.py와 동일)
LOSS_KEYS = {
    "6dof": ["box", "dfl", "obj", "cls", "rot", "size", "depth", "trans", "total"],
    "pose": ["box", "dfl", "cls", "seg", "kpt", "vis", "total"],
}


# --------------------------------------------------------------------------- #
#  1. Teacher 빌드 (frozen)
# --------------------------------------------------------------------------- #
def build_teacher(dc, device):
    """미세조정된 DINOv3 로드. ckpt가 없으면 MockDINOv3로 대체(스모크)."""
    ckpt = dc.get("teacher_ckpt", "")
    if ckpt:
        # 실제 DINOv3 로드 예시 (환경에 맞게 교체):
        #   backbone = torch.hub.load('facebookresearch/dinov3', 'dinov3_vitb16')
        #   backbone.load_state_dict(torch.load(ckpt))   # 태스크 미세조정 가중치
        backbone = torch.hub.load("facebookresearch/dinov3",
                                  dc.get("hub_name", "dinov3_vits16"))
        state = torch.load(ckpt, map_location="cpu")
        backbone.load_state_dict(state.get("model", state), strict=False)
        print(f"[teacher] DINOv3 로드: {ckpt}")
    else:
        backbone = MockDINOv3(dc.teacher_dim, dc.patch_size, dc.num_register_tokens)
        print("[teacher] teacher_ckpt 미지정 → MockDINOv3 (스모크용)")

    teacher = DINOv3Teacher(
        backbone, embed_dim=dc.teacher_dim, patch_size=dc.patch_size,
        num_register_tokens=dc.num_register_tokens, img_size=dc.teacher_imgsz,
    ).to(device)
    teacher.eval()                                         # ★ 항상 eval 고정
    return teacher


def move_targets(t, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}


# --------------------------------------------------------------------------- #
#  전체 학습 루프 (한 epoch)
# --------------------------------------------------------------------------- #
def train_one_epoch(student, projector, teacher, loss_fn, loader, optimizer,
                    device, epoch, total_epochs, dc, scaler=None, amp=False,
                    task="6dof"):
    student.train()
    projector.train()
    teacher.eval()                                         # ★ Teacher는 항상 eval
    is_6dof = task == "6dof"

    # 4. Decoupled scheduler — epoch에 따른 α(distill) / β(task)
    alpha, beta = loss_weights(epoch, total_epochs, dc.alpha0, dc.beta0, dc.beta1, dc.sched)

    # 일반 학습과 동일한 task 손실 항목 + distill 집계
    keys = LOSS_KEYS.get(task, LOSS_KEYS["pose"])
    agg = {k: 0.0 for k in keys}
    agg["distill"] = 0.0
    n = 0
    prog = Progress(epoch, total_epochs, len(loader), prefix=f"α={alpha:.2f} ")
    for it, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = move_targets(targets, device)

        with torch.autocast(device_type=device.type, enabled=amp):
            # --- Student forward (서브모듈 직접 호출로 neck 특징 확보) ---
            feats = student.backbone(imgs)
            n3, n4, n5 = student.neck(feats)               # 'Neck / Feature Fusion' 출력
            head_out = student.head((n3, n4, n5))          # train: raw dict
            if is_6dof:
                depth = student.depth_head(n3)             # Depth Head
                out = {"det": head_out, "depth": depth}
            else:
                out = head_out                             # MultiTaskHead raw dict

            # --- Task Loss ---
            task_loss, items = loss_fn(out, targets)

            # --- Distillation Loss ---
            # 2. Projector: neck 특징(n3) → DINOv3 차원
            student_proj = projector(n3)                   # [B, D, h, w]
            # 1. Teacher: frozen, no_grad / 3. register 제거된 순수 patch 공간맵
            with torch.no_grad():
                teacher_feat = teacher.extract_patch_tokens(imgs)   # [B, D, Hp, Wp]
            distill_loss = feature_distillation_loss(
                student_proj.float(), teacher_feat.float(), dc.loss, dc.align)

            # --- Decoupled total ---
            total = alpha * distill_loss + beta * task_loss

        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            continue
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(projector.parameters()), 10.0)
            scaler.step(optimizer); scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(projector.parameters()), 10.0)
            optimizer.step()

        for k, v in items.items():                     # task 손실 항목 (total=task total)
            if k in agg:
                agg[k] += v
        agg["distill"] += float(distill_loss.detach())
        n += 1
        prog.step(it, total=agg["total"] / n, distill=agg["distill"] / n)
    prog.close()

    stats = {k: v / max(n, 1) for k, v in agg.items()}
    return stats, alpha, beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    d, m, tr, dc = cfg.data, cfg.model, cfg.train, cfg.distill
    task = m.get("task", "6dof")
    is_6dof = task == "6dof"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(tr.out, exist_ok=True)

    # config 화면 표시 + runs 폴더에 복사 저장 (일반 학습과 동일)
    cfg_dump = yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False)
    print("=" * 60)
    print(f"device : {device}  (distill: teacher_dim={dc.teacher_dim})")
    print(f"config (saved to {tr.out}/config.yaml):")
    print("-" * 60)
    print(cfg_dump.rstrip())
    print("=" * 60)
    with open(os.path.join(tr.out, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(cfg_dump)

    # 데이터 (task에 따라 6DoF GT 또는 2D 멀티태스크 GT 생성)
    kpt_shape = tuple(m.get("kpt_shape", (4, 3)))
    ds_kw = dict(ini=d.ini, imgsz=d.imgsz, depth_scale=d.depth_scale,
                 rmse_thresh=d.rmse_thresh, pose_mode=d.get("pose_mode", "full"),
                 task=task, kpt_shape=kpt_shape, use_cache=d.get("use_cache", True))
    train_ds = PoseDataset(d.train_root, limit=tr.limit, **ds_kw)
    train_loader = DataLoader(train_ds, batch_size=tr.batch, shuffle=True,
                              num_workers=tr.workers, collate_fn=collate_fn,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    print(f"train frames: {len(train_ds)}")

    # 검증셋 (6DoF만 pose 지표 평가, 2D task는 손실 곡선만)
    val_ds = PoseDataset(d.val_root, **ds_kw) if d.get("val_root") else None
    viz_num = tr.get("val_viz_num", 0) if val_ds is not None else 0
    viz_idx = list(range(min(viz_num, len(val_ds)))) if viz_num else []

    # 모델: Student + Projector + Teacher
    light_head = m.get("light_head", True)
    nm = m.get("nm", 32)
    if is_6dof:
        student = YOLO6DoF(nc=d.nc, scale=m.scale, rot_repr=m.rot_repr,
                           light_head=light_head).to(device)
        loss_fn = Pose6DoFLoss(nc=d.nc, rot_repr=m.rot_repr,
                               weights=cfg.loss.to_dict()).to(device)
    else:
        tasks = TASK_PRESET[task]
        student = YOLOPose(nc=d.nc, scale=m.scale, tasks=tasks, nm=nm,
                           kpt_shape=kpt_shape).to(device)
        loss_fn = MultiTaskLoss(nc=d.nc, strides=student.head.strides, tasks=tasks,
                                kpt_shape=kpt_shape, nm=nm,
                                weights=cfg.loss.to_dict()).to(device)
    # Projector는 neck n3 채널 → teacher 차원 (구조 공유라 task 무관)
    projector = FeatureProjector(student.neck.out_channels[0], dc.teacher_dim).to(device)
    teacher = build_teacher(dc, device)
    # Student + Projector 만 최적화 (Teacher 제외)
    optimizer = torch.optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=tr.lr, weight_decay=tr.weight_decay)
    scaler = torch.amp.GradScaler(enabled=tr.amp and device.type == "cuda")

    # 일반 학습과 동일한 로깅·저장 (6DoF만 pose 지표/val_viz, 2D는 손실 곡선만)
    reporter = EpochReporter(tr.out, tr.epochs, tr.save_interval, val_ds=val_ds,
                             viz_idx=viz_idx, class_names=list(d.class_names),
                             conf=cfg.predict.conf, iou=cfg.predict.iou, task=task,
                             viz_conf=cfg.predict.get("viz_conf", cfg.predict.conf))
    meta = dict(nc=d.nc, scale=m.scale, rot_repr=m.get("rot_repr", "6d"), imgsz=d.imgsz,
                light_head=light_head, task=task, kpt_shape=list(kpt_shape),
                nm=nm, cfg=cfg.to_dict())
    best = float("inf")
    for ep in range(tr.epochs):
        lr_now = optimizer.param_groups[0]["lr"]
        stats, a, b = train_one_epoch(student, projector, teacher, loss_fn,
                                      train_loader, optimizer, device, ep, tr.epochs,
                                      dc, scaler if tr.amp else None, amp=tr.amp,
                                      task=task)
        # distill 전용 정보(α/β/distill)는 prefix로, 나머지(손실·pose지표·저장)는 동일
        prefix = f"α={a:.3f} β={b:.3f} distill={stats['distill']:.4f} "
        make_ckpt = lambda e=ep: {"model": student.state_dict(),
                                  "projector": projector.state_dict(), "epoch": e, **meta}
        best = reporter.step(ep, lr_now, student, stats, make_ckpt, prefix=prefix)

    print(f"done. best={best:.3f}  ckpt={tr.out}  curves={tr.out}/curves.png")


if __name__ == "__main__":
    main()
