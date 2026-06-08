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
from navidet.tools.train import resolve_device_ids
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
    """DINOv3 로드. teacher_ckpt가 없으면 MockDINOv3로 대체(스모크).

    DINOv3 사전학습 가중치는 라이선스 게이트라 torch.hub의 자동 다운로드가
    HTTP 403으로 막힌다. → Meta에서 미리 받은 .pth 경로를 teacher_ckpt로 주고
    weights= 로 직접 넘겨 네트워크 접근 없이 로드한다.
    repo_dir 를 지정하면 캐시된 로컬 레포(source='local')에서 빌드해 GitHub 접근도 건너뛴다.
    """
    ckpt = dc.get("teacher_ckpt", "")
    if ckpt:
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"[teacher] teacher_ckpt 파일 없음: {ckpt}\n"
                "  DINOv3 가중치는 자동 다운로드가 막혀(403) 직접 받아야 합니다.\n"
                "  https://github.com/facebookresearch/dinov3 에서 라이선스 동의 후\n"
                f"  '{dc.get('hub_name', 'dinov3_vits16')}' 사전학습 .pth 를 받아 경로를 지정하세요.")
        hub_name = dc.get("hub_name", "dinov3_vits16")
        repo_dir = dc.get("repo_dir", "")            # 비우면 GitHub(캐시), 지정 시 로컬 레포
        load_kw = dict(source="local") if repo_dir else {}
        backbone = torch.hub.load(repo_dir or "facebookresearch/dinov3",
                                  hub_name, weights=ckpt, **load_kw)
        print(f"[teacher] DINOv3({hub_name}) 가중치 로드: {ckpt}")
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
#  멀티 GPU용 스텝 래퍼 — student forward+projector+teacher+손실을 한 forward로 묶어
#  DataParallel이 배치를 GPU별로 분할/gather 할 수 있게 한다. (distill은 forward를
#  서브모듈로 쪼개 호출하므로 모듈로 감싸지 않으면 DataParallel을 적용할 수 없다.)
#  · 스칼라 손실은 [1] 형태로 반환 → gather가 GPU별 값을 [n_gpu]로 모음(0-dim 불가).
#  · autocast를 forward 안에 둬 DataParallel worker 스레드에서도 적용되게 한다.
# --------------------------------------------------------------------------- #
class DistillStep(torch.nn.Module):
    def __init__(self, student, projector, teacher, loss_fn, is_6dof, dc, amp):
        super().__init__()
        self.student = student
        self.projector = projector
        self.teacher = teacher
        self.loss_fn = loss_fn
        self.is_6dof = is_6dof
        self.dc = dc
        self.amp = amp

    def forward(self, imgs, targets, alpha, beta):
        with torch.autocast(device_type=imgs.device.type, enabled=self.amp):
            # --- Student forward (서브모듈 직접 호출로 neck 특징 확보) ---
            feats = self.student.backbone(imgs)
            n3, n4, n5 = self.student.neck(feats)          # 'Neck / Feature Fusion' 출력
            head_out = self.student.head((n3, n4, n5))     # train: raw dict
            if self.is_6dof:
                depth = self.student.depth_head(n3)        # Depth Head
                out = {"det": head_out, "depth": depth}
            else:
                out = head_out                             # MultiTaskHead raw dict

            # --- Task Loss ---
            task_loss, items = self.loss_fn(out, targets)

            # --- Distillation Loss ---
            student_proj = self.projector(n3)              # [B, D, h, w]
            with torch.no_grad():                          # Teacher: frozen, register 제거 patch맵
                teacher_feat = self.teacher.extract_patch_tokens(imgs)
            distill_loss = feature_distillation_loss(
                student_proj.float(), teacher_feat.float(), self.dc.loss, self.dc.align)

            total = alpha * distill_loss + beta * task_loss   # Decoupled total

        # gather가 가능하도록 모든 스칼라를 [1]로. items(파이썬 float)도 텐서화.
        ret = {"__loss__": total.reshape(1),
               "distill": distill_loss.detach().reshape(1)}
        for k, v in items.items():
            ret[k] = torch.as_tensor(float(v), device=imgs.device).reshape(1)
        return ret


# --------------------------------------------------------------------------- #
#  전체 학습 루프 (한 epoch)
# --------------------------------------------------------------------------- #
def train_one_epoch(step_module, loader, optimizer, device, epoch, total_epochs,
                    dc, scaler=None, amp=False, task="6dof"):
    step_module.train()
    core = (step_module.module if isinstance(step_module, torch.nn.DataParallel)
            else step_module)
    core.teacher.eval()                                    # ★ Teacher는 항상 eval
    clip_params = list(core.student.parameters()) + list(core.projector.parameters())

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

        ret = step_module(imgs, targets, alpha, beta)      # DataParallel이면 GPU별 분산
        total = ret["__loss__"].mean()                     # GPU별 평균 손실 → 전체 평균

        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            continue
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clip_params, 10.0)
            scaler.step(optimizer); scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(clip_params, 10.0)
            optimizer.step()

        for k in agg:                                      # task 손실 항목 (total=task total)
            if k in ret:
                agg[k] += ret[k].mean().item()
        agg["distill"] += ret["distill"].mean().item()
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

    device_ids = resolve_device_ids(tr.get("gpus", "auto"))
    device = torch.device(f"cuda:{device_ids[0]}") if device_ids else torch.device("cpu")
    os.makedirs(tr.out, exist_ok=True)

    # config 화면 표시 + runs 폴더에 복사 저장 (일반 학습과 동일)
    cfg_dump = yaml.safe_dump(cfg.to_dict(), allow_unicode=True, sort_keys=False)
    print("=" * 60)
    if len(device_ids) > 1:
        print(f"device : multi-GPU DataParallel {device_ids} (primary {device})"
              f"  (distill: teacher_dim={dc.teacher_dim})")
    else:
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

    # 멀티 GPU: student+projector+teacher+손실을 한 모듈로 묶어 DataParallel로 분산.
    # reporter/시각화/checkpoint는 원본 student를 그대로 쓴다.
    step_module = DistillStep(student, projector, teacher, loss_fn, is_6dof, dc, tr.amp)
    if len(device_ids) > 1:
        step_module = torch.nn.DataParallel(step_module, device_ids=device_ids)

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
        stats, a, b = train_one_epoch(step_module, train_loader, optimizer, device,
                                      ep, tr.epochs, dc, scaler if tr.amp else None,
                                      amp=tr.amp, task=task)
        # distill 전용 정보(α/β/distill)는 prefix로, 나머지(손실·pose지표·저장)는 동일
        prefix = f"α={a:.3f} β={b:.3f} distill={stats['distill']:.4f} "
        make_ckpt = lambda e=ep: {"model": student.state_dict(),
                                  "projector": projector.state_dict(), "epoch": e, **meta}
        best = reporter.step(ep, lr_now, student, stats, make_ckpt, prefix=prefix)

    print(f"done. best={best:.3f}  ckpt={tr.out}  curves={tr.out}/curves.png")


if __name__ == "__main__":
    main()
