"""
공통 epoch 리포터 — train.py / train_distill.py가 동일한 로깅·저장을 쓰도록.

매 epoch:
  - 검증 pose 지표(회전°·translation mm·width mm·recall) 평가 + 로그
  - history.json 기록, curves.png 갱신(save_interval)
  - save_interval마다 epoch_{i}/model.pt + last.pt + val_viz(추론) 저장, GT는 1회
  - pose 종합점수 기준 best.pt 저장
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

from .infer import render_gt, render_sample
from .metrics import evaluate_pose, format_metrics
from ..utils.visualize import plot_history


class Progress:
    """epoch 내 iteration 진행률을 한 줄로 갱신 표시 (멈춤 오인 방지)."""

    def __init__(self, epoch, total_epochs, n_total, prefix=""):
        self.ep, self.te, self.nt = epoch, total_epochs, n_total
        self.prefix = prefix
        self.t0 = time.time()
        self.every = max(1, n_total // 100)

    def step(self, it, **extras):
        if it % self.every and it != self.nt - 1:
            return
        done = it + 1
        sp = done / max(time.time() - self.t0, 1e-6)
        eta = (self.nt - done) / max(sp, 1e-6)
        ex = " ".join(f"{k}={v:.3f}" for k, v in extras.items())
        sys.stdout.write(f"\r  ep {self.ep+1}/{self.te} {self.prefix}{done}/{self.nt} "
                         f"({100*done/self.nt:3.0f}%) {sp:.1f}it/s ETA{eta:5.0f}s {ex}   ")
        sys.stdout.flush()

    def close(self):
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.flush()


class EpochReporter:
    def __init__(self, out, total_epochs, save_interval, val_ds=None, viz_idx=None,
                 class_names=None, conf=0.25, iou=0.5, task="6dof", viz_conf=None):
        self.out = out
        self.total = total_epochs
        self.save_interval = save_interval
        self.val_ds = val_ds
        self.viz_idx = viz_idx or []
        self.names = list(class_names or [])
        self.conf, self.iou = conf, iou
        # 시각화 전용 임계값: 정량평가(self.conf)와 분리 — 초기 epoch에도 약한 예측을 그려
        # 학습 진행을 눈으로 확인할 수 있게 한다. 미지정 시 conf와 동일.
        self.viz_conf = conf if viz_conf is None else viz_conf
        self.task = task                       # "6dof" → pose 지표 평가/시각화, 그 외 → 손실 로깅만
        self.history = []
        self.best = float("inf")
        os.makedirs(out, exist_ok=True)

    def step(self, ep, lr, model, train_stats, make_ckpt, prefix=""):
        """ep(0-base), lr, model, train_stats(손실 dict), make_ckpt(콜백), prefix(추가 로그)."""
        # task별로 표시할 손실 키만 골라 로그 구성 (없는 키는 건너뜀)
        keys = (["box", "cls", "rot", "size", "depth", "trans"] if self.task == "6dof"
                else ["box", "dfl", "cls", "kpt", "vis"])
        parts = " ".join(f"{k}={train_stats[k]:.3f}" for k in keys if k in train_stats)
        log = (f"[{ep+1}/{self.total}] {prefix}lr={lr:.2e} "
               f"total={train_stats['total']:.3f} | {parts}")

        # 6DoF만: 검증 pose 지표(회전°·translation mm 등) 평가. 2D task는 손실 곡선만.
        val = None
        score = train_stats["total"]
        if self.task == "6dof" and self.val_ds is not None:
            val = evaluate_pose(model, self.val_ds, conf=self.conf, iou=self.iou)
            log += " || val " + format_metrics(val)
            if not math.isnan(val["rot_deg"]):
                score = val["rot_deg"] + val["trans_mm"] / 100.0
        print(log)

        # history.json (매 epoch)
        self.history.append({"epoch": ep + 1, "lr": lr, "train": train_stats, "val": val})
        with open(os.path.join(self.out, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        # save_interval(+마지막): 모델 스냅샷 + 곡선 + 검증 시각화
        last_ep = ep == self.total - 1
        if (ep + 1) % self.save_interval == 0 or last_ep:
            ep_dir = os.path.join(self.out, f"epoch_{ep+1:03d}")
            os.makedirs(ep_dir, exist_ok=True)
            ckpt = make_ckpt()
            torch.save(ckpt, os.path.join(ep_dir, "model.pt"))
            torch.save(ckpt, os.path.join(self.out, "last.pt"))
            try:
                plot_history(self.history, os.path.join(self.out, "curves.png"))
            except Exception as e:
                print(f"  [warn] plot 실패: {e}")
            if self.viz_idx and self.task == "6dof":            # 6DoF 전용 시각화
                gt_dir = os.path.join(self.out, "val_viz_gt")
                if not os.path.isdir(gt_dir):                       # GT는 1회만
                    os.makedirs(gt_dir, exist_ok=True)
                    for i in self.viz_idx:
                        s = self.val_ds[i]
                        render_gt(s, self.names).save(os.path.join(gt_dir, f"{s['stem']}.png"))
                vdir = os.path.join(ep_dir, "val_viz")             # 추론 결과만
                os.makedirs(vdir, exist_ok=True)
                for i in self.viz_idx:
                    s = self.val_ds[i]
                    pil, _ = render_sample(model, s, self.names, draw_gt=False,
                                           conf=self.viz_conf, iou=self.iou)
                    pil.save(os.path.join(vdir, f"{s['stem']}.png"))
            print(f"  saved: {ep_dir}/model.pt" + (f" + val_viz({len(self.viz_idx)})" if self.viz_idx else ""))

        # best (pose 종합점수, 없으면 train total)
        if score < self.best:
            self.best = score
            torch.save(make_ckpt(), os.path.join(self.out, "best.pt"))
        return self.best
