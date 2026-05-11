# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable

import torch
from tqdm.auto import tqdm

from navidet.utils import dist_utils, printE, printS, printT
from navidet.utils import logger as utils
from navidet.utils.wandb_logger import wandb_log

GIGABYTE = 1024 ** 3

# Only these keys (plus the running total `loss`) are surfaced in the live
# tqdm postfix and the end-of-epoch summary print. The aux DETR losses
# (loss_*_dn_*, loss_*_enc_*, loss_*_pre, loss_*_0..N) still contribute to
# `loss` and the gradient, but flooding the terminal with them obscures the
# numbers that actually matter.
_ESSENTIAL_LOSS_KEYS = ("loss_vfl", "loss_keypoints", "loss_oks", "loss_kd")

def train_one_epoch(self_lr_scheduler,
                    lr_scheduler,
                    model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    batch_size:int,
                    grad_accum_steps:int, 
                    device: torch.device,
                    epoch: int,
                    max_norm: float = 0,
                    writer=None,
                    warmup_scheduler=None,
                    ema=None,
                    args=None):
    scaler = torch.amp.GradScaler(str(device), enabled=True) # FIXME
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # Create meters for all parameter groups
    for pg_idx in range(len(optimizer.param_groups)):
        lr_name = f'lr_pg{pg_idx}' if pg_idx > 0 else 'lr'
        metric_logger.add_meter(lr_name, utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    sub_batch_size = batch_size // args.grad_accum_steps

    printS(
        f"grad_accum_steps={args.grad_accum_steps}  "
        f"batch_size/GPU={batch_size}  "
        f"total_batch_size={batch_size * dist_utils.get_world_size()}"
    )

    optimizer.zero_grad()

    cur_iters = epoch * len(data_loader)
    is_master = dist_utils.is_main_process()
    pbar = tqdm(
        data_loader,
        desc=header,
        total=len(data_loader),
        disable=not is_master,
        dynamic_ncols=True,
        leave=False,
    )
    for i, (samples, targets) in enumerate(pbar):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        global_step = epoch * len(data_loader) + i

        for j in range(args.grad_accum_steps):
            start_idx = j * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples = samples[start_idx:final_idx]
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

            with torch.amp.autocast(str(device), enabled=True):
                outputs = model(new_samples, new_targets)
            
            with torch.amp.autocast(str(device), enabled=False):
                loss_dict = criterion(outputs, new_targets)
                losses = sum(loss_dict.values())

            if args.use_amp:
                scaler.scale(losses).backward()
            else:
                losses.backward()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced_scaled = sum(loss_dict_reduced.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            printE(f"Loss is {loss_value}, stopping training\n  loss_dict={loss_dict_reduced}")
            sys.exit(1)


        if args.use_amp:
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
                    
        if ema is not None:
            ema.update(model)
            
        # LR scheduling
        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if warmup_scheduler is not None:
                warmup_scheduler.step() 

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        # Update learning rates for all parameter groups
        for pg_idx, param_group in enumerate(optimizer.param_groups):
            lr_name = f'lr_pg{pg_idx}' if pg_idx > 0 else 'lr'
            metric_logger.update(**{lr_name: param_group["lr"]})

        # Live tqdm postfix — essentials only. tqdm batches refreshes itself,
        # so updating each iter is cheap.
        if is_master:
            postfix = {"loss": f"{loss_value:.3f}"}
            for k in _ESSENTIAL_LOSS_KEYS:
                if k in loss_dict_reduced:
                    postfix[k] = f"{float(loss_dict_reduced[k]):.3f}"
            postfix["lr"] = f"{optimizer.param_groups[0]['lr']:.2e}"
            pbar.set_postfix(postfix)

        if dist_utils.is_main_process() and global_step % 10 == 0:
            free, total = torch.cuda.mem_get_info(device) \
                if torch.cuda.is_available() else (0, 0)
            mem_used_GB = (total - free) / GIGABYTE
            if writer:
                writer.add_scalar('Loss/total', loss_value, global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f'Loss/{k}', v.item(), global_step)
                writer.add_scalar('Info/memory', mem_used_GB, global_step)
            # wandb mirrors the same scalars; key names are flat so the W&B
            # dashboard groups them under panels.
            wb_payload = {
                "train/loss": loss_value,
                "train/epoch": epoch,
                "train/mem_GB": mem_used_GB,
            }
            for j, pg in enumerate(optimizer.param_groups):
                wb_payload[f"train/lr_pg{j}"] = pg["lr"]
            for k, v in loss_dict_reduced.items():
                wb_payload[f"train/{k}"] = float(v.item())
            wandb_log(wb_payload, step=global_step)

        optimizer.zero_grad()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    pbar.close()
    # Concise end-of-epoch summary (essentials only). Verbose per-loss
    # breakdown is reserved for the validation logger.
    summary_keys = ("loss",) + _ESSENTIAL_LOSS_KEYS + ("lr",)
    summary = "  ".join(
        f"{k}: {metric_logger.meters[k].global_avg:.4f}"
        for k in summary_keys if k in metric_logger.meters
    )
    printT(f"Epoch [{epoch}] avg | {summary}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}




@torch.no_grad()
def evaluate(model, postprocessors, coco_evaluator, data_loader, device,
             writer=None, save_results=False,
             *, epoch: int | None = None, output_dir: str | None = None,
             save_vis_n: int = 10, vis_score_thresh: float = 0.3):
    """Evaluate the model on a dataloader.

    Optional viz dump: when `output_dir` is provided, picks `save_vis_n`
    random validation images, runs them through the postprocessor, and writes
    drawn predictions to ``<output_dir>/results/epoch_<N>/img_<idx>.jpg``.
    Master-rank only.
    """
    import os
    import random
    from PIL import Image
    from navidet.utils.draw_pose import draw_pose_prediction, default_skeleton

    model.eval()
    if coco_evaluator is not None:
        coco_evaluator.cleanup()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    res_json = []

    # --- prepare random sample set for visualization ----------------------
    base_dataset = data_loader.dataset
    num_body_points = getattr(postprocessors, "num_body_points", None) \
        or getattr(getattr(model, "decoder", None), "num_body_points", 4)
    vis_dir = None
    vis_idx_set: set[int] = set()
    if output_dir and save_vis_n > 0 and dist_utils.is_main_process():
        vis_dir = os.path.join(output_dir, "results",
                               f"epoch_{epoch:04d}" if epoch is not None else "eval")
        os.makedirs(vis_dir, exist_ok=True)
        rng = random.Random(epoch if epoch is not None else 0)
        all_idx = list(range(len(base_dataset)))
        vis_idx_set = set(rng.sample(all_idx, min(save_vis_n, len(all_idx))))
        printS(f"saving {len(vis_idx_set)} validation prediction(s) → {vis_dir}")

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, targets)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output
               for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # --- dump visualization for sampled indices -----------------------
        if vis_dir is not None:
            for t, r in zip(targets, results):
                idx = int(t["idx"].item())
                if idx not in vis_idx_set:
                    continue
                img_path = getattr(base_dataset, "img_files", None)
                if img_path is None or idx >= len(img_path):
                    continue
                try:
                    pil = Image.open(img_path[idx]).convert("RGB")
                    drawn = draw_pose_prediction(
                        pil,
                        keypoints=r["keypoints"].detach().cpu().numpy(),
                        scores=r["scores"].detach().cpu().numpy(),
                        labels=r["labels"].detach().cpu().numpy(),
                        num_keypoints=num_body_points,
                        score_thresh=vis_score_thresh,
                        skeleton=default_skeleton(num_body_points),
                    )
                    out_path = os.path.join(vis_dir, f"img_{idx:06d}_pred.jpg")
                    drawn.save(out_path, quality=90)
                except Exception as e:  # don't crash eval on a viz error
                    printS(f"viz skip idx={idx}: {e}")

        if save_results:
            for k, v in res.items():
                scores = v['scores']
                labels = v['labels']
                keypoints = v['keypoints']

                for s, l, kpt in zip(scores, labels, keypoints):
                    res_json.append(
                        {
                        "image_id": k,
                        "category_id": l.item(),
                        "keypoints": kpt.round(decimals=4).tolist(),
                        "score": s.item()
                        }
                        )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    printT(f"Validation averaged stats: {metric_logger}")
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if save_results:
        return res_json

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        stats['coco_eval_keypoints'] = coco_evaluator.coco_eval['keypoints'].stats.tolist()

    # Mirror essential val metrics + COCO AP to wandb. Step is left None so
    # wandb groups validation rows under the latest training step.
    wb_val = {f"val/{k}": float(v) for k, v in stats.items()
              if k in ("loss",) + _ESSENTIAL_LOSS_KEYS}
    if coco_evaluator is not None and 'coco_eval_keypoints' in stats:
        coco_stats = stats['coco_eval_keypoints']
        # COCO keypoints summary order: AP, AP.5, AP.75, AP_M, AP_L, AR, AR.5, AR.75, AR_M, AR_L
        names = ["AP", "AP50", "AP75", "AP_M", "AP_L",
                 "AR", "AR50", "AR75", "AR_M", "AR_L"]
        for n, v in zip(names, coco_stats):
            wb_val[f"val/coco_{n}"] = float(v)
    if wb_val:
        wandb_log(wb_val)
    return stats