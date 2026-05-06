import os
import torch
import torch.nn as nn
from tqdm import tqdm
from pose.utils import DISTRIBUTED, MASTER_RANK, printM, printT, printS, colored_msg
from pose.utils.dist import DDPManager, set_seed
from pose.eval import CocoKeypointEvaluator


class Trainer:
    def __init__(
            self,
            cfg,
            model:nn.Module,
            trainloader,
            validloader,
            optimizer,
            lr_scheduler,
            loss_fn,
            ddp_manager: DDPManager,
            metric=None,
            use_scalar=False,
        ):

        set_seed(cfg.seed, cfg.use_deterministic)
        self.cfg = cfg
        self.task = cfg.task
        self.ddp_manager = ddp_manager
        self.device = self.ddp_manager.device
        self.model = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.metric = metric
        self.scaler = torch.amp.GradScaler(enabled=True) if use_scalar else None

        self.output_path = cfg.trainer.save_path if hasattr(cfg.trainer, 'save_path') else "weights"
        self.output_path = os.path.join('output', self.output_path)
        os.makedirs(self.output_path, exist_ok=True)
        self.metric_score()

        es_cfg = getattr(cfg.trainer, 'early_stopping', None)
        self.es_enabled = bool(getattr(es_cfg, 'enabled', False)) if es_cfg is not None else False
        self.es_patience = getattr(es_cfg, 'patience', 10) if es_cfg is not None else 10
        self.es_min_delta = getattr(es_cfg, 'min_delta', 1e-4) if es_cfg is not None else 1e-4
        self.es_monitor = getattr(es_cfg, 'monitor', 'val_loss') if es_cfg is not None else 'val_loss'
        self.es_counter = 0
        self.es_best = float('inf')

        self.coco_evaluator = None
        eval_cfg = getattr(cfg.trainer, 'coco_eval', None)
        self.coco_eval_enabled = bool(getattr(eval_cfg, 'enabled', False)) if eval_cfg is not None else False
        if MASTER_RANK and self.coco_eval_enabled and validloader is not None:
            try:
                kpt_shape = tuple(cfg.model.nkpts)
                ncls = cfg.model.ncls
                img_size = cfg.dataset.img_size
                num_select = getattr(eval_cfg, 'num_select', 100)
                score_thresh = getattr(eval_cfg, 'score_thresh', 0.0)
                sigmas = getattr(eval_cfg, 'sigmas', None)
                sigmas = list(sigmas) if sigmas else None
                self.coco_evaluator = CocoKeypointEvaluator(
                    dataset=validloader.dataset,
                    ncls=ncls,
                    kpt_shape=kpt_shape,
                    img_size=img_size,
                    num_select=num_select,
                    score_thresh=score_thresh,
                    sigmas=sigmas,
                )
                printS(f"COCO keypoint evaluator initialized "
                       f"(ncls={ncls}, kpt_shape={kpt_shape}, img_size={img_size})")
            except Exception as e:
                printS(f"COCO evaluator init failed: {e} (continuing without it)")
                self.coco_evaluator = None

        self.wandb = None
        wb_cfg = getattr(cfg, 'wandb', None)
        if MASTER_RANK and wb_cfg is not None and getattr(wb_cfg, 'enabled', False):
            try:
                import wandb
                init_kwargs = dict(
                    project=getattr(wb_cfg, 'project', 'pose-estimation'),
                    config=self._cfg_to_dict(cfg),
                )
                entity = getattr(wb_cfg, 'entity', '')
                run_name = getattr(wb_cfg, 'run_name', '')
                tags = list(getattr(wb_cfg, 'tags', []) or [])
                if entity:
                    init_kwargs['entity'] = entity
                if run_name:
                    init_kwargs['name'] = run_name
                if tags:
                    init_kwargs['tags'] = tags
                wandb.init(**init_kwargs)
                self.wandb = wandb
                printS("wandb logging enabled")
            except Exception as e:
                printS(f"wandb init failed: {e} (continuing without wandb)")
                self.wandb = None

    @staticmethod
    def _cfg_to_dict(cfg):
        try:
            from yacs.config import CfgNode
            if isinstance(cfg, CfgNode):
                return {k: Trainer._cfg_to_dict(v) for k, v in cfg.items()}
        except Exception:
            pass
        if isinstance(cfg, dict):
            return {k: Trainer._cfg_to_dict(v) for k, v in cfg.items()}
        return cfg

    def metric_score(self):
        self.epoch_loss = 0.0
        self.best_loss = float('inf')

    def iter_one_epoch(self, epoch, dataloader=None, train=True):
        pbar = tqdm(dataloader,
                    total=len(dataloader),
                    dynamic_ncols=True,
                    mininterval=0.5,) if MASTER_RANK else dataloader
        self.epoch_loss = 0.0
        for images, targets in pbar:
            imgs = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            imgs = imgs.float() / 255.0
            self.optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                preds = self.model(imgs)
                loss, loss_items = self.loss_fn(preds, targets)

            if train:
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

            current_loss = loss.item()
            l_cls, l_kpt, l_obj = self.loss_fn.add_loss(loss_items)
            if train:
                self.epoch_loss += current_loss
            else:
                self.epoch_loss += current_loss
            if MASTER_RANK:
                tag = '[TRAIN]' if train else '[VALID]'
                color = 'green' if train else 'yellow'
                message = f"{epoch+1} epoch | loss: {current_loss:.4f} | obj: {l_obj:.4f} | cls: {l_cls:.4f} | kpt: {l_kpt:.4f}"
                pbar.desc = f" {colored_msg(tag, color)} {message}"


    def train(self):
        epochs = self.cfg.trainer.epochs
        warmup_epochs = self.cfg.lr_scheduler.warmup_epochs
        if warmup_epochs > 0:
            printM()
            printT(f"LR Warmup Epochs: {warmup_epochs} | Initial LR: {self.optimizer.param_groups[0]['lr']}")
        stopped = False
        for epoch in range(epochs):
            printM()
            self.model.train()
            self.loss_fn.set_train_loss()
            self.warmup(epoch, warmup_epochs) if epoch < warmup_epochs else None
            self.iter_one_epoch(epoch, self.trainloader)
            self.lr_scheduler.step()
            num_batches = max(1, len(self.trainloader))
            avg_loss = self.epoch_loss / num_batches
            avg_cls = self.loss_fn.cls_loss_sum / num_batches
            avg_kpt = self.loss_fn.kpt_loss_sum / num_batches
            avg_obj = self.loss_fn.obj_loss_sum / num_batches
            current_lr = self.optimizer.param_groups[0]['lr']
            printT(f"   total | loss: {avg_loss:.4f} | obj: {avg_obj:.4f} | cls: {avg_cls:.4f} | kpt: {avg_kpt:.4f}")

            log = {
                'epoch': epoch + 1,
                'lr': current_lr,
                'train/loss': avg_loss,
                'train/obj': avg_obj,
                'train/cls': avg_cls,
                'train/kpt': avg_kpt,
            }

            val_metrics = None
            if (epoch + 1) % self.cfg.trainer.valid_term == 0 or (epoch + 1) == epochs:
                val_metrics = self.validate(epoch)
                log.update({
                    'val/loss': val_metrics['loss'],
                    'val/obj': val_metrics['obj'],
                    'val/cls': val_metrics['cls'],
                    'val/kpt': val_metrics['kpt'],
                })
                if 'coco' in val_metrics:
                    for k, v in val_metrics['coco'].items():
                        log[f'val/coco_{k}'] = v

            if self.wandb is not None:
                self.wandb.log(log, step=epoch + 1)

            self.save_checkpoint(epoch, avg_loss)

            if self.es_enabled:
                monitor_val = None
                higher_better = False
                if self.es_monitor == 'val_loss' and val_metrics is not None:
                    monitor_val = val_metrics['loss']
                elif self.es_monitor == 'train_loss':
                    monitor_val = avg_loss
                elif self.es_monitor == 'val_coco_AP' and val_metrics is not None and 'coco' in val_metrics:
                    monitor_val = val_metrics['coco'].get('AP')
                    higher_better = True
                if monitor_val is not None:
                    if not hasattr(self, '_es_higher_better_init'):
                        self.es_best = -float('inf') if higher_better else float('inf')
                        self._es_higher_better_init = True
                    improved = (monitor_val > self.es_best + self.es_min_delta) if higher_better \
                        else (monitor_val < self.es_best - self.es_min_delta)
                    if improved:
                        self.es_best = monitor_val
                        self.es_counter = 0
                    else:
                        self.es_counter += 1
                        printS(f"EarlyStopping: no improvement on {self.es_monitor} "
                               f"({monitor_val:.4f} vs best {self.es_best:.4f}) "
                               f"[{self.es_counter}/{self.es_patience}]")
                        if self.es_counter >= self.es_patience:
                            printS(f"EarlyStopping triggered at epoch {epoch+1}")
                            stopped = True
                            break

        if self.wandb is not None:
            self.wandb.summary['stopped_early'] = stopped
            self.wandb.summary['best_loss'] = self.best_loss
            self.wandb.finish()

    def warmup(self, epoch, warmup_epochs):
        warmup_factor = (epoch + 1) / warmup_epochs
        for param_group in self.optimizer.param_groups:
            if 'initial_lr' in param_group:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
            else:
                param_group['lr'] = param_group['lr'] * warmup_factor


    def validate(self, epoch):
        # Loss pass: PoseHead returns feature maps when self.training=True,
        # which is what loss_fn expects. We keep grad disabled and use
        # train-mode head + eval-mode batchnorm via a manual switch.
        self.loss_fn.set_train_loss()
        was_training = self.model.training
        self.model.eval()
        # PoseHead checks self.training to decide its return path; flip only it.
        head = self._unwrap(self.model)
        pose_head = getattr(head, 'head', None)
        if pose_head is not None:
            pose_head.train()
        with torch.no_grad():
            self.iter_one_epoch(epoch=epoch, dataloader=self.validloader, train=False)
        num_batches = max(1, len(self.validloader))
        avg_loss = self.epoch_loss / num_batches
        avg_cls = self.loss_fn.cls_loss_sum / num_batches
        avg_kpt = self.loss_fn.kpt_loss_sum / num_batches
        avg_obj = self.loss_fn.obj_loss_sum / num_batches
        printT(f"[VALID] | loss: {avg_loss:.4f} | obj: {avg_obj:.4f} | cls: {avg_cls:.4f} | kpt: {avg_kpt:.4f}")

        out = {'loss': avg_loss, 'obj': avg_obj, 'cls': avg_cls, 'kpt': avg_kpt}

        if self.coco_evaluator is not None:
            if pose_head is not None:
                pose_head.eval()
            coco_stats = self._run_coco_eval(epoch)
            out['coco'] = coco_stats
            printT("[VALID/COCO] " + " | ".join(f"{k}: {v:.4f}" for k, v in coco_stats.items()))

        if was_training:
            self.model.train()
        return out

    def _unwrap(self, model):
        return model.module if DISTRIBUTED else model

    @torch.no_grad()
    def _run_coco_eval(self, epoch):
        self.coco_evaluator.reset()
        self.model.eval()
        loader = self.validloader
        bs = loader.batch_size
        next_id = 0
        pbar = tqdm(loader,
                    total=len(loader),
                    dynamic_ncols=True,
                    mininterval=0.5,
                    desc=f" {colored_msg('[COCO]', 'magenta')} {epoch+1}") if MASTER_RANK else loader
        for images, _targets in pbar:
            imgs = images.to(self.device, non_blocking=True).float() / 255.0
            with torch.amp.autocast(device_type='cuda', enabled=self.scaler is not None):
                raw = self.model(imgs)
            B = imgs.shape[0]
            image_ids = list(range(next_id, next_id + B))
            next_id += bs  # advance by full batch_size to stay aligned even if drop_last drops tail
            self.coco_evaluator.update_from_raw(raw.float(), image_ids)
        return self.coco_evaluator.summarize()

    def save_checkpoint(self, epoch, avg_loss):
        state_dict = self.model.module.state_dict() if DISTRIBUTED else self.model.state_dict()
        if (epoch + 1) % self.cfg.trainer.save_term == 0 or (epoch + 1) == self.cfg.trainer.epochs:
            torch.save(state_dict, f"{self.output_path}/pose_dino_epoch_{epoch+1}.pt")
            printS(f"Saved checkpoint: {self.output_path}/pose_dino_epoch_{epoch+1}.pt")

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save(state_dict, f"{self.output_path}/best.pt")
            printS(f"Saved best model (loss: {self.best_loss:.4f})")

    def load_checkpoint(self, path):
        if path and os.path.exists(path):
            param = torch.load(path, map_location=self.device)
            self.model.load_state_dict(param)
            printS(f"Loaded checkpoint from {path}")

    def cleanup(self):
        self.ddp_manager.cleanup()


