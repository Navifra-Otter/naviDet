from yacs.config import CfgNode as CN

# new_allowed=True at the root: lets EC-style YAML blocks
# (train_dataloader, ema, scaler, evaluator, etc.) merge in without having to
# pre-declare every key. The YacsYAMLConfig adapter (navidet/core/registry/
# yacs_config.py) routes those into yaml_cfg for the EdgeCrafter solver.
C = CN(new_allowed=True)
C.title = "naviDet"
# Task — one of: 'pose', 'detection', 'segmentation'
C.task = 'pose'
C.seed = 42
C.use_deterministic = True
C.gpus = 0,

C.model = CN()
C.model.model_name = 'custom_dinov3convnext'
C.model.nkpts = (4, 3)
C.model.ncls = 7
# Backbone selector — 'ec*' prefix routes to ecvit (own KD-style backbone),
# 'dinov3*' / 'custom_dinov3*' prefix routes to legacy dinov3 (ablation).
C.model.backbone_name = 'dinov3_convnext_base'
C.model.backbone_ckps = None
C.model.finetuning = True

# `method` is a free-form node populated by configs/method/<task>/<variant>.yaml.
# The Builder feeds this dict through EdgeCrafter's registry (`create()`) to
# instantiate ECPose / ECDet / ECSeg with their backbone+encoder+decoder.
C.method = CN(new_allowed=True)

# Knowledge distillation — when enabled, teacher is built via
# Builder.teacher_model() and used inside engine/{det,pose}_trainer.py.
C.kd = CN()
C.kd.enabled = False
C.kd.teacher_backbone = 'dinov3_convnext_base'
C.kd.teacher_ckpt = None
C.kd.feat_loss_weight = 1.0
C.kd.logit_loss_weight = 0.0

C.dataset = CN(new_allowed=True)
C.dataset.img_size = 512
C.dataset.dataset = 'yolo_pose'
C.dataset.train_dir = 'data/train'
C.dataset.valid_dir = 'data/valid'
C.dataset.cache_ram = False  # True 시 워커별로 디코딩+리사이즈된 uint8 numpy를 캐시 (persistent_workers 권장)

C.dataloader = CN()
C.dataloader.batch_size = 16
C.dataloader.num_workers = 4
C.dataloader.pin_memory = True
C.dataloader.shuffle = True
C.dataloader.drop_last = True
C.dataloader.persistent_workers = True
C.dataloader.prefetch_factor = 4

C.trainer = CN()
C.trainer.epochs = 100
C.trainer.save_path = "weights"
C.trainer.valid_term = 5
C.trainer.save_term = 100//10
C.trainer.early_stopping = CN()
C.trainer.early_stopping.enabled = True
C.trainer.early_stopping.patience = 10
C.trainer.early_stopping.min_delta = 1e-4
C.trainer.early_stopping.monitor = 'val_loss'  # 'val_loss' | 'train_loss' | 'val_coco_AP'

C.trainer.coco_eval = CN()
C.trainer.coco_eval.enabled = True
C.trainer.coco_eval.num_select = 100
C.trainer.coco_eval.score_thresh = 0.0
C.trainer.coco_eval.sigmas = []  # empty -> default [0.05] * nkpts

C.wandb = CN()
C.wandb.enabled = False
C.wandb.project = 'pose-estimation'
C.wandb.entity = ''
C.wandb.run_name = ''
C.wandb.tags = []


# lr_scheduler / optimizer are populated entirely from YAML. The minimal
# Builder path uses getattr() with sensible fallbacks for missing keys, so we
# don't pre-declare any specific fields here.
C.lr_scheduler = CN(new_allowed=True)
C.optimizer = CN(new_allowed=True)

