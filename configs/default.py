from yacs.config import CfgNode as CN

C = CN()
C.title = "Pose Estimation with DINOv3"
C.task = 'pose'
C.seed = 42
C.use_deterministic = True
C.gpus = 0,

C.model = CN()
C.model.model_name = 'custom_dinov3convnext'
C.model.nkpts = (4, 3)
C.model.ncls = 7
C.model.backbone_name = 'dinov3_convnext_base'
C.model.backbone_ckps = None
C.model.finetuning = True

C.dataset = CN()
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


C.lr_scheduler = CN()
C.lr_scheduler.type = 'cosine'
C.lr_scheduler.warmup_epochs = 5
C.lr_scheduler.min_lr = 1e-6

C.optimizer = CN()
C.optimizer.type = 'adamw'
C.optimizer.lr = 1e-4
C.optimizer.weight_decay = 1e-2

