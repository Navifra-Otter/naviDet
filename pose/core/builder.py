
import os 
import torch.nn as nn
from torch.utils.data import DataLoader
from pose.utils import DISTRIBUTED
from typing import List, Dict

class Builder:
    def __init__(self, cfg):
        self.cfg = cfg 
        self.task = cfg.task

    def model(self) -> nn.Module:
        backbone_name = self.cfg.model.model_name.lower()
        if backbone_name == 'custom_dinov3convnext':
            from pose.model.pose.dinov3pose import DINOv3Pose
            model = DINOv3Pose(
                backbone=self.cfg.model.backbone_name,
                nkpts=self.cfg.model.nkpts,
                ncls=self.cfg.model.ncls,
                backbone_ckps=self.cfg.model.backbone_ckps,
                finetuning=self.cfg.model.finetuning,
                )
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
    
        return model 
    
    def loss(self, model):
        if self.task == 'pose':
            from pose.core.loss_fn.pose import ComputeLoss 
            return ComputeLoss(model, kpt_loss_type='oks')
        else:
            raise ValueError(f"Unsupported task for loss function: {self.task}")
        
    
    def metric(self):
        pass 

    def optimizer(self, param: nn.Module | List[Dict[str, nn.Module]]):
        optim = self.cfg.optimizer.type.lower()

        if optim == 'adamw':
            from torch.optim import AdamW
            optimizer = AdamW(
                param,
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay
            )
            return optimizer
        else:
            raise ValueError(f"Unsupported optimizer type: {optim}")

    def lr_scheduler(self, optimizer):
        lr_sched_type = self.cfg.lr_scheduler.type.lower()
        if lr_sched_type == 'cosine':
            from pose.core.scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.cfg.trainer.epochs,
                eta_min=self.cfg.lr_scheduler.min_lr
            )
            return scheduler
        else:
            raise ValueError(f"Unsupported LR scheduler type: {lr_sched_type}")

    def dataset(self, data_path):
        dataset_type = self.cfg.dataset.dataset.lower()
        if dataset_type  == 'yolo_pose':
            from pose.data.datasets.yolo_pose import YoloPoseDataset
            img_dir = os.path.join(data_path, 'images')
            label_dir = os.path.join(data_path, 'labels')
            cache_ram = bool(getattr(self.cfg.dataset, 'cache_ram', False))
            dataset = YoloPoseDataset(
                img_dir=img_dir,
                label_dir=label_dir,
                img_size=self.cfg.dataset.img_size,
                nkpts=self.cfg.model.nkpts[0],
                cache_ram=cache_ram,
            )

            return dataset
        else:
            raise ValueError(f"can't find data in \"{data_path}\", check the directory.")

    def transformer(self):
        pass

    def epoch(self):
        pass 

    def set_device(self, model, trainDS, validDS, device):

        nw = self.cfg.dataloader.num_workers
        prefetch = getattr(self.cfg.dataloader, 'prefetch_factor', 2)
        persistent = bool(getattr(self.cfg.dataloader, 'persistent_workers', True)) and nw > 0
        common = dict(
            num_workers=nw,
            pin_memory=self.cfg.dataloader.pin_memory,
            persistent_workers=persistent,
        )
        if nw > 0:
            common['prefetch_factor'] = prefetch

        if DISTRIBUTED:
            from torch.utils.data.distributed import DistributedSampler
            from torch.nn.parallel import DistributedDataParallel as DDP
            t_sampler = DistributedSampler(trainDS, shuffle=self.cfg.dataloader.shuffle)
            v_sampler = DistributedSampler(validDS, shuffle=False)
            model = DDP(
                model.to(device),
                device_ids=[device],
                output_device=device,
                find_unused_parameters=False
            )
            trainloader = DataLoader(
                trainDS,
                batch_size=self.cfg.dataloader.batch_size,
                sampler=t_sampler,
                collate_fn=trainDS.collate_fn,
                drop_last=self.cfg.dataloader.drop_last,
                **common,
            )

            validloader = DataLoader(
                validDS,
                batch_size=self.cfg.dataloader.batch_size,
                sampler=v_sampler,
                collate_fn=validDS.collate_fn,
                drop_last=False,
                **common,
            )
        else:
            model = model.to(device)
            trainloader = DataLoader(
                trainDS,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=self.cfg.dataloader.shuffle,
                collate_fn=trainDS.collate_fn,
                drop_last=self.cfg.dataloader.drop_last,
                **common,
            )
            validloader = DataLoader(
                validDS,
                batch_size=self.cfg.dataloader.batch_size,
                shuffle=False,
                collate_fn=validDS.collate_fn,
                drop_last=False,
                **common,
            )
        return model, trainloader, validloader