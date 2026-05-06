import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os 
import signal
os.environ['PYTORCH_JIT'] = '0'  # JIT 비활성화 (일부 환경에서 문제 발생)
import torch
from configs import cfg, parse_args, update_config
from pose.core.builder import Builder
from pose.engine.trainer import Trainer
from pose.utils.dist import DDPManager
from pose.utils import printS, printE, DISTRIBUTED, signal_handler
signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        args = parse_args()
        update_config(cfg, args)
        ddp_manager = DDPManager(cfg.gpus)
        builder = Builder(cfg)
        model = builder.model()
        trainDS = builder.dataset(cfg.dataset.train_dir)
        validDS = builder.dataset(cfg.dataset.valid_dir)
        model, trainLD, validLD = builder.set_device(model, trainDS, validDS, ddp_manager.device)
        printS(f"Allocated GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.3f} GB")
        loss_fn = builder.loss(model)
        optimizer = builder.optimizer(model.parameters())
        lr_scheduler = builder.lr_scheduler(optimizer)
        
        trainer = Trainer(
            cfg, 
            model=model,
            trainloader=trainLD,  
            validloader=validLD,  
            optimizer=optimizer, 
            lr_scheduler=lr_scheduler, 
            loss_fn=loss_fn, 
            ddp_manager=ddp_manager,
            metric=None,
            use_scalar=True
        )

        trainer.train()
    except Exception as e:
        printE(e)
    finally:
        if DISTRIBUTED:
            import torch.distributed as distributed
            if distributed.is_initialized():
                distributed.destroy_process_group()