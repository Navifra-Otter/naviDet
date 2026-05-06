import argparse
from pose.utils import line, printM

def parse_args():
    parser = argparse.ArgumentParser(description="codes for ddp")
    parser.add_argument(
        '--cfg', 
		help='Path to the configuration file',
		default='configs/method/dinov3_pose.yaml',
		type=str,
	)
    parser.add_argument(
		'--test', 
		help='model test',
        default=False,
	)
    parser.add_argument(
        '--gpus', '-g',
        help='GPUs to use, e.g. 0,1,2,3 or -1 to use CPU',
        default=None,
        type=str,
	)
    parser.add_argument(
		'--ckpt', 
		help='Path to the checkpoint',
        default=None,
		type=str,
	)   
    return parser.parse_known_args()[0]

@line
def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if args.gpus is not None:
        gpus = args.gpus.replace(',', '')
        gpus = tuple(int(gpu) for gpu in gpus)
        cfg.gpus = gpus
    if args.ckpt is not None:
        cfg.model.checkpoint = args.ckpt
    cfg.freeze()
    printM(f"[Configuration]\n")
    printM(cfg)
