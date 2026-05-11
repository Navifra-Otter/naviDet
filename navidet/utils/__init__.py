import os
import sys
import traceback 

DISTRIBUTED = False
MASTER_RANK = True
if 'RANK' in os.environ:
    MASTER_RANK = int(os.environ['RANK']) == 0
    DISTRIBUTED = True if int(os.environ.get("WORLD_SIZE", "1")) > 1 else False

def master_only(func):
    """rank 0에서만 실행하는 데코레이터"""
    def wrapper(*args, **kwargs):
        if MASTER_RANK:
            return func(*args, **kwargs)
    return wrapper

class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored_msg(msg, color=None):
    if color:
        return f"{getattr(Colors, color.upper())}{msg}\033[0m"
    else:
        return msg

@master_only
def printE(message=''):
    print(f" {colored_msg('[ERROR]', 'red')} {message}\n")
    traceback.print_exc()

@master_only
def printS(message=''):
    print(f" {colored_msg('[SYSTEMS]', 'blue')} {message}")

@master_only
def printW(message=''):
    print(f" {colored_msg('[WARNING]', 'yellow')} {message}")

@master_only
def printT(message=''):
    print(f" {colored_msg('[TRAIN]', 'green')} {message}")

@master_only
def printM(message='', color=None):
    print(colored_msg(message, color))



def line(func):
    text_line = "=" * 70
    def wrapper(*args, **kwargs):
        printM(f"\n{text_line}\n")
        func(*args, **kwargs)  
        printM(f"\n{text_line}\n")
    return wrapper    

def time_check(func):
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        printM(f"Model inference for {end_time - start_time:.4f} seconds", end='\r')
        return result
    return wrapper

def signal_handler(sig, frame):
    # 필요한 정리 작업(파일 닫기, 로그 저장 등)이 있다면 여기에 작성
    sys.exit(0)


# ---------------------------------------------------------------------------
# EdgeCrafter compatibility re-exports.
# Trainer / criterion code expects `from navidet.utils import dist_utils,
# MetricLogger, SmoothedValue, stats`.
# ---------------------------------------------------------------------------

from . import dist_utils as dist_utils  # noqa: E402
from .logger import MetricLogger, SmoothedValue  # noqa: E402, F401
from .profiler import stats  # noqa: E402, F401