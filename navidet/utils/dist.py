import os
import sys
import random
import atexit
import signal
from typing import List, Optional
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta
from navidet.utils import line, printM, printS, printW


def set_seed(seed: int, use_deterministic: bool = True) -> None:
    """
    재현성을 위한 시드 설정
    
    Args:
        seed: 랜덤 시드 값
        use_deterministic: True면 완전 결정론적(느림), False면 빠르지만 약간의 비결정성
    """
    # 환경변수 설정
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Python/NumPy/PyTorch 시드
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 멀티 GPU
    
    if use_deterministic:
        # 완전한 재현성 (느림)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # 빠른 학습 (약간의 비결정성)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class DDPManager:
    """분산 학습(DDP) 및 디바이스 관리 클래스"""
    
    def __init__(self, gpus: List[int]):
        """
        Args:
            gpus: 사용할 GPU ID 리스트. [-1]이면 CPU 모드
        """
        self.cpu = gpus[0] == -1
        self.num_gpus = len(gpus)
        self.target_gpu_ids = gpus
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device: Optional[torch.device] = None
        self.is_master = True
        self.ddp_initialized = False
        
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
        
        self.setup_device()
    
    @line
    def setup_device(self) -> None:
        """디바이스 설정 (CPU/Single GPU/DDP)"""
        printM(" [GPU Setting]\n", 'blue')
        
        if self.cpu or not torch.cuda.is_available():
            if not torch.cuda.is_available():
                printW('CUDA is not available')
            self.set_cpu()
        elif self.num_gpus == 1:
            self.set_cuda()
        else:
            # DDP 환경 변수 확인
            if self._is_ddp_environment():
                self.set_ddp()
            else:
                printW(f"{self.num_gpus}개의 GPU를 요청했으나 DDP 환경이 감지되지 않았습니다.")
                printM(" 싱글 GPU 모드로 전환합니다.\n", 'blue')
                self.set_cuda()
    
    @staticmethod
    def _is_ddp_environment() -> bool:
        return 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ
    
    def optimize_cpu_threads(self) -> None:
        # 외부에서 OMP_NUM_THREADS 가 이미 설정돼 있으면 그대로 사용 (train.zsh 가 정확히 계산함).
        # 그렇지 않으면 보수적으로 1 (DataLoader 워커 환경에서 안전한 기본값).
        if 'OMP_NUM_THREADS' in os.environ and os.environ['OMP_NUM_THREADS'].strip():
            printS(f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} (from environment)")
            return
        os.environ['OMP_NUM_THREADS'] = '1'
        printS("OMP_NUM_THREADS set to 1 (default)")
    
    def init_ddp(self, device_id: int) -> None:
        if dist.is_initialized():
            printW("DDP already initialized")
            return
        
        try:
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                timeout=timedelta(seconds=1800),
                device_id=torch.device(f'cuda:{device_id}')  # 🔥 경고 해결
            )
            self.ddp_initialized = True
        except Exception as e:
            raise RuntimeError(f"DDP 초기화 실패: {e}")
    
    def ddp_check(self, available_gpus):
        for gpu_id in self.target_gpu_ids:
            assert gpu_id < available_gpus, f"GPU {gpu_id}가 존재하지 않습니다."

    def set_ddp(self) -> None:
        """DDP 설정"""
        # 환경변수 검증
        if not self._is_ddp_environment():
            raise RuntimeError(
                " DDP 모드는 torchrun으로 실행해야 합니다.\n"
                " 예: torchrun --nproc_per_node=2 train.py"
            )
        
        # 환경변수에서 rank 정보 가져오기
        local_rank_idx = int(os.environ['LOCAL_RANK'])
        self.rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.local_rank = local_rank_idx
        
        # GPU 매핑
        available_gpus = torch.cuda.device_count()
        self.ddp_check(available_gpus)
        
        if available_gpus == 0:
            raise RuntimeError("CUDA GPU를 찾을 수 없습니다!")
        
        # GPU ID 결정
        gpu_id = self.target_gpu_ids[local_rank_idx]
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f'cuda:{gpu_id}')
        
        # CPU 스레드 최적화
        self.optimize_cpu_threads()
        
        # DDP 초기화 (device_id 전달)
        self.init_ddp(device_id=gpu_id)
        
        # 마스터 프로세스 여부
        self.is_master = self.rank == 0
        
        # 정보 출력 (마스터만)
        if self.is_master:
            printS(f"DDP Mode Activated")
            printS(f"World Size: {self.world_size}")
            printS(f"Backend: nccl")
            printS(f"Target GPUs: {self.target_gpu_ids}")
            printS(f"Available GPUs: {available_gpus}")
        
        printS(f"Rank {self.rank}: Using GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
        
        # 동기화 (모든 프로세스가 여기까지 도달할 때까지 대기)
        dist.barrier()
    
    def _signal_handler(self, signum: int, frame) -> None:
        """시그널 핸들러"""
        printS(f"\n\n Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self) -> None:
        """DDP 정리"""
        if self.ddp_initialized and dist.is_initialized():
            try:
                dist.barrier()  # 모든 프로세스 동기화
                dist.destroy_process_group()
                printS("DDP process group destroyed.")
            except Exception as e:
                printW(f"Error during DDP cleanup: {e}")
            finally:
                self.ddp_initialized = False
    
    def set_cuda(self) -> None:
        """Single GPU 모드 설정"""
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_master = True
        
        # GPU 존재 여부 확인
        available_gpus = torch.cuda.device_count()
        if available_gpus == 0:
            printW("CUDA 사용 가능한 GPU가 없습니다. CPU 모드로 전환합니다.")
            return self.set_cpu()
        
        # GPU ID 결정
        if not self.target_gpu_ids or self.target_gpu_ids[0] == -1:
            printW("target_gpu_ids가 비어있습니다. GPU 0번을 사용합니다.")
            gpu_id = 0
        else:
            gpu_id = self.target_gpu_ids[0]
        
        # GPU 유효성 검증
        if gpu_id >= available_gpus:
            printW(f"GPU {gpu_id}가 존재하지 않습니다. GPU 0을 사용합니다.")
            gpu_id = 0
        
        # CPU 스레드 최적화
        self.optimize_cpu_threads()
        
        # 디바이스 설정
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f'cuda:{gpu_id}')
        
        printS(f"Using Single GPU: {gpu_id}")
        printS(f"GPU Name: {torch.cuda.get_device_name(gpu_id)}")
    
    def set_cpu(self) -> None:
        """CPU 모드 설정"""
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.is_master = True
        self.device = torch.device('cpu')
        
        printS("Using CPU")
    
    def __del__(self):
        """소멸자"""
        self.cleanup()


# 사용 예시
if __name__ == "__main__":
    # 시드 설정
    set_seed(42, use_deterministic=True)
    
    # DDP 매니저 초기화
    # torchrun --nproc_per_node=3 train.py 로 실행
    ddp_manager = DDPManager(gpus=[0, 1, 2])
    
    print(f"Device: {ddp_manager.device}")
    print(f"Rank: {ddp_manager.rank}")
    print(f"World Size: {ddp_manager.world_size}")
    print(f"Is Master: {ddp_manager.is_master}")