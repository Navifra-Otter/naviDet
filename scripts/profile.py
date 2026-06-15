#!/usr/bin/env python3
"""Pose 백엔드(YOLO vs EdgeCrafter) 추론 시간 분해 프로파일러.

같은 머신에서 두 backend 의 end-to-end FPS 와 *순수 GPU forward* 시간을 분리
측정해, 속도 차이가 (1) 모델 forward 자체인지 (2) CPU 전/후처리/파이썬 래핑
오버헤드인지 구분한다.

  end-to-end = preprocess(CPU resize/normalize) + forward(GPU) + postprocess(파이썬 래핑)
  overhead   = end-to-end - forward

forward 는 nn.Module.forward 를 CUDA Event 로 감싸 정확히 GPU 시간만 잰다.

사용 예:
    # repo 루트에서 실행 (체크포인트 상대경로 './checkpoints/...' 해석을 위해)
    python scripts/profile.py --type edgecrafter
    python scripts/profile.py --type naviai
    python scripts/profile.py --type edgecrafter --cudnn-benchmark
    python scripts/profile.py --type naviai --image some_frame.jpg

옵션:
    --config PATH      config yaml (기본 navieye/configs/config_mando.yaml)
    --type T           naviai | edgecrafter (기본: config 의 model_info.type)
    --image PATH       실제 이미지로 측정 (없으면 random BGR frame)
    --width/--height   random frame 크기 (기본 1280x720)
    --iters N          측정 반복 (기본 200)
    --warmup N         워밍업 반복 (기본 30)
    --cudnn-benchmark  cudnn autotuner 켜기 (고정 shape 에서 conv 가속; jitter↑)
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# repo 루트를 import path 에 추가 (scripts/ 하위에서 실행돼도 navieye 패키지 인식).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 이 파일명(profile.py)이 stdlib의 'profile' 모듈을 가린다. torch._dynamo가
# cProfile→profile을 import할 때 충돌하므로, 스크립트 디렉터리를 sys.path에서 제거.
# (sys.path엔 상대경로('scripts')로 들어있을 수 있어 resolve로 비교.)
_HERE = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if p and Path(p).resolve() != _HERE]
# stdlib profile/cProfile 를 지금(scripts/ 제거 후) 미리 import 해 sys.modules에
# 캐시한다. 이후 torch._dynamo가 cProfile→profile 을 import해도 이 파일이 아니라
# 캐시된 stdlib 모듈을 쓰게 된다.
import cProfile  # noqa: E402,F401


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="navieye/configs/config_mando.yaml")
    p.add_argument("--type", dest="mtype", default=None,
                   choices=["naviai", "edgecrafter", "navidet"],
                   help="model_info.type 오버라이드 (기본: config 값)")
    # --- navidet 백엔드 전용 ---
    p.add_argument("--ckpt", default=None, help="[navidet] 체크포인트 .pt")
    p.add_argument("--ini", default=None, help="[navidet] 카메라 intrinsics .ini (선택)")
    p.add_argument("--conf", type=float, default=0.25, help="[navidet] confidence 임계값")
    p.add_argument("--image", default=None, help="측정용 실제 이미지 경로")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--cudnn-benchmark", action="store_true",
                   help="cudnn.benchmark=True (기본 코드에선 False 로 꺼져 있음)")
    return p.parse_args()


def find_torch_module(predictor):
    """predictor 내부의 핵심 nn.Module(.forward 를 후킹할 대상)을 찾는다."""
    import torch.nn as nn
    # EdgeCrafter: ECPosePredictor.inferencer.model
    inf = getattr(predictor, "inferencer", None)
    if inf is not None and isinstance(getattr(inf, "model", None), nn.Module):
        return inf.model
    # Ultralytics YOLO: PosePredictor.model (첫 호출 후 lazy 생성됨)
    m = getattr(predictor, "model", None)
    if isinstance(m, nn.Module):
        return m
    return None


class ForwardTimer:
    """nn.Module.forward 를 감싸 GPU forward 시간(ms)을 CUDA Event 로 누적."""

    def __init__(self, module):
        import torch
        self.torch = torch
        self.module = module
        self.times_ms: list[float] = []
        self.enabled = False
        self.capture = False          # True면 다음 forward의 입력을 1회 저장(FLOPs 계산용)
        self.sample = None            # 캡처된 (args, kwargs)
        self._cuda = (next(module.parameters()).is_cuda
                      if any(True for _ in module.parameters()) else False)
        self._orig = module.forward
        module.forward = self._wrapped

    def _wrapped(self, *args, **kwargs):
        if self.capture and self.sample is None:
            self.sample = (args, kwargs)   # 실제 모델이 받는 입력 그대로 캡처
            self.capture = False
        if not self.enabled:
            return self._orig(*args, **kwargs)
        torch = self.torch
        if self._cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = self._orig(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            self.times_ms.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            out = self._orig(*args, **kwargs)
            self.times_ms.append((time.perf_counter() - t0) * 1000.0)
        return out

    def restore(self):
        self.module.forward = self._orig


def summarize(name, samples_ms):
    if not samples_ms:
        return f"  {name:<14}: (no samples)"
    s = sorted(samples_ms)
    mean = statistics.mean(s)
    median = statistics.median(s)
    p90 = s[int(0.90 * (len(s) - 1))]
    fps = 1000.0 / mean if mean > 0 else float("nan")
    return (f"  {name:<14}: mean {mean:6.2f} ms | median {median:6.2f} | "
            f"p90 {p90:6.2f} | {fps:6.1f} FPS")


def compute_gflops(module, args, kwargs):
    """모델 1회 forward의 GFLOPs(=MAC×2, vision 논문/Ultralytics 컨벤션) 추정.

    실제 forward가 받은 입력(args/kwargs)을 그대로 써서 측정한다.
    백엔드 우선순위: thop → fvcore → torch 내장(FlopCounterMode).
    반환: (gflops or None, 사용한 backend 이름).
    """
    import torch
    # 1) thop — YOLO 생태계 기본. MACs 반환 → ×2 로 FLOPs 환산.
    if not kwargs:
        try:
            import thop
            macs, _ = thop.profile(module, inputs=args, verbose=False)
            return 2 * macs / 1e9, "thop"
        except Exception:
            pass
    # 2) fvcore — MACs 반환 → ×2.
    if not kwargs:
        try:
            from fvcore.nn import FlopCountAnalysis
            fca = FlopCountAnalysis(module, args)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            return 2 * fca.total() / 1e9, "fvcore"
        except Exception:
            pass
    # 3) torch 내장 — get_total_flops()는 이미 MAC×2(FLOPs). kwargs도 지원.
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fcm = FlopCounterMode(display=False)
        with fcm, torch.no_grad():
            module(*args, **kwargs)
        return fcm.get_total_flops() / 1e9, "torch"
    except Exception as e:
        return None, f"실패({type(e).__name__})"


def main():
    args = parse_args()

    import torch

    if args.mtype == "navidet":
        # --- navidet 백엔드 (navieye 불필요). ckpt의 task로 6DoF/2D 자동 분기 ---
        if not args.ckpt:
            raise SystemExit("navidet 백엔드는 --ckpt 가 필요합니다.")
        # repo 루트를 import path에 추가 (navidet 패키지 인식)
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        from navidet.core.model import TASK_PRESET
        from navidet.module.predictor import Det6DoFPredictor, MultiTaskPredictor

        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
        # 체크포인트 meta만 가볍게 읽어 task 판별 (2D는 ini 불필요)
        task = torch.load(args.ckpt, map_location="cpu",
                          weights_only=False).get("task", "6dof")
        if task in TASK_PRESET:                       # detect/segment/pose → 2D
            predictor = MultiTaskPredictor(args.ckpt, conf=args.conf)
        else:                                          # 6dof
            predictor = Det6DoFPredictor(args.ckpt, ini=args.ini, conf=args.conf)
        mtype, device = f"navidet:{task}", str(predictor.device)
        print("=" * 72)
        print(f"  ckpt           : {args.ckpt}")
        print(f"  model type     : {mtype}  (imgsz={predictor.imgsz}, nc={predictor.nc})")
        print(f"  device         : {device}")
        print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        print("=" * 72)
    else:
        # navieye.modules.model 이 import 시 cudnn.benchmark=False 로 설정함.
        # 그 다음에 우리 플래그로 덮어쓴다.
        from navieye.modules import model as navi_model

        torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)

        cfg_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else Path(args.config)
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        model_info = cfg["model_info"]
        if args.mtype:
            model_info["type"] = args.mtype
        mtype = model_info["type"]
        conf = model_info["conf"]
        device = model_info["device"]

        print("=" * 72)
        print(f"  config         : {cfg_path}")
        print(f"  model type     : {mtype}")
        print(f"  device         : {device}")
        print(f"  cudnn.benchmark: {torch.backends.cudnn.benchmark}")
        print("=" * 72)

        # pose predictor 빌드 (앱과 동일 경로). edgecrafter 인데 .pth 없으면 YOLO fallback.
        predictor = navi_model._build_pose(model_info, conf, device)

    # 입력 프레임 준비 (BGR uint8) — 앱이 color_frame(BGR np.ndarray)을 넘기는 것과 동일.
    if args.image:
        import cv2
        frame = cv2.imread(str(REPO_ROOT / args.image if not Path(args.image).is_absolute()
                                else args.image))
        if frame is None:
            raise FileNotFoundError(f"이미지 로드 실패: {args.image}")
        print(f"  input frame    : {args.image}  {frame.shape[1]}x{frame.shape[0]}")
    else:
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (args.height, args.width, 3), dtype=np.uint8)
        print(f"  input frame    : random  {args.width}x{args.height}")

    def predict_once():
        return predictor(frame)

    # --- 워밍업 (CUDA context, cudnn autotune, lazy 모델 빌드 등) ---------------
    for _ in range(args.warmup):
        predict_once()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 워밍업 후에야 YOLO 의 내부 model 이 생성되므로 여기서 후킹.
    module = find_torch_module(predictor)
    timer = None
    n_params = None
    if module is not None:
        n_params = sum(p.numel() for p in module.parameters())
        # Ultralytics AutoBackend 등 래퍼는 .parameters() 가 비어 보일 수 있어
        # 내부 .model 로 한 단계 내려가 다시 센다.
        if n_params == 0 and hasattr(module, "model"):
            n_params = sum(p.numel() for p in module.model.parameters())
        timer = ForwardTimer(module)
        timer.enabled = True
        timer.capture = True          # 측정 첫 forward에서 입력을 캡처(FLOPs용)
    else:
        print("  [warn] 내부 nn.Module 을 못 찾음 — forward 격리 측정 생략 (end-to-end 만).")

    # --- 측정 -----------------------------------------------------------------
    e2e_ms: list[float] = []
    for _ in range(args.iters):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict_once()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e_ms.append((time.perf_counter() - t0) * 1000.0)

    if timer is not None:
        timer.restore()

    # --- GFLOPs (모델이 실제로 받은 입력 기준, 1 forward) ----------------------
    gflops = backend = None
    if module is not None and timer is not None and timer.sample is not None:
        cap_args, cap_kwargs = timer.sample
        with torch.no_grad():
            gflops, backend = compute_gflops(module, cap_args, cap_kwargs)
        # 캡처된 입력 shape (텐서 1개 가정) 표기용
        in_shape = tuple(cap_args[0].shape) if cap_args and hasattr(cap_args[0], "shape") else None

    # --- 리포트 ---------------------------------------------------------------
    print("-" * 72)
    if n_params is not None:
        print(f"  params         : {n_params/1e6:.2f} M")
    if gflops is not None:
        shape_s = f", input={in_shape}" if in_shape else ""
        print(f"  GFLOPs         : {gflops:.2f}  (MAC×2, via {backend}{shape_s})")
    elif backend is not None:
        print(f"  GFLOPs         : 측정 불가 ({backend})")
    print(summarize("end-to-end", e2e_ms))
    if timer is not None and timer.times_ms:
        fwd = timer.times_ms[-len(e2e_ms):]  # 측정 구간만
        print(summarize("forward(GPU)", fwd))
        overhead = statistics.mean(e2e_ms) - statistics.mean(fwd)
        pct = 100.0 * overhead / statistics.mean(e2e_ms)
        print(f"  {'overhead':<14}: mean {overhead:6.2f} ms  "
              f"(pre+post / 파이썬 래핑, end-to-end 의 {pct:.0f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
