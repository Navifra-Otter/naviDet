"""
EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation
Copyright (c) 2026 The EdgeCrafter Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

try:
    from calflops import calculate_flops
except ImportError:  # optional dep — only needed for `stats()`
    calculate_flops = None


def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:
    base_size = cfg.yaml_cfg["eval_spatial_size"]
    input_shape = (1, 3, *base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()
    params = sum(p.numel() for p in model_for_info.parameters())

    if calculate_flops is None:
        # `calflops` not installed; skip FLOPs/MACs report.
        del model_for_info
        return params, {f"Model Params:{params/1e6:.2f}M  (FLOPs/MACs unavailable — pip install calflops)"}

    flops, macs, _ = calculate_flops(model=model_for_info,
                                     input_shape=input_shape,
                                     output_as_string=True,
                                     output_precision=4,
                                     print_detailed=False)
    del model_for_info
    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" % (flops, macs, params)}
