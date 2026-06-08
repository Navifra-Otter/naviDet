"""navidet — CSP 백본 기반 통합 검출 프레임워크.

두 계열 모델을 한 패키지에서 제공한다(백본·넥·데이터 파이프라인 공유):
  · Det6DoF : 1-stage 6DoF Object Pose Estimation (Pose6DoFHead + DepthHead)
  · MultiTaskDet : 2D 멀티태스크 Detection / Segmentation / Pose (MultiTaskHead)

빌드는 `build_model(task=...)`로 선택 (task ∈ {6dof, detect, segment, pose}).

패키지 구조
  core/      신경망 코어 (blocks, backbone, head, model)
  module/    학습/처리 모듈 (loss, loss_mt, dataset, pose_label, trainer, metrics ...)
  utils/     유틸 (geometry, camera, nms, visualize, config)
  messaging/ 결과 직렬화/발행 (publisher)
  tools/     실행 스크립트 (train, predict, eval, build_labels, viz_gt, demo)
  config/    설정 (default_6dof.yaml, default_pose.yaml)
"""

from .core.head import DepthHead, MultiTaskHead, Pose6DoFHead, Proto
from .core.model import Det6DoF, MultiTaskDet, build_model
from .module.loss import Pose6DoFLoss, bbox_ciou
from .module.loss_mt import MultiTaskLoss
from .utils.geometry import (geodesic_loss, quaternion_to_matrix,
                             rotation_6d_to_matrix, unproject_translation)

__all__ = [
    # 6DoF 계열
    "Det6DoF", "Pose6DoFHead", "DepthHead", "Pose6DoFLoss",
    # 2D 멀티태스크 계열
    "MultiTaskDet", "MultiTaskHead", "Proto", "MultiTaskLoss",
    # 공통
    "build_model", "bbox_ciou",
    "rotation_6d_to_matrix", "quaternion_to_matrix",
    "geodesic_loss", "unproject_translation",
]
