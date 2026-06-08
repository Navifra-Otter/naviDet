"""
6DoF 추정 결과의 직렬화 / 발행(publish).

naviEYE의 pose_info 출력처럼, 모델/파이프라인이 만든 R,t,size를 외부로 내보내기
좋은 형태(평면 x,y + yaw, quaternion 등)로 변환하고 발행한다. 기본 구현은 JSON
파일/stdout 발행이며, ROS·소켓 등은 Publisher를 상속해 _emit만 교체하면 된다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class PoseMessage:
    cls: int                      # 클래스 id
    score: float                  # confidence
    t: list                       # [x, y, z] 카메라좌표 (m)
    quat: list                    # [w, x, y, z]
    yaw_deg: float                # 지면(X–Z) yaw (deg)
    size: list                    # [width, height, depth] (m)


def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """3x3 회전행렬 → quaternion [w,x,y,z]."""
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = np.argmax([m[0, 0], m[1, 1], m[2, 2]])
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


def build_message(cls: int, score: float, R: np.ndarray, t: np.ndarray,
                  size: np.ndarray) -> PoseMessage:
    """R,t,size → PoseMessage (지면 yaw = X축의 X–Z 평면 방위각)."""
    x_axis = R[:, 0]
    yaw = float(np.degrees(np.arctan2(x_axis[2], x_axis[0])))   # X–Z 평면 방위
    return PoseMessage(
        cls=int(cls), score=round(float(score), 4),
        t=[round(float(v), 4) for v in t],
        quat=[round(float(v), 5) for v in matrix_to_quaternion(R)],
        yaw_deg=round(yaw, 2),
        size=[round(float(v), 4) for v in size],
    )


class PosePublisher:
    """기본: JSON 파일/stdout 발행. 다른 채널은 _emit 오버라이드."""

    def __init__(self, out_path: str | None = None, verbose: bool = False):
        self.out_path = out_path
        self.verbose = verbose

    def publish(self, frame_id: str, messages: list[PoseMessage]):
        payload = {"frame": frame_id, "objects": [asdict(m) for m in messages]}
        self._emit(payload)
        return payload

    def _emit(self, payload: dict):
        line = json.dumps(payload, ensure_ascii=False)
        if self.verbose:
            print(line)
        if self.out_path:
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
