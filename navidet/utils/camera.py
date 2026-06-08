"""
카메라 intrinsics 로더 — 파일(.ini) **또는 카메라에서 직접**.

- 파일 경로: Femto Bolt 등에서 export된 [ColorIntrinsic]/[DepthIntrinsic]/
  [D2CTransformParam] .ini 를 읽는다 (`load_orbbec_ini`).
- 라이브 카메라: 실행 중인 Orbbec SDK 파이프라인/디바이스에서 직접 읽는다
  (`from_orbbec_pipeline` / `CameraIntrinsics.from_orbbec`). SDK 가 없어도 본
  모듈 import 는 실패하지 않도록 SDK 는 호출 시점에만 lazy import 한다.
- 그 외: K 행렬·파라미터·dict 등 무엇이 들어와도 `resolve_intrinsics` 가
  `CameraIntrinsics` 로 정규화한다 (predictor 등에서 소스 무관하게 사용).

데이터셋의 depth PNG 는 color 해상도(예 1920x1080)로 정합(D2C=identity)되어
있으므로, 키포인트(=color 정규좌표) 언프로젝션에는 ColorIntrinsic 을 사용한다.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx],
                         [0, self.fy, self.cy],
                         [0, 0, 1]], dtype=np.float64)

    def scaled_to(self, width: int, height: int) -> "CameraIntrinsics":
        """다른 해상도로 intrinsics를 스케일(예: depth 원해상도 ↔ color)."""
        sx, sy = width / self.width, height / self.height
        return CameraIntrinsics(self.fx * sx, self.fy * sy,
                                self.cx * sx, self.cy * sy, width, height)

    # ------------------------------------------------------------------ #
    #  생성자들 (파일 외 소스)
    # ------------------------------------------------------------------ #
    @classmethod
    def from_params(cls, fx, fy, cx, cy, width, height) -> "CameraIntrinsics":
        """개별 파라미터 → CameraIntrinsics."""
        return cls(float(fx), float(fy), float(cx), float(cy), int(width), int(height))

    @classmethod
    def from_K(cls, K, width: int, height: int) -> "CameraIntrinsics":
        """3x3 K 행렬 + 해상도 → CameraIntrinsics."""
        K = np.asarray(K, dtype=np.float64)
        return cls(K[0, 0], K[1, 1], K[0, 2], K[1, 2], int(width), int(height))

    @classmethod
    def from_orbbec(cls, intrinsic, width=None, height=None) -> "CameraIntrinsics":
        """
        Orbbec SDK 의 intrinsic 객체(OBCameraIntrinsic 등) → CameraIntrinsics.
        fx/fy/cx/cy(+width/height) 속성을 duck-typing 으로 읽는다. width/height 가
        객체에 없으면 인자로 받는다 (스트림 프로파일에서만 따로 줄 때 대비).
        """
        w = getattr(intrinsic, "width", None) if width is None else width
        h = getattr(intrinsic, "height", None) if height is None else height
        if w is None or h is None:
            raise ValueError("intrinsic 에 width/height 가 없음 — 인자로 명시하세요.")
        return cls(float(intrinsic.fx), float(intrinsic.fy),
                   float(intrinsic.cx), float(intrinsic.cy), int(w), int(h))


# ----------------------------------------------------------------------------- #
#  파일(.ini) 로더
# ----------------------------------------------------------------------------- #
def load_orbbec_ini(path: str, section: str = "ColorIntrinsic") -> CameraIntrinsics:
    """Orbbec .ini에서 지정 섹션의 intrinsics를 읽는다."""
    cfg = configparser.ConfigParser()
    # .ini에 BOM/공백 섹션이 있어도 견고하게 읽기
    with open(path, "r", encoding="utf-8-sig") as f:
        cfg.read_file(f)
    s = cfg[section]
    return CameraIntrinsics(
        fx=float(s["fx"]), fy=float(s["fy"]),
        cx=float(s["cx"]), cy=float(s["cy"]),
        width=int(s["width"]), height=int(s["height"]),
    )


# ----------------------------------------------------------------------------- #
#  라이브 카메라(Orbbec SDK)에서 직접
# ----------------------------------------------------------------------------- #
def from_orbbec_pipeline(pipeline, stream: str = "color") -> CameraIntrinsics:
    """
    실행 중인 pyorbbecsdk Pipeline 에서 intrinsics 를 직접 읽는다.

      from pyorbbecsdk import Pipeline
      pipeline = Pipeline(); pipeline.start(config)
      intr = from_orbbec_pipeline(pipeline, "color")   # color/depth

    color 는 depth-to-color 정합 기준이므로 키포인트 언프로젝션에 사용한다.
    SDK 버전별 속성명(rgb_intrinsic/color_intrinsic 등)을 모두 시도한다.
    """
    param = pipeline.get_camera_param()                 # OBCameraParam
    if stream == "depth":
        intr = getattr(param, "depth_intrinsic", None)
    else:
        intr = (getattr(param, "rgb_intrinsic", None)
                or getattr(param, "color_intrinsic", None))
    if intr is None:
        raise RuntimeError(f"카메라 파라미터에서 '{stream}' intrinsic 을 찾지 못함: "
                           f"{type(param).__name__} 속성 {dir(param)}")
    return CameraIntrinsics.from_orbbec(intr)


def from_orbbec_profile(stream_profile) -> CameraIntrinsics:
    """
    pyorbbecsdk VideoStreamProfile 에서 intrinsics 를 읽는다.
      profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR) ...
      intr = from_orbbec_profile(profile)
    """
    vsp = stream_profile
    if hasattr(vsp, "as_video_stream_profile"):
        vsp = vsp.as_video_stream_profile()
    intr = vsp.get_intrinsic()
    w = getattr(intr, "width", None) or vsp.get_width()
    h = getattr(intr, "height", None) or vsp.get_height()
    return CameraIntrinsics.from_orbbec(intr, width=w, height=h)


# ----------------------------------------------------------------------------- #
#  통합 정규화 — 소스 무관하게 CameraIntrinsics 로
# ----------------------------------------------------------------------------- #
def resolve_intrinsics(source, width: int | None = None, height: int | None = None,
                       section: str = "ColorIntrinsic") -> CameraIntrinsics:
    """
    무엇이 들어와도 CameraIntrinsics 로 정규화한다.

    지원 source:
      - CameraIntrinsics            : 그대로
      - str / Path (.ini)           : load_orbbec_ini
      - dict {fx,fy,cx,cy,width,height} : from_params
      - 3x3 K 행렬(ndarray/list)     : from_K  (width/height 필수)
      - Orbbec Pipeline (get_camera_param 보유) : from_orbbec_pipeline
      - Orbbec intrinsic 객체(fx/fy/cx/cy 보유)  : from_orbbec
    """
    import os

    if isinstance(source, CameraIntrinsics):
        return source
    if isinstance(source, (str, os.PathLike)):
        return load_orbbec_ini(str(source), section)
    if isinstance(source, dict):
        return CameraIntrinsics.from_params(**source)
    # 라이브 파이프라인 (color intrinsics 우선)
    if hasattr(source, "get_camera_param"):
        return from_orbbec_pipeline(source, "color")
    # Orbbec intrinsic 객체 (duck-typing)
    if all(hasattr(source, a) for a in ("fx", "fy", "cx", "cy")):
        return CameraIntrinsics.from_orbbec(source, width=width, height=height)
    # K 행렬
    arr = np.asarray(source, dtype=np.float64)
    if arr.shape == (3, 3):
        if width is None or height is None:
            raise ValueError("K 행렬 소스는 width/height 가 필요합니다.")
        return CameraIntrinsics.from_K(arr, width, height)
    raise TypeError(f"intrinsics 소스를 해석할 수 없음: {type(source)!r}")
