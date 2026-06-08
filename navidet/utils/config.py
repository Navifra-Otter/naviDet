"""
YAML 설정 로더.

중첩 dict를 속성 접근(cfg.train.epochs)이 가능한 네임스페이스로 감싼다.
CLI에서 `--set a.b=1 c=2` 형태의 override도 지원.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml")


class Config(SimpleNamespace):
    """dict처럼도(cfg["k"]) 속성처럼도(cfg.k) 접근 가능한 설정 컨테이너."""

    def __getitem__(self, k):
        return getattr(self, k)

    def get(self, k, default=None):
        return getattr(self, k, default)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out


def _wrap(d):
    if isinstance(d, dict):
        return Config(**{k: _wrap(v) for k, v in d.items()})
    return d


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    """YAML 로드 → Config. overrides: ['train.lr=1e-3', 'data.imgsz=512']."""
    path = path or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        _set_nested(data, key.strip(), yaml.safe_load(val))
    return _wrap(data)


def _set_nested(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value
