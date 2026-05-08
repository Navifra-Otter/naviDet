"""Registry/workspace adapted from EdgeCrafter (RT-DETR derived).

Used for the KD/module-build path that mirrors EdgeCrafter's `create()` pattern.
"""

from .workspace import GLOBAL_CONFIG, register, create
from .yaml_utils import *
from ._config import BaseConfig
from .yaml_config import YAMLConfig
from .yacs_config import YacsYAMLConfig
