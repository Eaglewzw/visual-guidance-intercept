"""YAML 配置加载，支持点号访问（cfg.camera.focal_length）"""
import os
import yaml

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")


class DotDict(dict):
    """支持属性访问的 dict（递归）"""

    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return DotDict(v) if isinstance(v, dict) else v

    __setattr__ = dict.__setitem__


def load_config(path=None) -> DotDict:
    path = path or DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return DotDict(yaml.safe_load(f))
