"""Configuration loading: default.yaml deep-merged with a disaster profile and overrides."""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DISASTERS = ("earthquake", "fire", "flood")


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(name: str) -> dict:
    with open(os.path.join(CONFIG_DIR, name), "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(disaster: str = "earthquake", overrides: dict | None = None) -> dict[str, Any]:
    if disaster not in DISASTERS:
        raise ValueError(f"Unknown disaster '{disaster}'. Choose from {DISASTERS}.")
    cfg = deep_merge(_read("default.yaml"), _read(f"{disaster}.yaml"))
    cfg["disaster"] = disaster
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
