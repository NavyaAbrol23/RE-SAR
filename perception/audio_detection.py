"""Convert acoustic direction-of-arrival detections into (very uncertain) position candidates."""
from __future__ import annotations

import math

import numpy as np


def detect_audio(audio_data: dict) -> list[dict]:
    out = []
    for d in audio_data.get("detections", []):
        r = float(np.clip(d["range"], 0.5, 10.0))
        b = d["bearing"]  # robot frame, + = right (y-down world frame)
        conf = float(np.clip((d["snr"] - 1.0) / 4.0, 0.1, 0.8))
        out.append({"f": r * math.cos(b), "l": r * math.sin(b), "conf": conf,
                    "sigma": 0.35 * r + 1.0, "modality": "audio", "bearing": b})
    return out
