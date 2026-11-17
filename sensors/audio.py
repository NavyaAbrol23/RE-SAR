"""Synthetic acoustic sensing: direction-of-arrival detections of intermittent distress calls.

A signal abstraction (inverse-square-like decay, per-wall attenuation, environment noise floor),
NOT a physical acoustic propagation model.
"""
from __future__ import annotations

import math

import numpy as np

from sensors.base_sensor import BaseSensor, SensorObservation


class AudioSensor(BaseSensor):
    name = "audio"

    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        self.c = cfg["sensors"]["audio"]

    def observe(self, world, cond, manual, failed, t, ctx):
        c = self.c
        noise_floor = 0.02 + 0.15 * cond["acoustic_noise"] + 0.3 * manual
        dets = []
        if not failed:
            att_env = 1 - world.env.get("audio_attenuation", 0) * world.severity * 0.8
            for v in world.victims:
                if not v.responsive or self.rng.random() > c["call_prob"]:
                    continue
                d = math.hypot(v.x - world.x, v.y - world.y)
                walls = world.walls_between(world.x, world.y, v.x, v.y)
                I = c["source_level"] / (1 + (d / c["ref_distance"]) ** 2)
                I *= c["wall_attenuation"] ** walls * att_env * (1 - 0.8 * manual)
                meas = I + abs(self.rng.normal(0, noise_floor))
                snr = meas / noise_floor
                if snr < c["detect_snr"] * self.rng.uniform(0.8, 1.2):
                    continue
                bearing = math.atan2(v.y - world.y, v.x - world.x) - world.theta
                bearing += self.rng.normal(0, c["doa_std"] * (1 + 2 / math.sqrt(snr)) + 0.6 * manual)
                rng_est = c["ref_distance"] * math.sqrt(max(c["source_level"] / meas - 1, 0.01))
                rng_est *= math.exp(self.rng.normal(0, 0.25))
                dets.append({"bearing": float(bearing), "intensity": float(meas), "snr": float(snr),
                             "range": float(rng_est), "true": True})
        lam = 0.04 + 0.35 * cond["acoustic_noise"] + 0.8 * manual + (1.5 if failed else 0.0)
        for _ in range(self.rng.poisson(lam)):
            meas = noise_floor * self.rng.uniform(1.2, 2.5)
            dets.append({"bearing": float(self.rng.uniform(-math.pi, math.pi)), "intensity": float(meas),
                         "snr": float(meas / noise_floor), "range": float(self.rng.uniform(1, 8)),
                         "true": False})
        for d in dets:  # robot never sees the GT flag
            d.pop("true")
        measured_floor = noise_floor * (1 + self.rng.normal(0, 0.1)) if not failed else 2.0
        return SensorObservation(self.name, t, True, {"detections": dets,
                                                      "noise_floor": float(max(measured_floor, 1e-3))})
