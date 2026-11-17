"""Victim tracks with reliability-weighted Bayesian evidence accumulation.

Each track keeps a log-odds L that a victim is present. For a modality m with reliability r_m:
  detection    : L += r_m * c * log(p_hit / p_fa)           (c = detector confidence)
  missed (cell inside m's field of view, m healthy enough):
                 L += r_m * 0.5 * log((1 - p_hit) / (1 - p_fa))
A track is CONFIRMED (counted as a discovered victim) when sigmoid(L) >= confirm_prob if two or
more modalities support it, >= single_modality_confirm (0.97) if only one imaging modality does,
and never on acoustic evidence alone. A confirmation counts only once. One unreliable sensor cannot produce certainty because its evidence is scaled
by r_m; a failed sensor (r_m = 0) contributes nothing, positive or negative.
Position: inverse-variance update with variance sigma^2 / max(r_m, 0.05).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from perception.data_association import associate

L_MIN, L_MAX, L_PRIOR = -6.0, 8.0, -1.0


@dataclass
class Track:
    tid: int
    x: float
    y: float
    var: float
    L: float = L_PRIOR
    hits: dict = field(default_factory=lambda: {"rgb": 0, "thermal": 0, "audio": 0})
    confirmed: bool = False
    confirmed_t: int = -1
    last_t: int = 0

    @property
    def prob(self) -> float:
        return 1.0 / (1.0 + math.exp(-self.L))


class VictimTracker:
    def __init__(self, cfg: dict):
        p = cfg["perception"]
        self.gate = p["gate"]
        self.confirm = p["confirm_prob"]
        self.confirm_single = p.get("single_modality_confirm", 0.97)
        self.llr_hit = {m: math.log(p["p_hit"][m] / p["p_fa"][m]) for m in p["p_hit"]}
        self.llr_miss = {m: math.log((1 - p["p_hit"][m]) / (1 - p["p_fa"][m])) for m in p["p_hit"]}
        self.reset()

    def reset(self):
        self.tracks: list[Track] = []
        self.next_id = 0

    @property
    def confirmed(self):
        return [t for t in self.tracks if t.confirmed]

    def update(self, cands: list[dict], r: dict, t: int, in_view_fn) -> list[Track]:
        """cands are in world frame. in_view_fn(modality, x, y) -> bool. Returns newly confirmed."""
        pairs, unassigned = associate(cands, self.tracks, self.gate)
        detected = set()
        for i, tr in pairs:
            c = cands[i]
            self._apply(tr, c, r)
            detected.add((tr.tid, c["modality"]))
            tr.last_t = t
        for i in unassigned:
            c = cands[i]
            if r.get(c["modality"], 0) <= 0.02:
                continue
            tr = Track(self.next_id, c["x"], c["y"], c["sigma"] ** 2, last_t=t)
            self.next_id += 1
            self._apply(tr, c, r, position=False)
            self.tracks.append(tr)
            detected.add((tr.tid, c["modality"]))
        # negative evidence from imaging modalities that should have seen the track
        for tr in self.tracks:
            for m in ("rgb", "thermal"):
                if (tr.tid, m) in detected or r.get(m, 0) <= 0.02:
                    continue
                if in_view_fn(m, tr.x, tr.y):
                    tr.L = max(L_MIN, tr.L + r[m] * 0.5 * self.llr_miss[m])
        newly = []
        for tr in self.tracks:
            mods = [m for m, n in tr.hits.items() if n > 0]
            if tr.confirmed or mods == ["audio"]:
                continue  # acoustic evidence alone is too coarse to confirm a victim
            thr = self.confirm if len(mods) >= 2 else max(self.confirm, self.confirm_single)
            if tr.prob >= thr:
                tr.confirmed, tr.confirmed_t = True, t
                newly.append(tr)
        # prune weak unconfirmed tracks, merge duplicates
        self.tracks = [tr for tr in self.tracks if tr.confirmed or tr.L > -4.0]
        self._merge()
        return newly

    def _apply(self, tr: Track, c: dict, r: dict, position=True):
        m = c["modality"]
        rm = max(r.get(m, 0.0), 0.0)
        tr.L = float(np.clip(tr.L + rm * c["conf"] * self.llr_hit[m], L_MIN, L_MAX))
        tr.hits[m] += 1
        if not position:  # new track: initialise variance from this measurement
            tr.var = c["sigma"] ** 2 / max(rm, 0.05)
            return
        # audio only refines position while no visual/thermal evidence exists (bearing-only, coarse)
        if m == "audio" and tr.hits["rgb"] + tr.hits["thermal"] > 0:
            return
        if m != "audio" and tr.hits["rgb"] + tr.hits["thermal"] == 1:
            tr.var = max(tr.var, 4.0)  # first precise fix replaces a coarse acoustic estimate
        mv = c["sigma"] ** 2 / max(rm, 0.05)
        k = tr.var / (tr.var + mv)
        tr.x += k * (c["x"] - tr.x)
        tr.y += k * (c["y"] - tr.y)
        tr.var = (1 - k) * tr.var

    def _merge(self):
        keep = []
        for tr in sorted(self.tracks, key=lambda t: (not t.confirmed, -t.L)):
            dup = next((k for k in keep if math.hypot(k.x - tr.x, k.y - tr.y) < 1.0), None)
            if dup is None:
                keep.append(tr)
            else:
                dup.L = max(dup.L, tr.L)
                for m in dup.hits:
                    dup.hits[m] += tr.hits[m]
        self.tracks = keep
