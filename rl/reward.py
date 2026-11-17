"""Reward for one RL decision (a macro-step of up to `macro_steps` low-level moves).

R = alpha * information_gain        (occupancy-entropy reduction, bits / 100)
  + beta  * new_victim_discovery    (newly CONFIRMED tracks; each track rewarded once)
  + eta   * exploration_progress    (increase of estimated known-map fraction * 10)
  - gamma * travel_cost             (commanded distance in cells)
  - delta * hazard_exposure         (sum of SENSED hazard at the estimated cell, robot-side)
  - lambda* collision_penalty       (bumper contacts — the robot physically feels these)
Additional terms: invalid frontier index -> -invalid_action; STOP while frontiers remain ->
-(2 + 5 * unexplored fraction) so the agent cannot farm reward by ending early, and every
macro-step costs travel, so endless wandering is not rewarded.
"""
from __future__ import annotations


def compute_reward(acc: dict, cfg: dict) -> float:
    c = cfg["rl"]["reward"]
    return float(c["alpha"] * acc["info_gain"] / 100.0
                 + c["beta"] * acc["new_victims"]
                 + c["eta"] * acc["coverage_delta"] * 10.0
                 - c["gamma"] * acc["travel"]
                 - c["delta"] * acc["hazard"]
                 - c["lambda"] * acc["collisions"]
                 - c["invalid_action"] * acc.get("invalid", 0)
                 - acc.get("stop_penalty", 0.0))
