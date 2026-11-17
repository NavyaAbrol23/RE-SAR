# RE-SAR Technical Specification

Coordinates: continuous cell units. Cell `(i, j)` spans `[i, i+1) x [j, j+1)`; its centre is `(i + 0.5, j + 0.5)`. `x` points right and `y` points down (image convention). The heading θ follows `atan2(dy, dx)` in that frame. The map frame equals the world frame (the start pose is known); after that the robot only has its estimate.

**Ground-truth boundary.** `environment.World` is read only by:
- `sensors/*` (to generate observations)
- `visualization` (rendering)
- `Mission.evaluate()`, experiments and the simulator's true-reliability log (evaluation)

The robot pipeline, the RL observation and the reward never read it. `tests/test_rl.py::test_observation_has_no_ground_truth_leak` checks this.

## config/
| File | Content |
|---|---|
| `default.yaml` | All shared parameters (world, robot, sensors, degradation levels, reliability weights, mapping, perception, navigation, RL reward and PPO) |
| `earthquake/fire/flood.yaml` | `environment:` block: darkness, dust, smoke, heat, water, acoustic_noise, rubble_density, wheel_slip, lidar_noise, audio_attenuation (all in [0, 1]) |
| `loader.py` | `load_config(disaster, overrides) -> dict` deep-merges default, disaster and overrides |

## environment/ (A, ground truth)
| Module | Inputs | Outputs |
|---|---|---|
| `map_generator.generate_layout` | size, disaster, difficulty, severity, rng | obstacle grid, rubble mask, start cell. Recursive-division rooms with doors, then disaster damage: earthquake (broken walls, rubble piles), flood (washed-out walls), clutter. Unreachable pockets are sealed. |
| `hazard_generator.generate_hazards` | grid, rubble, start, disaster, severity, env | `terrain` (floor/rubble/water/burning), `hazard` intensity, `heat`, `water`, `smoke`, `dust` fields |
| `victim_generator` | reachable mask, start, n | `Victim(x, y, temperature, responsive)`; `Decoy(kind = visual \| thermal)` |
| `terrain.py` | – | Terrain classes, traversal costs, slip factors, colours |
| `world.World` | cfg, seed | Ground truth plus `apply_motion(rot, dist, rng)` (slip, actuation noise, collisions, hazard exposure); vectorised `cast_rays`, `visible_mask`, `walls_between`; `ground_truth_conditions()` |

Cells whose hazard is ≥ `impassable_hazard` (0.85) count as impassable. The reachable mask excludes them, and victims are always placed in reachable cells.

## sensors/ (B)
Every sensor returns `SensorObservation(name, t, delivered, data)`. Ground-truth reliability is not part of the observation.

| Sensor | `data` | Degradation (environment + manual level m) | Failure mode |
|---|---|---|---|
| `rgb.RGBCamera` | `image` uint8 104×104×3 (egocentric top-down view, forward = up, 120° FOV, 6-cell range, occlusion by walls); `fov_mask` | Haze (smoke, dust), darkness, blur, noise, contrast loss | Black frame |
| `thermal.ThermalCamera` | `temp` float °C, `t_sat`, `fov_mask` | Heat interference lowers the saturation limit and raises background; noise; blur | Fully saturated |
| `lidar.LidarSensor` | `angles` (72, relative), `ranges` (NaN = dropout, inf = no return), `max_range` 8 | Range noise, dropouts (water, dust), spurious short returns (dust) | All NaN |
| `imu.IMUSensor` | `yaw` (absolute, drifting), `yaw_rate` (Δθ per step), `frozen` | Bias random walk (drift) and noise grow with m | Frozen output |
| `odometry.OdometrySensor` | `dist`, `d_theta` | Wheel slip: wheels turn while the robot doesn't move; scale error | Exactly zero output |
| `audio.AudioSensor` | `detections` [{bearing, intensity, snr, range}], `noise_floor` | Noise floor, attenuation (walls, flood), false detections | Saturated mic: floor = 2.0, bursts of false detections |

- `degradation.DegradationController` holds per-sensor levels (normal 0, mild 0.35, severe 0.7, failed 1). It combines them with the environment-induced degradation `e_env` and keeps the **ground-truth** effective reliability `r* = (1 − e_env)(1 − m)`, which is 0 if failed.
- `suite.SensorSuite.observe(world, t, ctx) -> {name: obs}`.

## reliability/ (C, primary contribution)
| Module | Inputs | Outputs |
|---|---|---|
| `environment_state.EnvironmentStateEstimator` | Mission profile (disaster, severity) and the **sensed** heat/water/rubble around the estimated pose | Estimated conditions; `r_env[s]` |
| `observation_quality` | Raw observations; commands (for proprioceptive residuals) | `q[s]` plus diagnostics (brightness, contrast, sharpness, saturation, valid-return ratio, residuals, noise floor) |
| `sensor_health.SensorHealthMonitor` | Observations, commands | `h[s]` ∈ [0, 1]; labels OK / DEGRADED / FAILED |
| `cross_modal.CrossModalAgreement` | Perception candidates, tracks, pose, usable mask; odometry/IMU vs scan-match motion | `a[s]` (EMA, neutral 0.5) |
| `hybrid_fusion.HybridReliability` | e, q, h, a; strategy ∈ {fixed, env, quality, env_quality, env_quality_health, hybrid, learned} | `ReliabilityReport`: r, components, normalised group weights, all-failed flags |
| `learned.py` (optional B6) | Logged (e, q, h, a, sensor) → r* pairs | NumPy-inference MLP, `rl/models/reliability_mlp.npz` |

## mapping/ (D1, E)
| Module | Inputs | Outputs |
|---|---|---|
| `localization.EKFLocalizer` | Odometry, IMU, command, reliabilities; scan-match pose | `mu = [x, y, θ]`, covariance `P` |
| `scan_matching.scan_match` | Occupancy probability, scan, predicted pose, EKF σ | Best pose and score (two-stage correlative search on an endpoint likelihood field, with a free-space penalty) |
| `occupancy_grid.OccupancyGrid` | Estimated pose, scan, weight | Log-odds `L`; `prob`, `known`, `free`, `occupied`, entropy |
| `traversability.TerrainEstimator` | RGB / thermal images at the estimated pose, reliability weights | Sensed `heat`, `water`, `rubble` layers → hazard and traversal cost |
| `uncertainty_map.UncertaintyMap` | Observed cells, σ_xy | Uncertainty layer |
| `shared_map.SharedMap` | All of the above plus victim tracks | `layers()`: occupancy, traversability, hazard, victim, unexplored, uncertainty |

## perception/ (D2)
| Module | Output |
|---|---|
| `rgb_detection.detect_rgb` | Candidates `{f, l, conf, sigma, modality}` in robot-local coordinates (skin blob with a clothing ring) |
| `thermal_detection.detect_thermal` | Compact 30–40 °C blobs with ≥ 3 °C contrast to their surroundings |
| `audio_detection.detect_audio` | Bearing + range-from-intensity candidates with large σ |
| `data_association` | Local → world (estimated pose); gated nearest-neighbour, at most one candidate per modality per track per step |
| `victim_tracking.VictimTracker` | Tracks with log-odds, per-modality hits, confirmation (counted once), pruning and merging |

## navigation/ (F, G)
| Module | Function |
|---|---|
| `costmap.build_costmap` | inf for occupied cells or sensed impassable hazard; otherwise traversability × (1 + 6·hazard) + 1.5·unknown + near-obstacle penalty |
| `astar.astar / dijkstra` | 8-connected, no corner cutting, octile heuristic |
| `frontier` | Frontier cells → clusters → `Frontier(x, y, size, dist, info_gain, hazard, victim, uncertainty)` sorted by path cost; selectors random / nearest / infogain |
| `safety.SafetyController` | Vetoes a step into a known-occupied cell, a sensed impassable hazard, or a close LiDAR return ahead |

## core/mission.py
`Mission(cfg, seed, reliability, nav, degradation, modalities, chooser)`:
- `step()` runs one perception–action cycle.
- `plan(index)` takes an RL action.
- `compute_frontiers()` lists the current frontiers.
- `evaluate()` returns ground-truth metrics.

Termination happens on `max_steps`, `all_victims_found` (confirmed ≥ expected and estimated coverage ≥ 0.8), `exploration_complete`, `unable_to_continue`, or `agent_stop` (RL only).

## rl/
| Module | Content |
|---|---|
| `observation.build_observation` | 228-dimensional vector (see file docstring for the layout) |
| `sar_env.SAREnvironment` | `Discrete(K + 1)` actions = 8 frontier slots + STOP; macro step of up to 15 moves; domain randomisation; training and evaluation seed ranges are disjoint |
| `reward.compute_reward` | Configurable α β η γ δ λ terms |
| `train.py` / `evaluate.py` / `policy.py` | SB3 PPO training (device auto), evaluation, and the GUI/experiment chooser |

## visualization/
| Module | Content |
|---|---|
| `simulator.SimulationController` | The actions behind every button: pause, step, reset, new map, cycle nav/fusion/sensor level, speed |
| `pygame_view.App` | Menu and dashboard (world, sensors, shared map, exploration, metrics, legend) |
| `panels.py` | Drawing helpers |

## experiments/
| Module | Content |
|---|---|
| `metrics.py` | Parallel mission runner, summary with 95% CI, paired bootstrap differences, plots |
| `sensor_failure.py` | Exp 1–3 |
| `fusion_ablation.py` | Exp 4 |
| `exploration_comparison.py` | Exp 5 + computational metrics |
| `generalization.py` | Exp 6 |
