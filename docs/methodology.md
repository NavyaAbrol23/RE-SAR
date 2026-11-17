# RE-SAR Methodology

## 1. Disaster environment modelling
- **Layout.**
  - The building is 40×30 cells, divided recursively into rooms joined by doors (2–3 cells wide).
  - The room size depends on difficulty, and so does clutter density (easy 2%, medium 4%, hard 7%, scaled by severity).
- **Damage by disaster.**
  - Earthquake: 8% of interior wall cells collapse, plus rubble piles (obstacles), rubble terrain and unstable ground.
  - Fire: 2–5 fire centres placed ≥ 8 cells from the start. Heat is a sum of Gaussians, scaled by 0.75 + 0.35·severity, so fire cores become impassable at medium severity and above. Smoke is blurred heat.
  - Flood: 25% of walls wash out. Smooth noise plus a gradient defines low ground; water covers a fraction of 0.25 + 0.45·severity of the map, and depth sets the hazard.
- **Separation of world states.** The ground-truth world, the robot's estimated map (occupancy, sensed hazard, victim tracks) and its unknown regions are kept separate.
- **Reproducibility.** Every map is reproducible from `seed`. Unreachable pockets are sealed so coverage is measured over the reachable area.

## 2. Sensor simulation
Sensors are synthetic but structurally meaningful. Each has a physical failure signature that the reliability layer has to detect without being told.
- **Cameras.** RGB and thermal are egocentric top-down views.
  - Each pixel is mapped to the world through the true pose for rendering, and back through the **estimated** pose for detection, so localisation error propagates into victim positions.
  - Victims are drawn as a body ellipse (clothing) plus a skin-coloured head.
  - Visual decoys are skin-coloured debris; thermal decoys are warm objects at 36–55 °C.
- **LiDAR.** 72 vectorised ray-marched beams.
- **IMU.** Integrates the true rotation plus bias; the bias follows a random walk (drift).
- **Odometry.** Measures true motion, but reports the *commanded* distance when the wheels slip or the robot collides.
- **Audio.** Victims call intermittently (p = 0.35; 75% of victims are responsive). Intensity is `I = L/(1 + (d/d0)²) · 0.6^walls · attenuation`. Detection requires SNR above 1.5; DOA noise grows as 1/√SNR; false detections arrive as a Poisson process. This is a signal abstraction, not acoustic propagation.

## 3. Sensor degradation
The effective ground-truth degradation of sensor s is `e_s = 1 − (1 − e_env,s)(1 − m_s)`, where m is the injected level (0 / 0.35 / 0.7 / 1). The simulator's true reliability is `r*_s = 1 − e_s` (0 when failed). The environment part uses the local conditions at the true pose:

| Sensor | e_env |
|---|---|
| RGB | 1 − (1 − 0.9·smoke)(1 − darkness)(1 − 0.7·dust) |
| Thermal | 0.85·heat·heat_profile + 0.15·smoke |
| LiDAR | lidar_noise·(dust + 0.5·smoke + 0.9·water) + 0.05·severity |
| IMU | 0.15·severity (earthquake), 0.05·severity (otherwise) |
| Odometry | 1.6·slip_probability |
| Audio | 0.7·acoustic_noise + 0.4·audio_attenuation·severity |

`r*` is used only to score the reliability estimators (MAE, ECE, failure-detection rate), never by the robot.

## 4. Reliability estimation (hybrid)
All terms lie in [0, 1].

- **E, environment prior.** The same functional form as above, but fed with *estimated* conditions: the mission profile (disaster type and severity known to the operator) plus the robot's sensed heat, water and rubble near its estimated pose. By construction it cannot see injected faults.
- **Q, observation quality.**
  - RGB: `√brightness · (0.5·contrast + 0.5·structure/(2 + 1.5·noise))`
  - Thermal: `(1 − saturated fraction) · √range factor · (0.4 + 0.6/(1 + noise²))`
  - LiDAR: `valid_ratio · (1 − adjacent-beam jump fraction)`
  - IMU: `exp(−rms(yaw_rate − rot_cmd)/0.08)`
  - Odometry: `exp(−EMA|d − d_cmd|/0.35) · exp(−EMA|Δθ − rot_cmd|/0.15)`
  - Audio: `1 − (noise_floor − 0.02)/0.45`
- **H, health.** A hard gate. Failure signatures:
  - RGB: black or frozen frame
  - Thermal: > 95% saturated
  - LiDAR: > 95% missing returns
  - IMU: bit-identical yaw while commanded to turn
  - Odometry: exactly 0 while commanded to move or turn
  - Audio: noise floor > 1

  Partial health values are given for partial saturation or dropout.
- **A, cross-modal agreement.** Computed only between related measurements:
  - RGB ↔ thermal candidates (shared field of view)
  - Audio bearing ↔ visual/thermal candidates and tracks
  - Odometry distance ↔ scan-match displacement
  - IMU rotation ↔ scan-match rotation (or ↔ odometry when there is no scan match)
  - LiDAR scan-to-map score

  A partner is used only while it is healthy (h > 0.5) and of usable quality (q > 0.3). Each term is an EMA (α = 0.25) that relaxes to 0.5 when there is no evidence.
- **Hybrid.** `r_s = h_s · (0.35·E + 0.45·Q + 0.20·A)` (weights sum to 1). It is smoothed as `r ← 0.7·r + 0.3·r_raw`, except that h = 0 sets r = 0 immediately.
- **Fusion weights** are normalised **per group**: localisation {LiDAR, IMU, odometry} and perception {RGB, thermal, audio}. If a group's sum is below 1e-6, the group is flagged `all_failed` and its consumers fall back (see below).
- **Ablation strategies:**
  - fixed: r = 1
  - env: r = E
  - quality: r = Q
  - env_quality: weighted E, Q
  - env_quality_health: h · (E, Q)
  - hybrid: everything above
  - learned (B6, optional): `h · MLP(E, Q, H, A, one-hot)`, trained on simulator labels

## 5. Sensor fusion (where reliability is used)
1. **EKF prediction.**
   - `Δθ = (r_imu·imu_rate + r_odo·odo_Δθ + c·rot_cmd)/(r_imu + r_odo + c)` and `d = (r_odo·odo_d + c·d_cmd)/(r_odo + c)`, with command prior c = 0.15.
   - Process noise grows as reliability drops: σ_d = 0.06 + 0.25(1 − r_odo)d and σ_θ = 0.01 + 0.12(1 − max(r_imu, r_odo))(|Δθ| + 0.2).
   - When both motion sensors fail, the commanded motion is the fallback.
2. **EKF update.**
   - The scan-matching pose is used as the measurement with `R = diag(0.05², 0.05², 0.02²)/(r_lidar·match_quality)`.
   - Fixed mode uses match_quality = 1 and r = 1, i.e. a conventional EKF.
3. **Occupancy mapping.** Log-odds increments are scaled by `min(1, r_lidar/0.6)` (1 in fixed mode).
4. **Hazard mapping.** RGB and thermal EMA rates are scaled by `min(1, r/0.6)`.
5. **Victim evidence.** Log-odds positive and negative evidence is scaled by r (§7).

## 6. Mapping and localisation
- **Occupancy grid.**
  - Log-odds with l_occ = 0.85 and l_free = −0.4, clipped to [−4, 4].
  - Each scan votes per cell: a cell receives occupied evidence only if at least as many beams end in it as pass through it. Hit points are pushed 0.1 cell past the measured range. Together these stop grazing beams at door jambs from closing doorways.
- **Scan matching.** Correlative search on an endpoint likelihood field:
  - The field is `exp(−d²/2·0.2²)`, where d is the distance to the occupied region at ×4 resolution, so a beam ending on a wall face scores 1.
  - A penalty of 0.5 is applied when a point 0.35 cell before the endpoint lies inside an occupied cell.
  - Coarse search covers ±0.75 to 1.5 cells and ±0.12 to 0.35 rad, scaled by the EKF covariance; a fine search follows.
- **Uncertainty layer.** `0.5·occupancy_entropy + 0.5·min(1, σ_xy/2)`, using σ_xy at the time the cell was observed.
- **Scope.** This is a SLAM-lite baseline (EKF + scan-to-map). It has no loop closure and is not state-of-the-art SLAM.

## 7. Victim detection and fusion
- **Detectors.**
  - RGB: an auto-gain image, then an HSV skin mask and connected components. A blob is accepted only if a ring of saturated non-skin clothing pixels surrounds it; confidence comes from the clothing fraction.
  - Thermal: compact blobs in the 30–40 °C band, below saturation, with fill ratio ≥ 0.35 and ≥ 3 °C contrast to a surrounding ring.
  - Audio: bearing plus range-from-intensity, with σ = 0.35·range + 1.
- **Association.** Candidates are converted to the map frame using the estimated pose. Association is gated nearest-neighbour (gate 2 + σ, capped at 4). Each track accepts at most one candidate per modality per step, so unrelated positions are never averaged.
- **Evidence.**
  - Detection: `L += r_m · conf · log(p_hit/p_fa)`.
  - Miss, when the track is inside modality m's field of view, ≤ 4 cells away and with line of sight on the estimated map: `L += r_m · 0.5 · log((1 − p_hit)/(1 − p_fa))`.
  - Per-frame rates measured on nominal frames: RGB p_hit 0.3 / p_fa 0.02; thermal 0.8 / 0.03; audio 0.5 / 0.15.
- **Confirmation.**
  - p ≥ 0.85 when two or more modalities support the track.
  - p ≥ 0.97 for a single imaging modality.
  - Audio alone never confirms.
  - A confirmation is counted (and rewarded) once.
- **Track position.** An inverse-variance update with variance σ²/r. Audio stops refining the position once a camera has seen the track.
- **Evaluation.** Confirmed tracks are greedily matched to ground-truth victims within 2 cells to give TP, FP and FN (precision, recall, F1).

## 8. Exploration
- **Frontier.** A known-free cell with an unknown 4-neighbour. Frontier cells are clustered with 8-connectivity; each cluster's target is the reachable member nearest its centroid. Path cost comes from one Dijkstra run on the costmap.
- **Information gain.** The number of unknown cells within radius 4 of the target.
- **Heuristics.**
  - random
  - nearest (minimum path cost)
  - info-gain: argmax of `(IG + 15·victim_conf)·exp(−0.08·cost)·(1 − 0.7·hazard)`
- **Planning and safety.** A* runs on the costmap; the safety controller can veto a step. Unreachable targets are blacklisted after two failures. Replanning happens every 12 steps.

## 9. RL design
- **Observation.** 228 floats, built only from estimates:
  - pose and uncertainty
  - progress
  - the six reliabilities
  - a 7×7 pooled local occupancy / unknown / hazard window
  - 8 frontier slots × 8 features
- **Action.** `Discrete(9)`: frontier slots ordered by path cost (slot 0 is the nearest), plus STOP. An empty slot maps to slot 0 with a −0.2 penalty.
- **Macro step.** Up to 15 low-level moves. It ends early on arrival, on a new confirmation, or when the episode ends.
- **Reward.**
  - `R = 1.0·ΔH/100 + 5·new_confirmed + 2·10·Δcoverage − 0.02·distance − 0.3·Σsensed_hazard − 1·bumper_contacts`
  - STOP while frontiers remain costs `2 + 5·(1 − coverage)`.
  - Every term is robot-side. Collisions come from a bumper contact the robot would physically feel. Hazard is the sensed hazard, not the ground truth.
- **Episode ends** on max steps (truncated), exploration complete, all expected victims confirmed with ≥ 80% estimated coverage, being unable to continue, or STOP.
- **Training.**
  - PPO with MlpPolicy [256, 128]: lr 3e-4, n_steps 1024, batch 64, γ 0.99, λ 0.95, clip 0.2.
  - Domain randomisation: disaster, severity U(0.2, 0.8), and with probability 0.6 one or two random sensor faults.
  - Training map seeds are in [0, 1e6); evaluation seeds start at 1e7.

## 10. Experimental methodology
- **Pairing.** Every method in an experiment runs on identical (disaster, map seed, degradation) tuples, with the same sensor noise seeds.
- **Summary statistics.** Mean ± 95% CI (1.96·sd/√n). Paired differences are reported with a bootstrap 95% CI and win/tie/loss counts.
- **Experiments.**
  - Exp 1: each sensor at "severe", plus nominal.
  - Exp 2: each sensor "failed".
  - Exp 3: disaster-specific simultaneous combinations, plus a mixed combination run on all disasters.
  - Exp 4: ablation B0–B5 (+ B6) over five scenarios.
  - Exp 5: E0–E4 plus info-gain with fixed fusion, on unseen maps under three conditions.
  - Exp 6: severities {0.2, 0.5, 0.8} × difficulties {easy, hard} on unseen maps.
- **Scope of claims.** All results are simulation measurements. Seeds and counts are in the CSVs under `results/logs/`.
