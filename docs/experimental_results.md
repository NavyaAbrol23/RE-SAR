# RE-SAR Experimental Results (executed)

All numbers below are **simulation measurements**. They come from experiments actually run in the development environment:
- 2-core cloud CPU, no GPU
- Python 3.13, NumPy 2.5, Gymnasium 1.4, Stable-Baselines3 2.9.2, PyTorch 2.15 (CPU)

Raw per-mission CSVs, summaries, paired statistics and the full tables (`tables.md`) are in `results/logs/`, and the plots are in `results/plots/`. You can regenerate the tables with `python -m experiments.report`.

**Protocol.**
- Every method within an experiment ran on identical (disaster, map seed, degradation) tuples, with identical sensor-noise seeds.
- Evaluation map seeds start at 10,000,000, so they are never seen during PPO training (training seeds are below 1,000,000).
- Values are reported as mean ± 95% CI (1.96·sd/√n).
- "Paired diff" is the mean difference (B − A) over matched runs, with a 4,000-sample bootstrap 95% CI and win/tie/loss counts.
- Map size 40×30, 5 victims, medium difficulty, severity 0.5 unless stated otherwise.

| Experiment | Missions run | Seeds × disasters |
|---|---|---|
| Exp 1–3 sensor degradation / failure | 360 | 4 × 3 |
| Exp 4 fusion ablation (B0–B6) | 450 | 3 × 3 × 5 scenarios |
| Exp 5 exploration (E0–E4) | 324 | 6 × 3 × 3 conditions |
| Exp 6 generalisation | 360 | 4 × 3 × 3 severities × 2 difficulties |
| `rl.evaluate` (RL reward on unseen maps) | 60 episodes | 12 per policy |

---

## Headline findings

1. **Single-sensor failure: hybrid reliability is clearly and consistently better (Exp 2).** Against fixed weights it changes:
   - victim F1 by **+0.26** [+0.17, +0.35]
   - localisation ATE by **−5.1 cells** [−7.0, −3.4]
   - coverage by **+0.14** [+0.09, +0.20]
   - collisions by **−6.4 per mission** [−9.4, −3.8]

   It flags failed sensors (r < 0.15) on 99–100% of steps; fixed fusion detects none. The difference is decisive for IMU and odometry failure: fixed-weight localisation diverges by 13–18 cells and the robot collides 13–25 times per mission.
2. **Single-sensor severe degradation: smaller but significant gains (Exp 1).** F1 **+0.09** [+0.02, +0.17]; ATE **−0.89 cells** [−1.59, −0.29]. Coverage and collisions show no significant difference.
3. **Simultaneous degradation: no overall advantage (Exp 3).** The paired F1 difference is +0.01 [−0.10, +0.14]. In **flood (LiDAR + odometry + audio severe)** hybrid is *worse*: F1 0.13 vs 0.40, ATE 3.0 vs 1.8. When every localisation sensor is degraded at once, the hybrid model scales them all down *in absolute terms*, which hands control to the commanded-motion prior. This is a genuine limitation of the current design (see "Negative results").
4. **Health gating carries most of the benefit (Exp 4 ablation).**
   - The environment prior alone (B2) and environment + quality (B4) are not better than fixed weights.
   - Adding sensor health (B4+H) gives F1 +0.14 and ATE −2.0 cells against fixed.
   - Adding cross-modal agreement (B5 hybrid) gives no further significant F1 gain. It gives slightly better ATE and coverage, but more collisions (0.40 vs 0.07).
   - The **learned** calibrator (B6) matches hybrid F1, has the best reliability calibration (MAE 0.06 vs 0.14), and has a worse ATE (0.55) and more collisions (1.49).
5. **PPO does not beat strong exploration heuristics (Exp 5, 6).** On unseen maps PPO/hybrid reaches 0.80 coverage and 2.69 victims, against 0.83 and 2.81 for info-gain. The paired coverage difference is −0.03 [−0.06, +0.01] (not significant), and PPO's path length is +34 cells [+10, +58] longer.
6. **Reliability in the RL loop matters (Exp 5).** PPO/hybrid vs PPO/fixed, paired:
   - coverage **+0.12** [+0.03, +0.21]
   - victims **+0.83** [+0.35, +1.35]
   - F1 **+0.20** [+0.09, +0.31]
   - collisions **−6.6** [−11.2, −2.7]

   The gain comes from the "degraded" condition (victims +2.8). Info-gain with fixed fusion shows the same drop as PPO/fixed, so most of the gap comes from fusion rather than from the learned policy.

---

## Exp 1: individual sensor degradation (each sensor "severe"), fixed vs hybrid

| scenario | F1 fixed | F1 hybrid | ATE fixed | ATE hybrid | coverage fixed | coverage hybrid | collisions fixed | collisions hybrid |
|---|---|---|---|---|---|---|---|---|
| nominal | 0.79 ± 0.11 | 0.76 ± 0.16 | 0.10 ± 0.01 | 0.08 ± 0.00 | 0.91 ± 0.04 | 0.88 ± 0.12 | 0.00 | 0.00 |
| rgb_severe (visibility) | 0.63 ± 0.07 | 0.72 ± 0.16 | 0.12 ± 0.02 | 0.08 ± 0.00 | 0.88 ± 0.04 | 0.85 ± 0.12 | 0.00 | 0.00 |
| thermal_severe (saturation) | 0.37 ± 0.14 | 0.45 ± 0.13 | 0.10 ± 0.01 | 0.08 ± 0.00 | 0.85 ± 0.14 | 0.89 ± 0.13 | 0.00 | 0.00 |
| lidar_severe (missing returns) | 0.45 ± 0.20 | 0.53 ± 0.19 | 1.17 ± 0.59 | 1.11 ± 0.64 | 0.67 ± 0.16 | 0.73 ± 0.12 | 0.08 ± 0.16 | 0.50 ± 0.66 |
| imu_severe (drift) | 0.37 ± 0.19 | 0.60 ± 0.22 | 6.71 ± 3.25 | 0.85 ± 1.31 | 0.69 ± 0.14 | 0.72 ± 0.16 | 2.33 ± 1.81 | 0.25 ± 0.35 |
| odometry_severe (slip) | 0.63 ± 0.19 | 0.78 ± 0.16 | 0.37 ± 0.32 | 0.16 ± 0.16 | 0.80 ± 0.10 | 0.87 ± 0.10 | 0.08 ± 0.16 | 0.00 |
| audio_severe (noise) | 0.64 ± 0.17 | 0.71 ± 0.15 | 0.10 ± 0.01 | 0.08 ± 0.00 | 0.84 ± 0.09 | 0.87 ± 0.12 | 0.00 | 0.00 |

Reliability-estimate MAE against the simulator's true reliability is 0.13–0.18 for hybrid, versus 0.16–0.27 for fixed (whose estimate is always 1). Under nominal conditions hybrid is slightly lower on F1 (0.76 vs 0.79, not significant): down-weighting evidence makes confirmation a little slower.

Plots: `exp1_f1.png`, `exp1_ate.png`, `exp1_coverage.png`, `exp1_rel_mae.png`.

## Exp 2: complete failure of each sensor

| scenario | F1 fixed | F1 hybrid | ATE fixed | ATE hybrid | coverage fixed | coverage hybrid | collisions fixed | collisions hybrid | failure detection (hybrid) |
|---|---|---|---|---|---|---|---|---|---|
| rgb_failed | 0.78 | 0.78 | 0.12 | 0.09 | 0.86 | 0.94 | 0 | 0 | 1.00 |
| thermal_failed | 0.50 | 0.53 | 0.12 | 0.08 | 0.96 | 0.95 | 0 | 0 | 1.00 |
| lidar_failed | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 1.00 |
| imu_failed | 0.04 | **0.76** | 17.67 | **0.09** | 0.67 | **0.93** | 13.3 | **0.0** | 0.99 |
| odometry_failed | 0.04 | **0.83** | 12.93 | **0.08** | 0.47 | **0.91** | 25.3 | **0.0** | 0.99 |
| audio_failed | 0.74 | 0.75 | 0.10 | 0.08 | 0.87 | 0.92 | 0 | 0 | 1.00 |

Fixed fusion has a failure-detection rate of 0 in every row. **With the LiDAR failed, neither method explores at all.** LiDAR is the only mapping sensor, so no free space is ever mapped, no frontier exists and the robot stays at the start. Hybrid detects the failure, but no fallback mapping modality exists. This is a real single-point-of-failure finding (see `limitations.md`).

Plots: `exp2_f1.png`, `exp2_ate.png`, `exp2_coverage.png`, `exp2_rel_mae.png`.

## Exp 3: simultaneous degradation

| scenario | F1 fixed | F1 hybrid | ATE fixed | ATE hybrid | coverage fixed | coverage hybrid | collisions fixed | collisions hybrid |
|---|---|---|---|---|---|---|---|---|
| fire: RGB + thermal | 0.29 ± 0.21 | 0.32 ± 0.22 | 0.08 | 0.11 | 0.74 | 0.77 | 0.0 | 0.0 |
| earthquake: LiDAR + odometry | 0.22 ± 0.16 | 0.45 ± 0.37 | 5.46 ± 4.28 | 3.22 ± 2.25 | 0.70 | 0.73 | 2.75 | 1.25 |
| flood: LiDAR + odometry + audio | **0.40 ± 0.20** | 0.13 ± 0.15 | **1.84 ± 0.83** | 3.03 ± 1.52 | 0.75 | 0.84 | 1.00 | 2.50 |
| mixed: RGB + LiDAR + thermal | 0.18 ± 0.12 | 0.21 ± 0.14 | 1.06 ± 0.45 | 2.03 ± 1.44 | 0.76 | 0.77 | 0.00 | 1.75 |

Paired over all of Exp 3 (n = 24):
- F1: +0.01 [−0.10, +0.14]
- ATE: +0.31 [−0.83, +1.47]
- coverage: +0.03 [−0.05, +0.10]

None of these is significant. Hybrid's reliability MAE is still much lower (0.14–0.23 vs 0.34–0.45): it *knows* the sensors are bad, but with every localisation source degraded at once it has nothing trustworthy to shift weight to.

## Exp 4: fusion ablation (5 scenarios: nominal, thermal_failed, rgb_failed+audio_severe, odometry_failed+lidar_mild, mixed_severe)

| method | F1 | precision | recall | ATE | coverage | collisions | reliability MAE | ECE | failure detection |
|---|---|---|---|---|---|---|---|---|---|
| B0 rgb_only | 0.20 | 0.34 | 0.16 | 0.34 | 0.89 | 0.11 | 0.13 | 0.09 | 1.00 |
| B0 thermal_only | 0.36 | 0.38 | 0.34 | 0.47 | 0.87 | 0.36 | 0.13 | 0.09 | 1.00 |
| B0 audio_only | 0.00 | 0.00 | 0.00 | 0.27 | 0.89 | 0.07 | 0.14 | 0.08 | 1.00 |
| B1 fixed | 0.42 | 0.49 | 0.41 | 2.41 | 0.82 | 3.84 | 0.33 | 0.33 | 0.00 |
| B2 env | 0.40 | 0.49 | 0.37 | 2.33 | 0.81 | 4.11 | 0.19 | 0.18 | 0.00 |
| B3 quality | 0.45 | 0.55 | 0.42 | 1.99 | 0.81 | 0.07 | 0.17 | 0.12 | 0.97 |
| B4 env_quality | 0.41 | 0.48 | 0.38 | 2.43 | 0.82 | 3.02 | 0.18 | 0.11 | 0.05 |
| B4+H env_quality_health | **0.56** | **0.69** | 0.52 | 0.39 | 0.83 | **0.07** | 0.12 | 0.08 | 1.00 |
| B5 hybrid | **0.56** | 0.65 | 0.52 | **0.32** | **0.87** | 0.40 | 0.14 | 0.08 | 1.00 |
| B6 learned (MLP) | **0.56** | 0.63 | **0.54** | 0.55 | 0.82 | 1.49 | **0.06** | **0.04** | 1.00 |

Paired against B1 fixed (n = 45):

| method | F1 diff | ATE diff (cells) |
|---|---|---|
| B4+H | +0.14 [+0.05, +0.24] | −2.01 [−3.44, −0.79] |
| B5 hybrid | +0.13 [+0.05, +0.22] | −2.08 [−3.51, −0.85] |
| B6 learned | +0.14 [+0.06, +0.22] | −1.86 [−3.32, −0.57] |
| B2 env, B3 quality, B4 env+quality | all F1 diffs straddle 0 | – |

**Reading.** The B0 single-modality baselines use hybrid localisation, so their low ATE reflects localisation, not the modality. Audio alone never confirms a victim, by design. Thermal alone is the best single modality (F1 0.36). Fusing modalities with health-gated reliability gives F1 0.56.

Plots: `exp4_ablation_f1.png`, `exp4_ablation_ate.png`, `exp4_ablation_rel_mae.png`, `exp4_ablation_collisions.png`, `exp4_f1_by_scenario.png`.

**Learned reliability (B6) training.**
- 45 missions, 56,418 per-sensor samples, 40 epochs, CPU, 188 s.
- Held-out-row MAE: 0.042 (MLP) vs 0.135 (hybrid formula without smoothing).
- These rows come from the same missions as training. The out-of-sample evidence is the ablation above, run on unseen seeds, where the MLP had MAE 0.06.
- It learns this simulator's degradation model.

## PPO training

| model | timesteps | device | wall time | episodes | final rolling reward |
|---|---|---|---|---|---|
| ppo_hybrid | 25,000 | CPU | 5,597 s* | 1,062 | 28.9 |
| ppo_fixed | 25,000 | CPU | 5,627 s* | 1,064 | 28.4 |

\* Measured while four jobs shared two cores. Alone, training runs at about 7.8 decisions/s, roughly 55 min for 25k steps.

- Reward curves (`reward_curve_ppo_hybrid.png`, `reward_curve_ppo_fixed.png`) rise from about 0–5 to about 28 by 10k steps, then plateau.
- The budget is **25k timesteps, a quarter of the 100k in the brief**, because of sandbox compute. The README gives the command for a 100k-step GPU run.

## Exp 5: exploration comparison (unseen maps; conditions: nominal / degraded = RGB severe + odometry failed + audio mild / LiDAR severe + thermal failed)

| method | coverage | victims found (of 5) | F1 | path length | steps | collisions | hazard exposure | coverage per 100 cells | success (5/5) | ATE |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 random | 0.78 | 2.43 | 0.54 | 259.9 | 339.8 | 0.06 | 35.2 | 4.33 | 0.15 | 0.17 |
| E1 nearest | **0.83** | **2.81** | **0.61** | 209.4 | 283.2 | 0.02 | 29.7 | 4.44 | **0.24** | 0.23 |
| E2 info-gain | **0.83** | **2.81** | 0.60 | **198.2** | **263.9** | 0.31 | 30.2 | **4.48** | 0.20 | 0.31 |
| E2f info-gain / fixed fusion | 0.74 | 1.81 | 0.37 | 200.1 | 308.3 | 8.07 | 33.5 | 4.20 | 0.11 | 4.73 |
| E3 PPO / fixed | 0.68 | 1.85 | 0.38 | 192.9 | 291.8 | 7.06 | 32.1 | 4.20 | 0.09 | 3.65 |
| E4 PPO / hybrid | 0.80 | 2.69 | 0.58 | 232.5 | 302.6 | 0.41 | **29.3** | 4.40 | 0.15 | 0.31 |

Coverage by condition:

| method | nominal | degraded | LiDAR + thermal |
|---|---|---|---|
| E0 random | 0.82 | 0.85 | 0.67 |
| E1 nearest | 0.91 | 0.91 | 0.67 |
| E2 info-gain | 0.90 | 0.87 | 0.71 |
| E2f info-gain/fixed | 0.89 | 0.52 | **0.81** |
| E3 PPO/fixed | 0.90 | 0.38 | 0.77 |
| E4 PPO/hybrid | 0.91 | 0.87 | 0.63 |

Paired comparisons (n = 54):
- **E4 PPO/hybrid vs E2 info-gain:** coverage −0.03 [−0.06, +0.01], victims −0.13 [−0.37, +0.11], path +34 cells [+10, +58]. PPO did **not** outperform the heuristic, and travelled further.
- **E4 PPO/hybrid vs E3 PPO/fixed:** coverage +0.12 [+0.03, +0.21], victims +0.83 [+0.35, +1.35], F1 +0.20 [+0.09, +0.31], collisions −6.6 [−11.2, −2.7], ATE −3.3 [−5.0, −1.9]. By condition, victims: nominal +0.00, degraded **+2.83** [+2.06, +3.56], LiDAR + thermal −0.33 [−1.00, +0.33].
- In the LiDAR-severe condition, the fixed-fusion variants cover more than the hybrid ones (0.77–0.81 vs 0.63–0.71). This matches Exp 3: hybrid scales down LiDAR map updates (`min(1, r/0.6)`), so the map grows more slowly when LiDAR is degraded.

**`rl.evaluate`, RL episode reward** (12 unseen episodes, random disaster/severity/fault):

| policy | reward | coverage | victims |
|---|---|---|---|
| nearest | 32.2 | 0.82 | 3.25 |
| info-gain | 31.6 | 0.79 | 3.42 |
| PPO/fixed | 29.4 | 0.72 | 3.17 |
| PPO/hybrid | 28.1 | 0.78 | 3.00 |
| random | 22.2 | 0.75 | 2.83 |

Even on its own reward function, PPO stays below the two strong heuristics.

## Exp 6: generalisation (unseen maps; severity 0.2/0.5/0.8 × easy/hard; LiDAR + RGB mild)

| coverage | 0.2 | 0.5 | 0.8 |
|---|---|---|---|
| E1 nearest | 0.76 | 0.83 | 0.85 |
| E2 info-gain | 0.76 | 0.86 | 0.85 |
| E2f info-gain/fixed | 0.78 | 0.82 | 0.82 |
| E3 PPO/fixed | 0.76 | 0.80 | 0.79 |
| E4 PPO/hybrid | 0.78 | 0.79 | 0.80 |

| F1 | 0.2 | 0.5 | 0.8 |
|---|---|---|---|
| E1 nearest | 0.40 | 0.37 | 0.48 |
| E2 info-gain | 0.41 | 0.46 | 0.50 |
| E2f info-gain/fixed | 0.32 | 0.30 | 0.33 |
| E3 PPO/fixed | 0.27 | 0.23 | 0.27 |
| E4 PPO/hybrid | 0.38 | 0.41 | 0.40 |

- **Severity.** All methods degrade gracefully across severity and difficulty (hard maps cost about 0.05–0.06 coverage).
- **Fixed fusion vs reliability-aware.** Under the persistent mild LiDAR + RGB degradation, the fixed-fusion variants have consistently lower F1 (0.23–0.33 vs 0.37–0.50). They find a similar *number* of victims, but with more false positives.
- **Counter-intuitive coverage trend.** Coverage rises with severity because more severe earthquakes and floods break more walls, which makes maps more open.
- **PPO.** PPO stays at or slightly below the heuristics at every severity.

Plots: `exp6_coverage_vs_severity.png`, `exp6_f1_vs_severity.png`, `exp6_victims_found_vs_severity.png`.

## Computational metrics (`results/logs/computational.json`)

| quantity | value |
|---|---|
| Perception–action step (all six sensors, reliability, EKF, scan match, mapping, perception, planning) | 11.9 ms on 1 CPU core (≈ 84 steps/s) |
| PPO policy inference | 1.1 ms per decision (idle CPU) |
| Peak Python memory, one mission | 5.7 MB (tracemalloc) |
| GUI | 16–17 FPS at speed 2–5 in headless rendering |
| PPO training | ≈ 7.8 decisions/s per process (CPU), dominated by simulation, not network updates |

## Negative results and threats to validity

- **PPO < heuristics.** With 25k steps, an MLP policy and macro-actions over the same frontier candidates the heuristics see, PPO learned a reasonable policy (reward 4 → 28) but did not beat nearest-frontier or info-gain. Plausible reasons:
  - a small training budget
  - no memory, so no handling of partial observability
  - frontier candidates that are already heuristically good

  A 100k-step GPU run is the next test.
- **Hybrid under all-localisation-sensor degradation** (flood combination, LiDAR-severe conditions) can be worse than fixed fusion. Two fixes are not yet tested:
  - normalise motion-sensor weights *within* the localisation group in the EKF
  - stop scaling map updates by absolute LiDAR reliability
- **LiDAR is a single point of failure** for mapping, under every fusion strategy.
- **Model match.** The environment prior uses the simulator's own degradation family, with estimated inputs. It still did not help on its own (B2 ≈ B1).
- **Statistics.** Sample sizes are small. Where a CI crosses 0 above, the result is a "no clear difference", not a negative claim.

## UCS760 learning-outcome mapping

| Course area | Where it is demonstrated |
|---|---|
| RL: environment design | `rl/sar_env.py`: Gymnasium env (passes `check_env`), observation, action and reward design, termination |
| RL: PPO and TD learning | `rl/train.py`: SB3 PPO (clipped policy gradient, GAE-λ advantage estimation with bootstrapped TD targets), training curves, checkpointing |
| RL: evaluation | `rl/evaluate.py`, Exp 5–6: unseen maps, paired baselines, an honest negative result |
| Computer vision | `sensors/rgb.py`, `perception/rgb_detection.py` (auto-gain, HSV segmentation, morphology, connected components, context ring), `perception/thermal_detection.py` (band thresholding, blob contrast), `reliability/observation_quality.py` (contrast, Laplacian structure, saturation) |
| Robotics: navigation and mapping | EKF localisation, correlative scan matching, log-odds occupancy grid, costmap, A*, frontier exploration, safety controller |
| Sensor fusion | Hybrid reliability, group-normalised weights, reliability-weighted EKF and Bayesian victim tracking |
| Autonomous decision-making | The closed perception–action loop in `core/mission.py`; PPO frontier selection |
| GPU computing | `--device auto` in PPO and B6 training (CUDA when available, CPU fallback). The dev sandbox had no GPU; GPU runs are listed in the README's verification steps |
