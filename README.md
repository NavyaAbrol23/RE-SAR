# RE-SAR: Reliability-Aware Autonomous Exploration for Search and Rescue in Sensor-Degraded Environments

UCS760 — Edge AI and Robotics: Reinforcement Learning & Conversational AI

RE-SAR is a single autonomous search-and-rescue robot in a **custom 2D Python simulation**. It explores procedurally generated earthquake, fire and flood buildings with six simulated sensors (RGB, thermal, LiDAR, IMU, wheel odometry, audio). The project studies one question:

> **Can hybrid reliability-aware sensor fusion make autonomous SAR exploration more robust than conventional fixed-weight fusion when sensors are degraded individually or several at once?**

> **Honesty note.** This is a controlled 2D simulation with synthetic sensor models. It is not a physically validated SAR robot, and none of its numbers are real-world performance claims. See [`docs/limitations.md`](docs/limitations.md).

---

## 1. Project overview

The robot runs a closed perception–action loop at every timestep:

1. Read six sensors.
2. Assess each sensor's health and observation quality.
3. Compute reliability scores.
4. Update localization (EKF + scan matching) and the occupancy map.
5. Detect, associate and track victims.
6. Update the shared uncertainty-aware map.
7. Choose an exploration target (heuristic or PPO).
8. Plan a route with A* and pass it through the safety controller.
9. Move.

Everything runs locally on CPU. A GPU (e.g. an RTX 4060) is used automatically for PPO and the learned-reliability model when available.

## 2. Research problem

Disaster sites degrade sensors in different ways:
- smoke and darkness blind cameras
- heat saturates thermal imagers
- dust and water corrupt LiDAR
- rubble and water make wheels slip
- noise masks voices

Conventional fusion with fixed weights keeps trusting a sensor after it has gone bad. RE-SAR estimates each sensor's reliability online and uses it everywhere fusion happens: localization, mapping, hazard mapping and victim evidence. It also feeds those reliabilities to the RL exploration policy.

## 3. Research objectives

- **O1.** Simulate three disaster profiles and six sensors with configurable, simultaneous degradation (normal, mild, severe, failed).
- **O2.** Estimate per-sensor reliability from four interpretable sources: environment prior, observation quality, sensor health and cross-modal agreement.
- **O3.** Keep **localization reliability** (LiDAR / IMU / odometry) separate from **victim-perception reliability** (RGB / thermal / audio).
- **O4.** Compare fixed-weight fusion, each reliability component (ablation) and the full hybrid model under identical, paired conditions.
- **O5.** Train PPO to select exploration frontiers, and compare it against random, nearest-frontier and information-gain exploration on unseen maps.

## 4. Main contributions (scope: this simulation)

- A transparent **hybrid reliability model**, `r = h · (wE·E + wQ·Q + wA·A) / (wE+wQ+wA)`, with a hard health gate. Weights are normalized separately within the localization group and the perception group, with a safe fallback when a whole group fails.
- **Reliability-weighted EKF + scan matching.** Fused motion increments, adaptive process noise, and a LiDAR-quality-scaled measurement noise. It falls back to commanded motion when every motion sensor has failed.
- **Reliability-weighted Bayesian victim tracking.** Gated association, positive and negative evidence scaled by r, single-modality and audio-only confirmation guards, and one reward per victim.
- **Gymnasium environment + PPO** that choose among frontier targets, with reliability scores in the observation.
- Reproducible paired experiments (Exp 1–6), plus an optional learned-reliability baseline (B6, PyTorch).

## 5. System architecture

```
A  Disaster environment (ground truth: map, hazards, victims, robot pose)  -- environment/
        |  (only sensors, rendering and evaluation may read ground truth)
B  Six sensor generators + degradation controller                          -- sensors/
        |
C  Reliability: environment prior, quality, health, cross-modal -> hybrid  -- reliability/
        |                                   |
D1 Localisation (EKF + scan match)   D2 Victim detection (RGB/thermal/audio)
   + log-odds occupancy mapping         + association + reliability-weighted tracks
   -- mapping/                          -- perception/
        \___________________________________/
E  Shared uncertainty-aware map (occupancy, traversability, hazard,
   victim confidence, unexplored, uncertainty)                             -- mapping/shared_map.py
        |
F  Exploration: random / nearest / info-gain frontier, or PPO (Gymnasium)  -- navigation/, rl/
        |
G  A* on the costmap + safety controller                                   -- navigation/
        |
H  Pygame dashboard                                                        -- visualization/
   (core/mission.py runs the loop for the GUI, the RL env and the experiments)
```

## 6. Technical stack

Python 3.11+ (tested on 3.13), NumPy, Pandas, PyYAML, Pygame, OpenCV, Matplotlib, Gymnasium, Stable-Baselines3, PyTorch and Pytest. There are no simulators, ROS, cloud APIs or downloaded datasets, and nothing needs internet access at runtime.

## 7. Installation (Windows)

```bat
:: 1. Python 3.11 (python.org). In the RE-SAR folder:
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip

:: 2. (GPU, RTX 4060) install the CUDA build of PyTorch FIRST.
::    Pick the exact command for your driver at https://pytorch.org/get-started/locally/ ; e.g.
pip install torch --index-url https://download.pytorch.org/whl/cu126
::    (skip this step for CPU-only; requirements.txt then installs the CPU build)

:: 3. Everything else
pip install -r requirements.txt

:: 4. Check the GPU is visible (should print True on the RTX 4060)
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Linux and macOS use the same steps with `python3 -m venv .venv && source .venv/bin/activate`.

## 8. Environment setup / configuration

- `config/default.yaml` holds the world size, sensors, degradation levels, reliability weights, mapping, perception, navigation, reward coefficients and PPO hyper-parameters.
- `config/earthquake.yaml`, `fire.yaml` and `flood.yaml` hold the disaster-specific environment parameters (dust, smoke, heat, water, acoustic noise, wheel slip, LiDAR noise, audio attenuation). All values are normalized to [0, 1].
- Seeds make every map, sensor stream and mission reproducible.

## 9. Launch the simulation

```bat
python main.py                                   :: menu -> choose disaster, difficulty, severity, victims, seed
python main.py --disaster fire --seed 7 --start  :: skip the menu
python main.py --nav ppo --fusion hybrid --start :: PPO exploration (uses rl/models/ppo_hybrid.zip)
python main.py --headless-demo 300 --disaster flood --degrade rgb=failed,lidar=severe   :: no window; saves screenshots
```

**Controls:**

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `N` | Single step |
| `R` | Reset |
| `M` | New map |
| `V` | Toggle ground-truth / robot-estimate view |
| `L` | LiDAR overlay |
| `P` | Cycle navigation mode |
| `F` | Cycle fusion strategy |
| `+` / `-` | Speed |
| `1`–`6` | Cycle degradation of RGB / thermal / LiDAR / IMU / odometry / audio |
| `ESC` | Back to menu |

Every key also has an on-screen button. Click a sensor's level button to inject a fault.

## 10. Train PPO

```bat
python -m rl.train --reliability hybrid --timesteps 100000 --n-envs 4 --device auto   :: E4
python -m rl.train --reliability fixed  --timesteps 100000 --n-envs 4 --device auto   :: E3
```

- `--device auto` picks CUDA when PyTorch sees the GPU.
- The simulation runs on CPU, so `--n-envs` (parallel worker processes) matters more for speed than the GPU does.
- Outputs:
  - `rl/models/ppo_<rel>.zip`
  - checkpoints in `results/checkpoints/`
  - Monitor logs and `training_stats.json` in `results/logs/ppo_<rel>/`
  - the reward curve in `results/plots/reward_curve_ppo_<rel>.png`

## 11. Load a trained PPO model

- **GUI:** set Navigation to `ppo` in the menu or press `P`. The app loads `rl/models/ppo_hybrid.zip`, or `ppo_fixed.zip` when fusion is `fixed`. With no model it says so and falls back to info-gain.
- **Evaluate on unseen maps:** `python -m rl.evaluate --episodes 12`
- **In code:** `from rl.policy import PPOChooser; Mission(cfg, nav="ppo", chooser=PPOChooser("rl/models/ppo_hybrid.zip"))`

## 12. Run the experiments

```bat
python -m experiments.sensor_failure --seeds 4          :: Exp 1-3 (individual, failure, simultaneous)
python -m reliability.learned --missions 60 --device auto :: (optional) trains B6 learned reliability
python -m experiments.fusion_ablation --seeds 3         :: Exp 4 (B0-B5, + B6 if trained)
python -m experiments.exploration_comparison --seeds 6  :: Exp 5 (E0-E4) + computational metrics
python -m experiments.generalization --seeds 4          :: Exp 6 (severity x difficulty, unseen maps)
python -m pytest -q                                     :: unit + integration + headless GUI tests
```

- CSVs, summaries and paired statistics go to `results/logs/`. Plots go to `results/plots/`.
- All methods in an experiment run on the **same maps, seeds and degradations** (paired comparisons).
- Evaluation seeds (10,000,000+) never overlap PPO training seeds.

## 13. Interpreting the visualization

- **World view.**
  - Ground truth is shown by default; press `V` for the robot's estimate.
  - Robot pose: white filled triangle = true pose; cyan outline = EKF estimate. The thin lines are their trajectories.
  - Victims: white rings marked "GT" are ground-truth victims (evaluation only). Yellow rings are tentative tracks; green dots marked V# are confirmed detections.
  - Navigation: orange squares are frontiers (the filled one is selected). The magenta line is the A* path; the magenta X is the target.
  - Green dots are LiDAR returns.
- **Sensor panel.**
  - Previews: RGB, thermal, and a LiDAR polar view.
  - Per sensor:
    - the reliability bar
    - a white tick marking the simulator's true reliability
    - E/Q/H/A components
    - the health label (OK / DEGRADED / FAILED)
    - live observation status
    - a clickable degradation level
  - The normalized fusion weights for the localization and perception groups are shown at the top.
- **Shared map:** occupancy, uncertainty, sensed hazard, victim confidence, exploration coverage and traversability cost.
- **Exploration / metrics.**
  - Navigation mode, the deciding policy and the selected frontier.
  - Per-step and cumulative reward.
  - Coverage, victims found, P/R/F1, current localization error and ATE, collisions, hazard exposure, and reliability MAE.

## 14. Evaluation metrics

| Area | Metrics |
|---|---|
| Localization | ATE (RMSE), final position error, mean heading error |
| Mapping | Occupancy accuracy (known cells), coverage of reachable area |
| Reliability | MAE and ECE vs simulator ground-truth reliability; failure-detection rate (r < 0.15 for a failed sensor) |
| Detection | Precision, recall, F1, victims found, time to discovery |
| Exploration / navigation | Coverage, path length, efficiency, mission success, collisions, hazard exposure, safety vetoes |
| Computational | Steps/s, per-step latency, PPO inference latency, training time, peak Python memory |

## 15. Limitations

See [`docs/limitations.md`](docs/limitations.md). In short:
- The geometry is 2D and 1-cell resolution.
- The sensor models are synthetic, with a hand-written degradation model.
- There is no real acoustics, optics or SLAM back-end.
- PPO was trained with a modest budget in development. The results doc reports exactly what was run.

## 16. Future improvements

- Continuous-space dynamics with a particle filter or graph SLAM.
- Learned detectors trained on real thermal and RGB datasets.
- Recurrent PPO (partial observability).
- Multi-robot coordination.
- ROS 2 bridge to a physical platform.
- Online learning of fusion weights.
- A conversational operator interface (out of scope here).

## Results at a glance (executed in simulation; details in `docs/experimental_results.md`)

All results are paired comparisons on unseen seeds, with 95% bootstrap CIs. Models included: `rl/models/ppo_hybrid.zip` and `ppo_fixed.zip` (25k steps each, CPU) and `reliability_mlp.npz` (B6).

| Question | Result |
|---|---|
| Single-sensor **failure**, hybrid vs fixed fusion | F1 **+0.26** [0.17, 0.35], ATE **−5.1 cells**, coverage **+0.14**, collisions **−6.4/mission**; failures detected on 99–100% of steps |
| Single-sensor **severe** degradation | F1 +0.09 [0.02, 0.17], ATE −0.89 cells |
| **Simultaneous** degradation | No significant difference; hybrid is worse for flood (LiDAR + odometry + audio). Reported as a limitation |
| Ablation: which component matters? | **Sensor health gating** (B4+H ≈ B5 hybrid ≈ B6 learned, F1 ≈ 0.56 vs fixed 0.42). The environment prior alone adds nothing |
| PPO/hybrid vs PPO/fixed | Coverage **+0.12**, victims **+0.83**, F1 **+0.20**, collisions **−6.6** |
| PPO vs heuristics | PPO did **not** beat nearest-frontier or info-gain (coverage −0.03, n.s.; longer paths) |

## UCS760 course alignment

- **RL:** Gymnasium env, PPO (GAE / TD targets), reward design, evaluation on unseen maps.
- **Computer vision:** RGB and thermal processing, multi-modal victim detection.
- **Robotics:** EKF, scan matching, occupancy grids, A*, frontiers, safety, sensor fusion.
- **GPU:** CUDA auto-selection with CPU fallback.

The full mapping is in `docs/experimental_results.md`. Conversational AI was optional and was not implemented.

## Project structure

See [`docs/technical_specification.md`](docs/technical_specification.md) for every module's inputs and outputs, [`docs/methodology.md`](docs/methodology.md) for the equations, and [`docs/experimental_results.md`](docs/experimental_results.md) for executed results.

## Verify on your own machine (RTX 4060 laptop)

These could not be checked in the development sandbox, which had no display and no GPU:

1. `python main.py`: the window opens, the menu works, and Start runs a mission. Click sensor levels, Pause/Step, View, Nav and Fusion.
2. `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
3. `python -m rl.train --reliability hybrid --timesteps 100000 --n-envs 4 --device cuda`: the log line shows `device=cuda`.
4. Re-run Exp 5 with the stronger models: `python -m experiments.exploration_comparison --seeds 6`.
