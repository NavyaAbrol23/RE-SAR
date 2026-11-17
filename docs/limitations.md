# Limitations: synthetic simulation vs real-world SAR

RE-SAR is a **controlled 2D simulation built for research and teaching**. Its results show how the algorithms behave relative to each other *inside this simulator*. They are not evidence of real-world SAR performance.

## Simulation fidelity
- **Geometry.** 2D, 1-cell resolution (roughly a robot footprint), static environment. There are no stairs, no 3D voids, no collapsing structure during the mission, and no moving victims or people.
- **Motion.** Rotate-then-drive kinematics with Gaussian actuation noise and a probabilistic slip model. There is no dynamics, traction physics or terrain mechanics.
- **Cameras.** Top-down egocentric abstractions, not perspective images. Victims are stylised colour shapes, and smoke, dust and darkness are simple blending, blur and noise. The RGB detector is colour/shape image processing; it would not work on real images.
- **Thermal.** A temperature field plus Gaussian blobs, with saturation modelled as a falling clip limit. There is no emissivity, reflection or atmospheric transmission model.
- **LiDAR.** Ideal ray casting plus noise, dropouts and spurious returns. There is no beam divergence, multi-echo or specular physics.
- **Audio.** A signal abstraction: inverse-square-like decay, a fixed per-wall attenuation and Gaussian DOA noise. It is not acoustic propagation, and there are no microphone arrays or speech detection.
- **Degradation model.** Hand-written, and the environment-prior estimator uses the same functional family as the simulator. This gives the "env" strategy a favourable model match that a real system would not have; only its *inputs* are estimated. The quality, health and cross-modal signals are model-agnostic and carry more of the general argument.
- **Failure signatures.** Each failure mode is idealised so that it can be detected: black frame, full saturation, all-NaN LiDAR, frozen IMU, zero odometry, saturated microphone. Real failures can be subtler (bias, intermittent faults, plausible but wrong data), and health detection would be harder.

## Algorithms
- **Localisation.** EKF + correlative scan-to-map matching, with no loop closure. Not state-of-the-art SLAM. The known start pose and 1-cell grid make the problem easier than real deployments.
- **Victim tracking.** Detection probabilities were set from measured per-frame rates *in this simulator*.
- **PPO.**
  - In development, PPO was trained for a modest number of timesteps on 2 CPU cores (exact figures are in `docs/experimental_results.md`). Results may change with the full 100k+ budget on a GPU machine.
  - The policy chooses among heuristic frontier candidates, which limits both what it can gain over strong heuristics and what it can lose.
  - It is a feed-forward MLP, so there is no memory under partial observability.
- **Learned reliability (B6).** Learns the simulator's own labels, so it measures calibration within the simulator only.

## Findings that are limitations of this design
- **LiDAR is a single point of failure for mapping.** With the LiDAR failed, no fusion strategy can map free space, so the robot cannot explore (Exp 2). A camera-based or depth fallback would be needed.
- **When every localisation sensor is degraded at once** (e.g. flood LiDAR + odometry), the hybrid model scales all of them down in absolute terms. The EKF then leans on the commanded-motion prior, and LiDAR map updates are slowed. In those cases it can be worse than fixed fusion (Exp 3, Exp 5 "LiDAR + thermal" condition). Normalising within the group inside the EKF is the obvious next step; it is not yet evaluated.
- **PPO did not outperform the frontier heuristics** with the budget used (25k steps).

## Evaluation
- **Sample sizes.** Small (a few seeds per condition × 3 disasters). Confidence intervals are reported, and overlapping intervals should be read as "no clear difference".
- **Map size.** 40×30 maps with 5 victims. Behaviour at larger scales was not tested.
- **Ground truth.** Used for evaluation, and to train the B6 calibrator offline. The robot never uses it at run time.

## What would be needed for real-world claims
- Physical or high-fidelity 3D simulation, with sensor models validated against hardware data.
- Real datasets for the detectors.
- Real fault injection.
- Field trials with SAR practitioners.
- Safety certification.
