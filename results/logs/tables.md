
### exp1_individual

| scenario | f1 fixed | f1 hybrid | ate fixed | ate hybrid | coverage fixed | coverage hybrid | collisions fixed | collisions hybrid |
|---|---|---|---|---|---|---|---|---|
| audio_severe | 0.64 ± 0.17 | 0.71 ± 0.15 | 0.10 ± 0.01 | 0.08 ± 0.00 | 0.84 ± 0.09 | 0.87 ± 0.12 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| imu_severe | 0.37 ± 0.19 | 0.60 ± 0.22 | 6.71 ± 3.25 | 0.85 ± 1.31 | 0.69 ± 0.14 | 0.72 ± 0.16 | 2.33 ± 1.81 | 0.25 ± 0.35 |
| lidar_severe | 0.45 ± 0.20 | 0.53 ± 0.19 | 1.17 ± 0.59 | 1.11 ± 0.64 | 0.67 ± 0.16 | 0.73 ± 0.12 | 0.08 ± 0.16 | 0.50 ± 0.66 |
| nominal | 0.79 ± 0.11 | 0.76 ± 0.16 | 0.10 ± 0.01 | 0.08 ± 0.00 | 0.91 ± 0.04 | 0.88 ± 0.12 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| odometry_severe | 0.63 ± 0.19 | 0.78 ± 0.16 | 0.37 ± 0.32 | 0.16 ± 0.16 | 0.80 ± 0.10 | 0.87 ± 0.10 | 0.08 ± 0.16 | 0.00 ± 0.00 |
| rgb_severe | 0.63 ± 0.07 | 0.72 ± 0.16 | 0.12 ± 0.02 | 0.08 ± 0.00 | 0.88 ± 0.04 | 0.85 ± 0.12 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| thermal_severe | 0.37 ± 0.14 | 0.45 ± 0.13 | 0.10 ± 0.01 | 0.08 ± 0.00 | 0.85 ± 0.14 | 0.89 ± 0.13 | 0.00 ± 0.00 | 0.00 ± 0.00 |

| scenario | rel_mae fixed | rel_mae hybrid | failure_detection_rate fixed | failure_detection_rate hybrid |
|---|---|---|---|---|
| audio_severe | 0.25 ± 0.03 | 0.18 ± 0.02 | – | – |
| imu_severe | 0.27 ± 0.03 | 0.15 ± 0.01 | – | – |
| lidar_severe | 0.27 ± 0.02 | 0.15 ± 0.01 | – | – |
| nominal | 0.16 ± 0.02 | 0.13 ± 0.01 | – | – |
| odometry_severe | 0.26 ± 0.02 | 0.16 ± 0.01 | – | – |
| rgb_severe | 0.25 ± 0.01 | 0.17 ± 0.02 | – | – |
| thermal_severe | 0.27 ± 0.01 | 0.13 ± 0.01 | – | – |

### exp2_failure

| scenario | f1 fixed | f1 hybrid | ate fixed | ate hybrid | coverage fixed | coverage hybrid | collisions fixed | collisions hybrid |
|---|---|---|---|---|---|---|---|---|
| audio_failed | 0.74 ± 0.11 | 0.75 ± 0.12 | 0.10 ± 0.01 | 0.08 ± 0.00 | 0.87 ± 0.04 | 0.92 ± 0.05 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| imu_failed | 0.04 ± 0.06 | 0.76 ± 0.08 | 17.67 ± 2.68 | 0.09 ± 0.00 | 0.67 ± 0.09 | 0.93 ± 0.04 | 13.33 ± 4.32 | 0.00 ± 0.00 |
| lidar_failed | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| odometry_failed | 0.04 ± 0.05 | 0.83 ± 0.08 | 12.93 ± 3.19 | 0.08 ± 0.01 | 0.46 ± 0.14 | 0.91 ± 0.05 | 25.33 ± 9.71 | 0.00 ± 0.00 |
| rgb_failed | 0.78 ± 0.16 | 0.78 ± 0.12 | 0.12 ± 0.02 | 0.09 ± 0.00 | 0.85 ± 0.12 | 0.94 ± 0.04 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| thermal_failed | 0.50 ± 0.09 | 0.53 ± 0.10 | 0.12 ± 0.02 | 0.08 ± 0.00 | 0.96 ± 0.04 | 0.95 ± 0.04 | 0.00 ± 0.00 | 0.00 ± 0.00 |

| scenario | rel_mae fixed | rel_mae hybrid | failure_detection_rate fixed | failure_detection_rate hybrid |
|---|---|---|---|---|
| audio_failed | 0.28 ± 0.02 | 0.12 ± 0.01 | 0.00 ± 0.00 | 1.00 ± 0.00 |
| imu_failed | 0.31 ± 0.02 | 0.11 ± 0.01 | 0.00 ± 0.00 | 0.99 ± 0.00 |
| lidar_failed | 0.29 ± 0.01 | 0.08 ± 0.01 | 0.00 ± 0.00 | 1.00 ± 0.00 |
| odometry_failed | 0.29 ± 0.02 | 0.12 ± 0.01 | 0.00 ± 0.00 | 0.99 ± 0.00 |
| rgb_failed | 0.28 ± 0.01 | 0.12 ± 0.01 | 0.00 ± 0.00 | 1.00 ± 0.00 |
| thermal_failed | 0.32 ± 0.01 | 0.09 ± 0.01 | 0.00 ± 0.00 | 1.00 ± 0.00 |

### exp3_simultaneous

| scenario | f1 fixed | f1 hybrid | ate fixed | ate hybrid | coverage fixed | coverage hybrid | collisions fixed | collisions hybrid |
|---|---|---|---|---|---|---|---|---|
| earthquake:lidar+odometry | 0.22 ± 0.16 | 0.45 ± 0.37 | 5.46 ± 4.28 | 3.22 ± 2.25 | 0.70 ± 0.19 | 0.73 ± 0.16 | 2.75 ± 2.93 | 1.25 ± 1.86 |
| fire:rgb+thermal | 0.29 ± 0.21 | 0.32 ± 0.22 | 0.08 ± 0.01 | 0.11 ± 0.04 | 0.74 ± 0.35 | 0.77 ± 0.37 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| flood:lidar+odometry+audio | 0.40 ± 0.20 | 0.13 ± 0.15 | 1.84 ± 0.83 | 3.03 ± 1.52 | 0.75 ± 0.11 | 0.84 ± 0.04 | 1.00 ± 1.13 | 2.50 ± 3.05 |
| mixed:rgb+lidar+thermal | 0.18 ± 0.12 | 0.21 ± 0.14 | 1.06 ± 0.45 | 2.03 ± 1.44 | 0.76 ± 0.11 | 0.77 ± 0.10 | 0.00 ± 0.00 | 1.75 ± 1.95 |

| scenario | rel_mae fixed | rel_mae hybrid | failure_detection_rate fixed | failure_detection_rate hybrid |
|---|---|---|---|---|
| earthquake:lidar+odometry | 0.34 ± 0.01 | 0.18 ± 0.00 | – | – |
| fire:rgb+thermal | 0.35 ± 0.00 | 0.14 ± 0.00 | – | – |
| flood:lidar+odometry+audio | 0.41 ± 0.00 | 0.23 ± 0.01 | – | – |
| mixed:rgb+lidar+thermal | 0.45 ± 0.01 | 0.18 ± 0.01 | – | – |

### paired fixed→hybrid

| experiment | A | B | metric | n pairs | mean(B−A) | 95% CI | B higher / tie / lower |
|---|---|---|---|---|---|---|---|
| exp1_individual | fixed | hybrid | f1 | 84 | +0.094 | [+0.022, +0.165] | 48 / 18 / 18 |
| exp1_individual | fixed | hybrid | ate | 84 | -0.889 | [-1.594, -0.288] | 16 / 0 / 68 |
| exp1_individual | fixed | hybrid | coverage | 84 | +0.026 | [-0.028, +0.077] | 51 / 4 / 29 |
| exp1_individual | fixed | hybrid | collisions | 84 | -0.250 | [-0.607, +0.048] | 5 / 71 / 8 |
| exp1_individual | fixed | hybrid | rel_mae | 84 | -0.091 | [-0.105, -0.077] | 6 / 0 / 78 |
| exp2_failure | fixed | hybrid | f1 | 72 | +0.257 | [+0.171, +0.346] | 34 / 29 / 9 |
| exp2_failure | fixed | hybrid | ate | 72 | -5.088 | [-7.043, -3.363] | 6 / 12 / 54 |
| exp2_failure | fixed | hybrid | coverage | 72 | +0.139 | [+0.090, +0.196] | 40 / 15 / 17 |
| exp2_failure | fixed | hybrid | collisions | 72 | -6.444 | [-9.444, -3.805] | 0 / 48 / 24 |
| exp2_failure | fixed | hybrid | rel_mae | 72 | -0.189 | [-0.201, -0.177] | 0 / 0 / 72 |
| exp3_simultaneous | fixed | hybrid | f1 | 24 | +0.013 | [-0.101, +0.137] | 9 / 5 / 10 |
| exp3_simultaneous | fixed | hybrid | ate | 24 | +0.314 | [-0.829, +1.468] | 13 / 0 / 11 |
| exp3_simultaneous | fixed | hybrid | coverage | 24 | +0.027 | [-0.049, +0.100] | 12 / 1 / 11 |
| exp3_simultaneous | fixed | hybrid | collisions | 24 | +0.875 | [-0.083, +2.083] | 9 / 12 / 3 |
| exp3_simultaneous | fixed | hybrid | rel_mae | 24 | -0.228 | [-0.247, -0.209] | 0 / 0 / 24 |

### exp4 overall

| method | f1 | precision | recall | ate | coverage | collisions | rel_mae | rel_ece | failure_detection_rate |
|---|---|---|---|---|---|---|---|---|---|
| audio_only | 0.00 | 0.00 | 0.00 | 0.27 | 0.89 | 0.07 | 0.14 | 0.08 | 1.00 |
| env | 0.40 | 0.49 | 0.37 | 2.33 | 0.81 | 4.11 | 0.19 | 0.18 | 0.00 |
| env_quality | 0.41 | 0.48 | 0.38 | 2.43 | 0.82 | 3.02 | 0.18 | 0.11 | 0.05 |
| env_quality_health | 0.56 | 0.69 | 0.52 | 0.39 | 0.83 | 0.07 | 0.12 | 0.08 | 1.00 |
| fixed | 0.42 | 0.49 | 0.41 | 2.41 | 0.82 | 3.84 | 0.33 | 0.33 | 0.00 |
| hybrid | 0.56 | 0.65 | 0.52 | 0.32 | 0.87 | 0.40 | 0.14 | 0.08 | 1.00 |
| learned | 0.56 | 0.63 | 0.54 | 0.55 | 0.82 | 1.49 | 0.06 | 0.04 | 1.00 |
| quality | 0.45 | 0.55 | 0.42 | 1.99 | 0.81 | 0.07 | 0.17 | 0.12 | 0.97 |
| rgb_only | 0.20 | 0.34 | 0.16 | 0.34 | 0.89 | 0.11 | 0.13 | 0.09 | 1.00 |
| thermal_only | 0.36 | 0.38 | 0.34 | 0.47 | 0.87 | 0.36 | 0.13 | 0.09 | 1.00 |

### exp4 F1 by scenario

| method | mixed_severe | nominal | odometry_failed+lidar_mild | rgb_failed+audio_severe | thermal_failed |
|---|---|---|---|---|---|
| audio_only | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| env | 0.15 | 0.70 | 0.09 | 0.60 | 0.47 |
| env_quality | 0.13 | 0.78 | 0.07 | 0.56 | 0.49 |
| env_quality_health | 0.19 | 0.78 | 0.75 | 0.63 | 0.46 |
| fixed | 0.15 | 0.74 | 0.08 | 0.61 | 0.55 |
| hybrid | 0.12 | 0.79 | 0.72 | 0.60 | 0.56 |
| learned | 0.20 | 0.78 | 0.73 | 0.62 | 0.48 |
| quality | 0.18 | 0.77 | 0.17 | 0.61 | 0.52 |
| rgb_only | 0.00 | 0.37 | 0.30 | 0.00 | 0.31 |
| thermal_only | 0.00 | 0.61 | 0.62 | 0.57 | 0.00 |

### exp4 ATE by scenario

| method | mixed_severe | nominal | odometry_failed+lidar_mild | rgb_failed+audio_severe | thermal_failed |
|---|---|---|---|---|---|
| audio_only | 0.95 | 0.08 | 0.18 | 0.08 | 0.08 |
| env | 1.21 | 0.08 | 10.17 | 0.10 | 0.08 |
| env_quality | 1.03 | 0.08 | 10.86 | 0.08 | 0.08 |
| env_quality_health | 1.54 | 0.08 | 0.19 | 0.09 | 0.07 |
| fixed | 0.96 | 0.10 | 10.74 | 0.13 | 0.11 |
| hybrid | 1.29 | 0.08 | 0.09 | 0.08 | 0.08 |
| learned | 2.41 | 0.08 | 0.11 | 0.08 | 0.08 |
| quality | 0.78 | 0.07 | 8.95 | 0.08 | 0.08 |
| rgb_only | 1.34 | 0.08 | 0.11 | 0.08 | 0.08 |
| thermal_only | 1.98 | 0.08 | 0.14 | 0.08 | 0.08 |

### exp4 paired vs fixed

| experiment | A | B | metric | n pairs | mean(B−A) | 95% CI | B higher / tie / lower |
|---|---|---|---|---|---|---|---|
|  | fixed | rgb_only | f1 | 45 | -0.229 | [-0.343, -0.117] | 7 / 11 / 27 |
|  | fixed | rgb_only | ate | 45 | -2.070 | [-3.485, -0.870] | 7 / 0 / 38 |
|  | fixed | thermal_only | f1 | 45 | -0.066 | [-0.191, +0.070] | 10 / 12 / 23 |
|  | fixed | thermal_only | ate | 45 | -1.937 | [-3.408, -0.623] | 8 / 0 / 37 |
|  | fixed | audio_only | f1 | 45 | -0.425 | [-0.526, -0.323] | 0 / 12 / 33 |
|  | fixed | audio_only | ate | 45 | -2.136 | [-3.523, -0.939] | 5 / 0 / 40 |
|  | fixed | env | f1 | 45 | -0.022 | [-0.069, +0.023] | 11 / 19 / 15 |
|  | fixed | env | ate | 45 | -0.082 | [-0.934, +0.801] | 14 / 0 / 31 |
|  | fixed | quality | f1 | 45 | +0.026 | [-0.036, +0.095] | 13 / 25 / 7 |
|  | fixed | quality | ate | 45 | -0.416 | [-1.501, +0.500] | 12 / 0 / 33 |
|  | fixed | env_quality | f1 | 45 | -0.017 | [-0.064, +0.029] | 12 / 19 / 14 |
|  | fixed | env_quality | ate | 45 | +0.018 | [-0.658, +0.782] | 14 / 0 / 31 |
|  | fixed | env_quality_health | f1 | 45 | +0.137 | [+0.046, +0.236] | 23 / 13 / 9 |
|  | fixed | env_quality_health | ate | 45 | -2.014 | [-3.440, -0.789] | 10 / 0 / 35 |
|  | fixed | hybrid | f1 | 45 | +0.130 | [+0.047, +0.223] | 19 / 17 / 9 |
|  | fixed | hybrid | ate | 45 | -2.084 | [-3.514, -0.854] | 7 / 0 / 38 |
|  | fixed | learned | f1 | 45 | +0.136 | [+0.056, +0.221] | 18 / 21 / 6 |
|  | fixed | learned | ate | 45 | -1.856 | [-3.320, -0.573] | 11 / 0 / 34 |

### exp5 overall

| method | coverage | victims_found | f1 | path_length | steps | collisions | hazard_exposure | efficiency | success | ate | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 random | 0.78 | 2.43 | 0.54 | 259.90 | 339.83 | 0.06 | 35.24 | 4.33 | 0.15 | 0.17 | 54 |
| E1 nearest | 0.83 | 2.81 | 0.61 | 209.43 | 283.19 | 0.02 | 29.68 | 4.44 | 0.24 | 0.23 | 54 |
| E2 info-gain | 0.83 | 2.81 | 0.60 | 198.20 | 263.93 | 0.31 | 30.15 | 4.48 | 0.20 | 0.31 | 54 |
| E2f info-gain/fixed | 0.74 | 1.81 | 0.37 | 200.08 | 308.33 | 8.07 | 33.47 | 4.20 | 0.11 | 4.73 | 54 |
| E3 PPO/fixed | 0.68 | 1.85 | 0.38 | 192.94 | 291.78 | 7.06 | 32.08 | 4.20 | 0.09 | 3.65 | 54 |
| E4 PPO/hybrid | 0.80 | 2.69 | 0.58 | 232.48 | 302.61 | 0.41 | 29.25 | 4.40 | 0.15 | 0.31 | 54 |

### exp5 coverage by condition

| method | degraded | lidar+thermal | nominal |
|---|---|---|---|
| E0 random | 0.85 | 0.67 | 0.82 |
| E1 nearest | 0.91 | 0.67 | 0.91 |
| E2 info-gain | 0.87 | 0.71 | 0.90 |
| E2f info-gain/fixed | 0.52 | 0.81 | 0.89 |
| E3 PPO/fixed | 0.38 | 0.77 | 0.90 |
| E4 PPO/hybrid | 0.87 | 0.63 | 0.91 |

### exp5 victims_found by condition

| method | degraded | lidar+thermal | nominal |
|---|---|---|---|
| E0 random | 2.56 | 1.56 | 3.17 |
| E1 nearest | 3.33 | 1.50 | 3.61 |
| E2 info-gain | 3.28 | 1.61 | 3.56 |
| E2f info-gain/fixed | 0.44 | 1.67 | 3.33 |
| E3 PPO/fixed | 0.39 | 1.67 | 3.50 |
| E4 PPO/hybrid | 3.22 | 1.33 | 3.50 |

### exp5 path_length by condition

| method | degraded | lidar+thermal | nominal |
|---|---|---|---|
| E0 random | 261.71 | 288.89 | 229.11 |
| E1 nearest | 186.10 | 245.41 | 196.79 |
| E2 info-gain | 156.96 | 262.29 | 175.34 |
| E2f info-gain/fixed | 179.15 | 235.97 | 185.10 |
| E3 PPO/fixed | 126.15 | 234.64 | 218.02 |
| E4 PPO/hybrid | 192.07 | 256.35 | 249.01 |

### exp5 paired

| experiment | A | B | metric | n pairs | mean(B−A) | 95% CI | B higher / tie / lower |
|---|---|---|---|---|---|---|---|
|  | E2 info-gain | E0 random | coverage | 54 | -0.048 | [-0.100, +0.003] | 17 / 5 / 32 |
|  | E2 info-gain | E0 random | victims_found | 54 | -0.389 | [-0.741, -0.037] | 14 / 16 / 24 |
|  | E2 info-gain | E0 random | path_length | 54 | +61.703 | [+40.429, +82.414] | 44 / 3 / 7 |
|  | E2 info-gain | E1 nearest | coverage | 54 | +0.002 | [-0.038, +0.046] | 27 / 5 / 22 |
|  | E2 info-gain | E1 nearest | victims_found | 54 | +0.000 | [-0.259, +0.278] | 15 / 24 / 15 |
|  | E2 info-gain | E1 nearest | path_length | 54 | +11.235 | [-6.624, +27.369] | 31 / 3 / 20 |
|  | E2 info-gain | E2f info-gain/fixed | coverage | 54 | -0.088 | [-0.166, -0.013] | 21 / 5 / 28 |
|  | E2 info-gain | E2f info-gain/fixed | victims_found | 54 | -1.000 | [-1.481, -0.537] | 8 / 18 / 28 |
|  | E2 info-gain | E2f info-gain/fixed | path_length | 54 | +1.878 | [-20.617, +26.090] | 26 / 2 / 26 |
|  | E2 info-gain | E3 PPO/fixed | coverage | 54 | -0.146 | [-0.231, -0.062] | 22 / 4 / 28 |
|  | E2 info-gain | E3 PPO/fixed | victims_found | 54 | -0.963 | [-1.481, -0.481] | 9 / 20 / 25 |
|  | E2 info-gain | E3 PPO/fixed | path_length | 54 | -5.261 | [-29.981, +19.588] | 26 / 2 / 26 |
|  | E2 info-gain | E4 PPO/hybrid | coverage | 54 | -0.027 | [-0.061, +0.007] | 17 / 6 / 31 |
|  | E2 info-gain | E4 PPO/hybrid | victims_found | 54 | -0.130 | [-0.371, +0.112] | 10 / 28 / 16 |
|  | E2 info-gain | E4 PPO/hybrid | path_length | 54 | +34.282 | [+10.101, +57.887] | 38 / 4 / 12 |

### exp6 coverage by severity

| method | 0.2 | 0.5 | 0.8 |
|---|---|---|---|
| E1 nearest | 0.76 | 0.83 | 0.85 |
| E2 info-gain | 0.76 | 0.86 | 0.85 |
| E2f info-gain/fixed | 0.78 | 0.82 | 0.82 |
| E3 PPO/fixed | 0.76 | 0.80 | 0.79 |
| E4 PPO/hybrid | 0.78 | 0.79 | 0.80 |

### exp6 victims_found by severity

| method | 0.2 | 0.5 | 0.8 |
|---|---|---|---|
| E1 nearest | 3.62 | 3.50 | 3.25 |
| E2 info-gain | 3.12 | 3.54 | 3.33 |
| E2f info-gain/fixed | 3.46 | 3.54 | 3.50 |
| E3 PPO/fixed | 3.33 | 3.17 | 3.12 |
| E4 PPO/hybrid | 3.17 | 3.12 | 2.83 |

### exp6 f1 by severity

| method | 0.2 | 0.5 | 0.8 |
|---|---|---|---|
| E1 nearest | 0.40 | 0.37 | 0.48 |
| E2 info-gain | 0.41 | 0.46 | 0.50 |
| E2f info-gain/fixed | 0.32 | 0.30 | 0.33 |
| E3 PPO/fixed | 0.27 | 0.23 | 0.27 |
| E4 PPO/hybrid | 0.38 | 0.41 | 0.40 |

### exp6 coverage / victims by difficulty

| method | easy | hard |
|---|---|---|
| E1 nearest | 0.84 | 0.78 |
| E2 info-gain | 0.85 | 0.79 |
| E2f info-gain/fixed | 0.83 | 0.78 |
| E3 PPO/fixed | 0.81 | 0.76 |
| E4 PPO/hybrid | 0.81 | 0.77 |
| method | easy | hard |
|---|---|---|
| E1 nearest | 3.42 | 3.50 |
| E2 info-gain | 3.22 | 3.44 |
| E2f info-gain/fixed | 3.58 | 3.42 |
| E3 PPO/fixed | 3.17 | 3.25 |
| E4 PPO/hybrid | 2.97 | 3.11 |

### rl.evaluate

| policy | episode_reward | coverage | victims_found | f1 | path_length | steps |
|---|---|---|---|---|---|---|
| PPO/fixed | 29.36 | 0.72 | 3.17 | 0.63 | 185.73 | 237.08 |
| PPO/hybrid | 28.12 | 0.78 | 3.00 | 0.66 | 186.77 | 232.50 |
| infogain/hybrid | 31.55 | 0.79 | 3.42 | 0.72 | 158.04 | 197.42 |
| nearest/hybrid | 32.24 | 0.82 | 3.25 | 0.67 | 179.65 | 228.08 |
| random/hybrid | 22.15 | 0.75 | 2.83 | 0.63 | 261.92 | 333.08 |

### computational.json
```
{
 "peak_python_memory_MB_one_mission": 5.7,
 "mission_steps_per_s": 83.9,
 "perception_action_step_ms": 11.91,
 "training_ppo_fixed": {
  "timesteps": 25000,
  "device": "cpu",
  "training_seconds": 5627.2,
  "episodes": 1064
 },
 "training_ppo_hybrid": {
  "timesteps": 25000,
  "device": "cpu",
  "training_seconds": 5596.9,
  "episodes": 1062
 },
 "ppo_inference_ms_mean_under_load": {
  "E3 PPO/fixed": 8.636,
  "E4 PPO/hybrid": 9.693
 },
 "ppo_inference_ms_idle": 1.14,
 "ppo_mission_step_ms_idle": 11.64,
 "note": "idle single-process measurement on a 2-core cloud CPU (no GPU); training_seconds were measured while 4 jobs shared 2 cores"
}
```

### learned_reliability_stats.json
```
{
 "device": "cpu",
 "samples": 56418,
 "missions": 45,
 "epochs": 40,
 "collect_seconds": 105.5,
 "train_seconds": 187.5,
 "val_mae_learned": 0.0422,
 "val_mae_hybrid_formula_unsmoothed": 0.1354,
 "note": "validation rows are per-step samples from held-out shuffle of the same missions"
}
```
