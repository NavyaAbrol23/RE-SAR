import os
import tempfile

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl.observation import K_FRONTIERS, OBS_DIM
from rl.sar_env import SAREnvironment


def test_gymnasium_check_env():
    env = SAREnvironment("hybrid", seed=0)
    check_env(env, skip_render_check=True)


def test_reset_step_api_and_bounds():
    env = SAREnvironment("fixed", seed=1)
    obs, info = env.reset(seed=5)
    assert obs.shape == (OBS_DIM,) and env.action_space.n == K_FRONTIERS + 1
    for _ in range(5):
        obs, r, term, trunc, info = env.step(0)
        assert np.isfinite(obs).all() and obs.min() >= -1 and obs.max() <= 1
        assert np.isfinite(r)
        if term or trunc:
            break


def test_invalid_action_is_penalised_not_crashing():
    env = SAREnvironment("hybrid", seed=2)
    env.reset(seed=2)
    while len(env.mission.frontiers) >= K_FRONTIERS:
        env.step(0)
    obs, r, term, trunc, info = env.step(K_FRONTIERS - 1)
    assert info["reward_terms"]["invalid"] == 1


def test_stop_action_terminates_with_penalty():
    env = SAREnvironment("hybrid", seed=3)
    env.reset(seed=3)
    _, r, term, trunc, info = env.step(K_FRONTIERS)
    assert term and r < 0


def test_observation_has_no_ground_truth_leak():
    """Moving GT victims must not change the observation (agent never sees GT)."""
    env = SAREnvironment("hybrid", seed=4)
    o1, _ = env.reset(seed=4)
    for v in env.mission.world.victims:
        v.x, v.y = 1.5, 1.5
    from rl.observation import build_observation
    o2 = build_observation(env.mission)
    assert np.array_equal(o1, o2)


def test_ppo_smoke_train_save_load():
    from stable_baselines3 import PPO
    env = SAREnvironment("hybrid", seed=6)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, verbose=0, device="cpu")
    model.learn(total_timesteps=64)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.zip")
        model.save(p)
        m2 = PPO.load(p, device="cpu")
        obs, _ = env.reset(seed=7)
        a1, _ = model.predict(obs, deterministic=True)
        a2, _ = m2.predict(obs, deterministic=True)
        assert int(a1) == int(a2)


def test_ppo_chooser_drives_mission():
    from stable_baselines3 import PPO
    from config.loader import load_config
    from core.mission import Mission
    from rl.policy import PPOChooser
    env = SAREnvironment("hybrid", seed=8)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, verbose=0, device="cpu")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.zip")
        model.save(p)
        ch = PPOChooser(p)
        m = Mission(load_config("fire"), seed=9, nav="ppo", chooser=ch)
        for _ in range(30):
            m.step()
        assert m.last_action_source in ("ppo", "-") and m.t == 30
