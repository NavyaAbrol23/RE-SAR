import numpy as np
import pytest

from config.loader import DISASTERS, load_config
from environment.map_generator import reachable_mask
from environment.world import World


@pytest.mark.parametrize("disaster", DISASTERS)
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_world_generation(disaster, difficulty):
    w = World(load_config(disaster, {"world": {"difficulty": difficulty}}), seed=3)
    assert w.grid.shape == (w.H, w.W)
    assert w.grid[0].all() and w.grid[-1].all() and w.grid[:, 0].all() and w.grid[:, -1].all()
    sx, sy = int(w.start[0]), int(w.start[1])
    assert not w.grid[sy, sx] and w.hazard[sy, sx] == 0
    assert len(w.victims) == 5
    for v in w.victims:  # every victim is reachable and not inside an obstacle
        assert w.reachable[int(v.y), int(v.x)]
    assert np.all((w.hazard >= 0) & (w.hazard <= 1))
    for f in (w.smoke, w.dust, w.heat, w.water):
        assert np.isfinite(f).all() and f.min() >= 0 and f.max() <= 1


def test_seed_reproducibility():
    a = World(load_config("fire"), seed=11)
    b = World(load_config("fire"), seed=11)
    c = World(load_config("fire"), seed=12)
    assert np.array_equal(a.grid, b.grid) and np.array_equal(a.hazard, b.hazard)
    assert [(v.x, v.y) for v in a.victims] == [(v.x, v.y) for v in b.victims]
    assert not np.array_equal(a.grid, c.grid)


def test_victim_count_configurable():
    w = World(load_config("flood", {"world": {"num_victims": 8}}), seed=2)
    assert len(w.victims) == 8


def test_disaster_specific_hazards():
    fire = World(load_config("fire", {"world": {"severity": 0.8}}), seed=1)
    flood = World(load_config("flood", {"world": {"severity": 0.8}}), seed=1)
    assert fire.heat.max() > 0.5 and fire.smoke.max() > 0.2
    assert flood.water.max() > 0.3
    assert fire.water.max() == 0 and flood.heat.max() == 0


def test_collision_blocks_motion():
    w = World(load_config("earthquake"), seed=0)
    rng = np.random.default_rng(0)
    w.x, w.y, w.theta = w.start[0], w.start[1], 0.0
    for _ in range(60):  # drive straight into the boundary wall eventually
        w.apply_motion(0.0, 1.0, rng)
    assert not w.blocked(w.x, w.y)
    assert w.collisions >= 1


def test_ray_casting_hits_wall():
    w = World(load_config("earthquake"), seed=0)
    r = w.cast_rays(w.start[0], w.start[1], np.linspace(-np.pi, np.pi, 36), 60.0)
    assert np.isfinite(r).all()  # building is enclosed
    assert (r > 0).all()


def test_reachable_mask():
    g = np.zeros((5, 5), bool)
    g[:, 2] = True
    m = reachable_mask(g, (0, 0))
    assert m[:, :2].all() and not m[:, 3:].any()
