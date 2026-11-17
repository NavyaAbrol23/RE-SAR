"""Headless GUI checks (SDL dummy driver): every button and key performs a real action."""
import os

import pygame
import pytest

from sensors.base_sensor import SENSOR_NAMES


@pytest.fixture(scope="module")
def app():
    from visualization.pygame_view import App
    a = App(headless=True)
    yield a
    pygame.quit()


def test_menu_buttons(app):
    s = app.settings
    app.frame()
    names = [b.label() if callable(b.label) else b.label for b in app.menu_buttons]
    assert "START SIMULATION" in names
    sev, vic, seed, fus, nav = s.severity, s.victims, s.seed, s.fusion, s.nav
    for b in app.menu_buttons:
        lab = b.label() if callable(b.label) else b.label
        if lab in ("START SIMULATION", "Quit"):
            continue
        b.callback()
    assert s.disaster == "flood" and s.difficulty == "hard"  # last button of each group wins
    assert (s.severity, s.victims, s.fusion, s.nav) != (sev, vic, fus, nav)


def test_simulation_buttons_and_keys(app):
    s = app.settings
    s.disaster, s.fusion, s.nav, s.victims, s.severity = "fire", "hybrid", "infogain", 4, 0.5
    app.start()
    sim = app.sim
    assert app.state == "sim"
    app.frame()
    t0 = sim.mission.t
    for _ in range(3):
        app.frame()
    assert sim.mission.t > t0
    lab = {(b.label() if callable(b.label) else b.label): b for b in app.sim_buttons}
    lab["Pause"].callback()
    assert sim.paused
    t = sim.mission.t
    lab["Step"].callback()
    assert sim.mission.t == t + 1
    sp = sim.speed
    lab["Faster"].callback()
    assert sim.speed == sp + 1
    lab["Slower"].callback()
    v = app.show_truth
    lab["View"].callback()
    assert app.show_truth != v
    lab["LiDAR"].callback()
    seed = s.seed
    lab["New map"].callback()
    assert s.seed == seed + 1
    lab["Reset"].callback()
    assert sim.mission.t == 0
    for b in app.sensor_buttons:
        b.callback()
    assert all(sim.mission.suite.degradation.levels[x] == "mild" for x in SENSOR_NAMES)
    nav_btn = [b for b in app.sim_buttons if callable(b.label) and b.label().startswith("Nav")][0]
    fus_btn = [b for b in app.sim_buttons if callable(b.label) and b.label().startswith("Fusion")][0]
    nav_btn.callback()
    fus_btn.callback()
    assert s.fusion == "fixed"
    for k in (pygame.K_SPACE, pygame.K_n, pygame.K_1, pygame.K_v, pygame.K_l, pygame.K_EQUALS, pygame.K_MINUS):
        app.handle_key(k)
    app.frame()
    lab["Menu"].callback()
    assert app.state == "menu"
