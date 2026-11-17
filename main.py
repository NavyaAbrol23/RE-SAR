"""RE-SAR entry point.

  python main.py                                  # interactive Pygame application
  python main.py --disaster fire --seed 7 --start # skip the menu
  python main.py --headless-demo 300              # no window: run a mission, save screenshots + metrics
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main(argv=None):
    ap = argparse.ArgumentParser(description="RE-SAR reliability-aware SAR simulator")
    ap.add_argument("--disaster", default="earthquake", choices=["earthquake", "fire", "flood"])
    ap.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--severity", type=float, default=0.5)
    ap.add_argument("--victims", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fusion", default="hybrid",
                    choices=["fixed", "env", "quality", "env_quality", "env_quality_health", "hybrid"])
    ap.add_argument("--nav", default="infogain", choices=["random", "nearest", "infogain", "ppo"])
    ap.add_argument("--start", action="store_true", help="skip the menu")
    ap.add_argument("--headless-demo", type=int, default=0, metavar="STEPS",
                    help="run without a display for STEPS steps and save screenshots")
    ap.add_argument("--degrade", default="", help="e.g. rgb=failed,lidar=severe")
    a = ap.parse_args(argv)

    from visualization.pygame_view import App
    from visualization.simulator import Settings
    s = Settings()
    s.disaster, s.difficulty, s.severity, s.victims = a.disaster, a.difficulty, a.severity, a.victims
    s.seed, s.fusion, s.nav = a.seed, a.fusion, a.nav
    app = App(s, headless=a.headless_demo > 0)
    out = os.path.join(ROOT, "results", "plots")
    os.makedirs(out, exist_ok=True)
    if a.headless_demo:
        app.frame()
        app.screenshot(os.path.join(out, "gui_menu.png"))
        app.start()
        for kv in filter(None, a.degrade.split(",")):
            k, v = kv.split("=")
            app.sim.mission.set_degradation(k, v)
        app.sim.speed = 5
        frames = 0
        while not app.sim.mission.done and app.sim.mission.t < a.headless_demo:
            app.frame()
            frames += 1
        app.screenshot(os.path.join(out, f"gui_simulation_{a.disaster}.png"))
        ev = app.sim.mission.evaluate()
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in ev.items()}, indent=1))
        print(f"[headless-demo] {frames} frames rendered; screenshots in results/plots/")
        return ev
    if a.start:
        app.start()
    app.run()


if __name__ == "__main__":
    main()
