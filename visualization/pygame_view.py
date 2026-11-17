"""RE-SAR interactive Pygame application: main menu + simulation dashboard.

Keyboard: SPACE pause/resume, N single step, R reset, M new map, V ground-truth/estimate view,
L LiDAR overlay, P cycle navigation mode, F cycle fusion strategy, +/- speed,
1-6 cycle degradation of rgb/thermal/lidar/imu/odometry/audio, ESC back to menu / quit.
"""
from __future__ import annotations

import os

import numpy as np
import pygame

from reliability.sensor_health import label as health_label
from sensors.base_sensor import SENSOR_NAMES
from visualization import panels as P
from visualization.simulator import DIFFICULTIES, Settings, SimulationController

W, H = 1440, 900
CS = 16                     # world cell size (px)
WORLD_ORIGIN = (12, 52)
MAP_CS = 4


class App:
    def __init__(self, settings: Settings | None = None, headless: bool = False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.display.set_caption("RE-SAR - Reliability-Aware Search and Rescue")
        self.screen = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()
        self.settings = settings or Settings()
        self.state = "menu"
        self.sim = None
        self.show_truth = True
        self.show_lidar = True
        self.running = True
        self._gt_cache = None
        self._build_menu()

    # ================================================================ menu
    def _build_menu(self):
        s = self.settings
        B = P.Button
        cx = W // 2
        y0 = 250
        self.menu_buttons = []
        for i, d in enumerate(("earthquake", "fire", "flood")):
            self.menu_buttons.append(B((cx - 240 + i * 165, y0, 150, 40), d.capitalize(),
                                      lambda d=d: setattr(s, "disaster", d),
                                      color=lambda d=d: (60, 110, 170) if s.disaster == d else (52, 57, 68)))
        for i, d in enumerate(DIFFICULTIES):
            self.menu_buttons.append(B((cx - 240 + i * 165, y0 + 80, 150, 40), d.capitalize(),
                                      lambda d=d: setattr(s, "difficulty", d),
                                      color=lambda d=d: (60, 110, 170) if s.difficulty == d else (52, 57, 68)))

        def stepper(y, get, dec, inc):
            self.menu_buttons.append(B((cx + 20, y, 44, 38), "-", dec))
            self.menu_buttons.append(B((cx + 196, y, 44, 38), "+", inc))
        stepper(y0 + 160, None, lambda: setattr(s, "severity", round(max(0.0, s.severity - 0.1), 1)),
                lambda: setattr(s, "severity", round(min(1.0, s.severity + 0.1), 1)))
        stepper(y0 + 210, None, lambda: setattr(s, "victims", max(1, s.victims - 1)),
                lambda: setattr(s, "victims", min(12, s.victims + 1)))
        stepper(y0 + 260, None, lambda: setattr(s, "seed", max(0, s.seed - 1)), lambda: setattr(s, "seed", s.seed + 1))
        self.menu_buttons.append(B((cx + 250, y0 + 260, 90, 38), "Random", lambda: setattr(s, "seed", int(np.random.randint(0, 99999)))))
        from reliability.hybrid_fusion import STRATEGIES
        from visualization.simulator import NAV_MODES
        self.menu_buttons.append(B((cx + 20, y0 + 310, 220, 38), lambda: f"Fusion: {s.fusion}",
                                   lambda: setattr(s, "fusion", STRATEGIES[(STRATEGIES.index(s.fusion) + 1) % len(STRATEGIES)])))
        self.menu_buttons.append(B((cx + 20, y0 + 360, 220, 38), lambda: f"Navigation: {s.nav}",
                                   lambda: setattr(s, "nav", NAV_MODES[(NAV_MODES.index(s.nav) + 1) % len(NAV_MODES)])))
        self.menu_buttons.append(B((cx - 120, y0 + 440, 240, 56), "START SIMULATION", self.start, color=(40, 140, 90)))
        self.menu_buttons.append(B((cx - 60, y0 + 510, 120, 36), "Quit", self.quit, color=(110, 50, 50)))

    def draw_menu(self, mouse):
        sc, s = self.screen, self.settings
        sc.fill(P.BG)
        P.text(sc, "RE-SAR", (W // 2, 110), 84, P.ACCENT, bold=True, center=True)
        P.text(sc, "Reliability-Aware Autonomous Exploration for Search and Rescue in Sensor-Degraded Environments",
               (W // 2, 165), 24, P.TEXT, center=True)
        P.text(sc, "UCS760 - controlled 2D simulation (not a physically validated SAR robot)", (W // 2, 195), 20,
               P.MUTED, center=True)
        cx, y0 = W // 2, 250
        for lab, y in (("Disaster", y0 + 10), ("Difficulty", y0 + 90)):
            P.text(sc, lab, (cx - 400, y), 24, P.MUTED)
        for lab, val, y in (("Severity", f"{s.severity:.1f}", y0 + 170), ("Victims", str(s.victims), y0 + 220),
                            ("Random seed", str(s.seed), y0 + 270)):
            P.text(sc, lab, (cx - 400, y), 24, P.MUTED)
            P.text(sc, val, (cx + 130, y + 10), 28, P.TEXT, center=True)
        P.text(sc, "Reliability model", (cx - 400, y0 + 320), 24, P.MUTED)
        P.text(sc, "Exploration policy", (cx - 400, y0 + 370), 24, P.MUTED)
        for b in self.menu_buttons:
            b.draw(sc, mouse)

    def start(self):
        self.sim = SimulationController(self.settings)
        self._gt_cache = P.ground_truth_image(self.sim.mission.world)
        self._build_sim_buttons()
        self.state = "sim"

    def quit(self):
        self.running = False

    # ========================================================== simulation
    def _build_sim_buttons(self):
        B, sim = P.Button, self.sim
        x0, y0 = 672, 838
        specs = [("Pause" if True else "", sim.toggle_pause), ("Step", sim.step_once), ("Reset", self._reset),
                 ("New map", self._new_map), ("View", self.toggle_view), ("LiDAR", self.toggle_lidar),
                 ("Slower", sim.slower), ("Faster", sim.faster), ("Menu", self.to_menu)]
        self.sim_buttons = []
        for i, (lab, cb) in enumerate(specs):
            label = (lambda: "Resume" if sim.paused else "Pause") if i == 0 else lab
            self.sim_buttons.append(B((x0 + i * 84, y0, 78, 34), label, cb))
        self.sim_buttons.append(B((1100, 590, 160, 30), lambda: f"Nav: {sim.s.nav}", self._cycle_nav))
        self.sim_buttons.append(B((1268, 590, 160, 30), lambda: f"Fusion: {sim.s.fusion}", sim.cycle_fusion))
        self.sensor_buttons = []
        for i, s in enumerate(SENSOR_NAMES):
            self.sensor_buttons.append(B((1336, 230 + i * 50, 92, 26),
                                         lambda s=s: sim.mission.suite.degradation.levels[s].upper(),
                                         lambda s=s: sim.cycle_sensor(s),
                                         color=lambda s=s: tuple(int(c * 0.55) for c in P.LEVEL_COLORS[sim.mission.suite.degradation.levels[s]])))

    def _reset(self):
        self.sim.reset()
        self._gt_cache = P.ground_truth_image(self.sim.mission.world)

    def _new_map(self):
        self.sim.new_map()
        self._gt_cache = P.ground_truth_image(self.sim.mission.world)

    def _cycle_nav(self):
        self.sim.cycle_nav()

    def toggle_view(self):
        self.show_truth = not self.show_truth

    def toggle_lidar(self):
        self.show_lidar = not self.show_lidar

    def to_menu(self):
        self.state = "menu"

    def draw_sim(self, mouse):
        sc, sim, m = self.screen, self.sim, self.sim.mission
        sc.fill(P.BG)
        w = m.world
        # ---- header
        P.text(sc, "RE-SAR", (12, 12), 34, P.ACCENT, bold=True)
        status = "DONE: " + m.done_reason if m.done else ("PAUSED" if sim.paused else "RUNNING")
        P.text(sc, f"{w.disaster.upper()}  |  difficulty {w.difficulty}  |  severity {w.severity:.1f}  |  seed {sim.s.seed}"
               f"  |  {status}  |  {sim.steps_per_second:.0f} steps/s  |  {self.clock.get_fps():.0f} FPS",
               (130, 18), 22, P.TEXT)
        # ---- world view
        P.panel(sc, (WORLD_ORIGIN[0] - 4, WORLD_ORIGIN[1] - 4, w.W * CS + 8, w.H * CS + 8))
        P.draw_world(sc, m, WORLD_ORIGIN, CS, self.show_truth, self.show_lidar, self._gt_cache)
        P.text(sc, "World view: " + ("GROUND TRUTH (evaluation only)" if self.show_truth else "ROBOT ESTIMATE (occupancy + sensed hazard)"),
               (WORLD_ORIGIN[0], WORLD_ORIGIN[1] + w.H * CS + 6), 19, P.MUTED)
        # ---- map panel
        self.draw_maps(12, 560)
        # ---- sensor panel
        self.draw_sensors(664, 52, mouse)
        # ---- exploration / metrics
        self.draw_exploration(664, 560)
        self.draw_metrics(1094, 628)
        self.draw_legend(664, 708)
        for b in self.sim_buttons + self.sensor_buttons:
            b.draw(sc, mouse)
        if sim.message:
            P.text(sc, sim.message, (664, 880), 17, P.WARN)

    def draw_maps(self, x0, y0):
        sc, m = self.screen, self.sim.mission
        L = m.shared.layers()
        P.panel(sc, (x0 - 4, y0 - 4, 648, 336), "Shared uncertainty-aware map (robot estimate)")
        known = m.shared.occ.known
        cv = P.cv2
        cov_img = np.where(known[..., None], np.array([40, 170, 150], np.uint8), np.array([45, 45, 52], np.uint8))
        items = [("Occupancy P(occ)", P.colormap(1 - L["occupancy"], 0, 1, cv.COLORMAP_BONE)),
                 ("Uncertainty", P.colormap(L["uncertainty"], 0, 1, cv.COLORMAP_VIRIDIS)),
                 ("Hazard (sensed)", P.colormap(L["hazard"], 0, 1, cv.COLORMAP_HOT)),
                 ("Victim confidence", P.colormap(L["victim"], 0, 1, cv.COLORMAP_INFERNO)),
                 ("Exploration coverage", cov_img.astype(np.uint8)),
                 ("Traversability cost", P.colormap(L["traversability"], 1, 4, cv.COLORMAP_CIVIDIS))]
        mw, mh = m.world.W * MAP_CS, m.world.H * MAP_CS
        for i, (name, img) in enumerate(items):
            cx, cy = x0 + 6 + (i % 3) * (mw + 52), y0 + 26 + (i // 3) * (mh + 30)
            sc.blit(P.array_surface(img, (mw, mh)), (cx, cy))
            P.text(sc, name, (cx, cy + mh + 3), 17, P.MUTED)

    def draw_sensors(self, x0, y0, mouse):
        sc, sim, m = self.screen, self.sim, self.sim.mission
        P.panel(sc, (x0, y0, 768, 500), "Sensors - reliability, health, degradation")
        obs = m.last["obs"]
        if obs:
            sz = (104, 104)
            sc.blit(P.rgb_preview(obs["rgb"].data["image"], sz), (x0 + 12, y0 + 32))
            sc.blit(P.thermal_preview(obs["thermal"].data["temp"], sz), (x0 + 128, y0 + 32))
            P.lidar_preview(sc, pygame.Rect(x0 + 244, y0 + 32, 104, 104), obs["lidar"].data)
            for i, n in enumerate(("RGB camera", "Thermal (10-80 C)", "LiDAR (fwd = up)")):
                P.text(sc, n, (x0 + 12 + i * 116, y0 + 140), 16, P.MUTED)
            rep = m.last["rep"]
            P.text(sc, "Fusion weights (normalised per group)", (x0 + 372, y0 + 34), 18, P.MUTED)
            P.text(sc, "Localisation: " + "  ".join(f"{k[:4]} {v:.2f}" for k, v in rep.weights_loc.items())
                   + ("  [ALL FAILED -> command fallback]" if rep.loc_all_failed else ""), (x0 + 372, y0 + 56), 18)
            P.text(sc, "Perception:   " + "  ".join(f"{k[:4]} {v:.2f}" for k, v in rep.weights_perc.items())
                   + ("  [ALL FAILED]" if rep.perc_all_failed else ""), (x0 + 372, y0 + 78), 18)
            P.text(sc, f"Strategy: {rep.strategy}", (x0 + 372, y0 + 100), 18, P.ACCENT)
            P.text(sc, "r = h * (wE*E + wQ*Q + wA*A)/(wE+wQ+wA)", (x0 + 372, y0 + 120), 17, P.MUTED)
        P.text(sc, "sensor        reliability (white tick = simulator ground truth)      E     Q     H     A   health   status / level",
               (x0 + 12, y0 + 160), 16, P.MUTED)
        for i, row in enumerate(sim.sensor_rows()):
            y = y0 + 180 + i * 50
            P.text(sc, row["name"].upper(), (x0 + 12, y + 4), 21, P.TEXT, bold=True)
            P.bar(sc, (x0 + 100, y + 6, 190, 14), row["r"], P.ACCENT, marker=row["true"])
            P.text(sc, f"{row['r']:.2f}", (x0 + 296, y + 4), 19)
            for j, k in enumerate(("env", "q", "h", "a")):
                P.text(sc, f"{row[k]:.2f}", (x0 + 340 + j * 42, y + 4), 18, P.MUTED)
            hl = health_label(row["h"])
            P.text(sc, hl, (x0 + 512, y + 4), 18, P.HEALTH_COLORS[hl], bold=True)
            P.text(sc, row["status"], (x0 + 100, y + 26), 16, P.MUTED)
        P.text(sc, "click a level (or keys 1-6) to inject: NORMAL > MILD > SEVERE > FAILED", (x0 + 380, y0 + 482), 15, P.MUTED)

    def draw_exploration(self, x0, y0):
        sc, sim, m = self.screen, self.sim, self.sim.mission
        P.panel(sc, (x0, y0, 422, 140), "Exploration")
        f = m.target
        lines = [f"Navigation mode: {m.nav}   (decision by {m.last_action_source})",
                 f"Selected frontier: {m.target_index if f else '-'}  of {len(m.frontiers)}"
                 + (f"  @({f.x},{f.y})  IG {f.info_gain:.0f}  cost {f.dist:.1f}" if f else ""),
                 f"Step reward {sim.last_reward:+.2f}   cumulative {sim.cum_reward:+.1f}",
                 f"Known cells {int(m.shared.occ.known.sum())}   confirmed victims {len(m.tracker.confirmed)}"
                 f"   tracks {len(m.tracker.tracks)}",
                 f"Safety: {m.last.get('safety', '-')}  (vetoes {m.safety.vetoes})"]
        if getattr(m.chooser, "last_action", None) is not None and m.nav == "ppo":
            lines[0] += f"   PPO action {m.chooser.last_action}"
        for i, l in enumerate(lines):
            P.text(sc, l, (x0 + 10, y0 + 30 + i * 21), 18)

    def draw_metrics(self, x0, y0):
        sc, sim, m = self.screen, self.sim, self.sim.mission
        e = sim.metrics
        P.panel(sc, (x0, y0, 338, 200), "Metrics (simulation ground truth)")
        x, y, th = m.ekf.pose
        cur = float(np.hypot(x - m.world.x, y - m.world.y))
        lines = [f"Coverage {100 * e['coverage']:.1f} %     steps {m.t}/{m.cfg['robot']['max_steps']}",
                 f"Victims found {e['victims_found']}/{e['victims_total']}   false pos. {e['false_positives']}",
                 f"Precision {e['precision']:.2f}  Recall {e['recall']:.2f}  F1 {e['f1']:.2f}",
                 f"Loc. error now {cur:.2f}   ATE {e['ate']:.2f} cells",
                 f"Collisions {e['collisions']}   hazard exp. {e['hazard_exposure']:.1f}",
                 f"Rel. MAE {e['rel_mae']:.2f}   step {e['step_ms']:.1f} ms",
                 f"Path {e['path_length']:.0f} cells   occ. acc {e['occupancy_accuracy']:.2f}"]
        for i, l in enumerate(lines):
            P.text(sc, l, (x0 + 10, y0 + 30 + i * 22), 18)

    def draw_legend(self, x0, y0):
        sc = self.screen
        P.panel(sc, (x0, y0, 422, 120), "Legend")
        items = [(P.TRUE_C, "robot TRUE pose (filled)"), (P.EST_C, "robot ESTIMATED pose (outline)"),
                 ((255, 255, 255), "GT victim ring  (evaluation only)"), (P.GOOD, "confirmed detection"),
                 (P.WARN, "tentative victim track"), (P.PATH_C, "A* path / target X"),
                 (P.FRONTIER_C, "frontier (filled = selected)"), ((110, 255, 140), "LiDAR returns")]
        for i, (c, l) in enumerate(items):
            cx, cy = x0 + 12 + (i % 2) * 205, y0 + 32 + (i // 2) * 21
            pygame.draw.rect(sc, c, (cx, cy + 3, 12, 12))
            P.text(sc, l, (cx + 18, cy), 17)

    # ================================================================ loop
    def handle_key(self, k):
        if self.state != "sim":
            if k == pygame.K_RETURN:
                self.start()
            elif k == pygame.K_ESCAPE:
                self.quit()
            return
        sim = self.sim
        keymap = {pygame.K_SPACE: sim.toggle_pause, pygame.K_n: sim.step_once, pygame.K_r: self._reset,
                  pygame.K_m: self._new_map, pygame.K_v: self.toggle_view, pygame.K_l: self.toggle_lidar,
                  pygame.K_p: self._cycle_nav, pygame.K_f: sim.cycle_fusion, pygame.K_EQUALS: sim.faster,
                  pygame.K_PLUS: sim.faster, pygame.K_MINUS: sim.slower, pygame.K_ESCAPE: self.to_menu}
        for i, s in enumerate(SENSOR_NAMES):
            keymap[pygame.K_1 + i] = (lambda s=s: sim.cycle_sensor(s))
        if k in keymap:
            keymap[k]()

    def click(self, pos):
        btns = self.menu_buttons if self.state == "menu" else self.sim_buttons + self.sensor_buttons
        for b in btns:
            if b.handle(pos):
                return True
        return False

    def frame(self):
        mouse = pygame.mouse.get_pos()
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.quit()
            elif ev.type == pygame.KEYDOWN:
                self.handle_key(ev.key)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self.click(ev.pos)
        if self.state == "menu":
            self.draw_menu(mouse)
        else:
            self.sim.update()
            self.draw_sim(mouse)
        pygame.display.flip()
        self.clock.tick(30)

    def run(self):
        while self.running:
            self.frame()
        pygame.quit()

    def screenshot(self, path):
        pygame.image.save(self.screen, path)
