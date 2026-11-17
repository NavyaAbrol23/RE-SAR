# DEVELOPMENT_STATUS

## Completed (implemented, executed, tested)
- Environment: 3 disaster profiles, procedural maps, hazards, victims/decoys, seeds (environment/)
- Six sensors + centralised degradation controller (sensors/)
- Hybrid reliability: environment prior, observation quality, health, cross-modal, fusion (reliability/)
- Optional B6 learned reliability calibrator (reliability/learned.py, PyTorch, device auto)
- EKF + scan matching localisation, log-odds occupancy, sensed hazard/traversability, uncertainty (mapping/)
- Victim detection (RGB/thermal/audio), association, reliability-weighted tracking (perception/)
- Costmap, A*, frontier exploration (random/nearest/info-gain), safety controller (navigation/)
- Gymnasium SAREnvironment (passes gymnasium check_env), PPO train/evaluate/policy loader (rl/)
- Pygame menu + dashboard; every button verified headlessly (tests/test_gui.py)
- Experiments 1-6 scripts + paired statistics + plots (experiments/)
- Docs: README, technical specification, methodology, limitations, experimental results

## Executed
- pytest: 70 passed; gymnasium check_env passed; GUI rendered headless for all 3 disasters + PPO mode
- PPO fixed & hybrid trained 25k steps each (CPU); B6 trained; Exp 1-6 + rl.evaluate executed (~1,550 missions)

## Known limitations
See docs/limitations.md. GUI window and CUDA training could not be executed in the dev sandbox
(no display, no GPU); headless rendering and CPU training were executed instead.

## Next actions (optional)
- Train PPO for >=100k steps on the RTX 4060 laptop and re-run Exp 5/6.
