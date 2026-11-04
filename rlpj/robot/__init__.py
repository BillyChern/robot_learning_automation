"""Robot layer: calib, kortex, cams, kin, plan, sim, executor, home (hardware-verified) + session, recorder,
watchdog, liveview, fit_kin, check. Programs use rlpj.robot.session.RobotSession only."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")   # headless replay videos; must be set before mujoco is imported
