"""Robot program: model-checked joint move to the Kinova Home pose (run through run_attempt: scripts/run_go_home.py)."""
import numpy as np

from rlpj.robot.session import RobotSession

HOME = np.array([0.0, 15.0, 230.0, 0.0, 55.0, 90.0])

with RobotSession(program="go_home") as s:
    s.phase("home")
    try:
        s.move_joint("to_home", HOME, "slow")
    except Exception as e:                       # a wide joint sweep can leave the cell: straight tool line first
        s.decide(f"joint move refused ({str(e)[30:90]}); straight line to the Home tool pose, then the joints")
        s.move_line("line_to_home", s.kin.fk(HOME), "slow")
        s.move_joint("to_home", HOME, "slow")
    s.claim("success")
