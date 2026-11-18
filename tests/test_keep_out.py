"""Keep-out boxes of the model check (rlpj.robot.sim.in_boxes), with the drawer task's cabinet box."""
import numpy as np
import yaml

from rlpj import config
from rlpj.robot.sim import in_boxes

TASK = yaml.safe_load((config.ROOT / "config/tasks/drawer_rl.yaml").read_text())
B = [(b["name"], np.asarray(b["center"]), np.asarray(b["half"]), np.radians(b["yaw_deg"])) for b in TASK["keep_out"]]
N = np.asarray(TASK["fixtures"]["cabinet"]["front_normal"])
T = np.array([-N[1], N[0]])
OFF = TASK["fixtures"]["cabinet"]["front_offset"]


def at(d, t, h):
    """Point at depth d behind the cabinet front plane, t along it, h above the table."""
    xy = (OFF + d) * N + t * T
    return np.array([[xy[0], xy[1], -0.016 + h]])


def test_inside_cabinet_is_refused():
    assert in_boxes(at(0.05, -0.08, 0.15), B) == ["cabinet"]


def test_front_of_cabinet_is_allowed():
    assert in_boxes(at(-0.02, -0.08, 0.05), B) == []          # pushing at the front face
    assert in_boxes(at(0.005, -0.08, 0.05), B) == []         # touching the front plane


def test_slot_opening_is_open_to_5cm():
    assert in_boxes(at(0.012, -0.08, 0.05), B) == []          # closed fingertips at the drawer face, modelled 1.2 cm deep
    assert in_boxes(at(0.011, -0.077, 0.029), B) == []        # reset attempt 16: holding the pull tab
    assert in_boxes(at(0.045, -0.08, 0.05), B) == []          # reaching a drawer recessed ~2.5 cm
    assert in_boxes(at(0.055, -0.08, 0.05), B) == ["cabinet"]
    assert in_boxes(at(0.015, -0.08, 0.15), B) == ["cabinet_front"]   # same depth above the slot: refused
    assert in_boxes(at(0.015, -0.14, 0.05), B) == ["cabinet_front"]   # beside the slot opening: refused


def test_beside_cabinet_is_allowed():
    assert in_boxes(at(0.05, -0.25, 0.05), B) == []


def test_escape_from_inside_is_allowed_deeper_is_not():
    from rlpj.robot.sim import box_depths
    inside, deeper, out = at(0.015, -0.08, 0.15), at(0.025, -0.08, 0.15), at(-0.02, -0.08, 0.15)
    d0, faces = box_depths(inside, B)
    assert box_depths(out, B, faces)[0] == {}                                         # leaving: nothing inside
    assert box_depths(deeper, B, faces)[0]["cabinet_front"] > d0["cabinet_front"] + 0.005   # 1 cm deeper: +1 cm
