import json, os, sys, tempfile, types, cv2, numpy as np
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; sys.path.insert(0, W + "/programs")
import touch, vision as V
for n in (10, 26, 28, 30, 32):
    A = f"{W}/attempts/attempt_{n}/"; print("attempt", n, "cabinet shift px", round(V.cabinet_shift_px(cv2.imread(A + "side_before.jpg"), cv2.imread(A + "side_after.jpg")), 2))
b = cv2.imread(W + "/attempts/attempt_32/side_before.jpg"); m = np.float32([[1, 0, 9], [0, 1, 4]]); print("known shift (9, 4) px reads", round(V.cabinet_shift_px(b, cv2.warpAffine(b, m, (640, 360))), 2))
d = tempfile.mkdtemp(dir=W + "/dev_out"); json.dump(dict(centre=[0.0048, -0.0019, 0.75]), open(d + "/touch_centre.json", "w")); print("old-format file ->", touch.load_centre(d).tolist())
print("after an entry at (3.5, -2.5 mm, 0.3 deg) ->", np.round(touch.save_centre(d, [0.0035, -0.0025, 0.3], dict(episode=1)), 5).tolist(), "reloaded", np.round(touch.load_centre(d), 5).tolist())
from rlpj.success import checks
from rlpj import config
TASK = config.load("drawer_lying")["task"]; fx = TASK["fixtures"]["cabinet"]; y = np.radians(fx["yaw_deg"]); a_, b_ = np.array([np.cos(y), np.sin(y)]), np.array([-np.sin(y), np.cos(y)]); r = np.random.default_rng(0)
def blob(h): 
    s, t = r.uniform(-.03, .03, 1500), r.uniform(-.05, .05, 1500); xy = np.asarray(fx["target_centre"]) + s[:, None] * a_ + t[:, None] * b_; z = -0.0154 + h + r.normal(0, .001, 1500)
    return [dict(points=np.c_[xy, z], z_top=-0.0154 + 0.097, height=0.097, xy=[0.5, 0.01], extent=[0.28, 0.18], n=12000)]
print("flush compartment, no before blobs:", checks.drawer_lying([], blob(0.097), TASK)[0], "| empty:", checks.drawer_lying([], blob(0.03), TASK)[0])
os.environ["RLPJ_RL_DIR"] = d; import importlib; 
