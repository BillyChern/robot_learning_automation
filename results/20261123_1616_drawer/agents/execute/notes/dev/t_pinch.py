"""Why did the entry offset jump (a +3 mm in attempts 30-34, +9.8 in 36)? Pinch point vs the drawer: the outline centre against the metal label frame."""
import json, os, sys, cv2, numpy as np
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; sys.path.insert(0, W + "/programs")
import vision as V
from rlpj.robot.kin import Kin
kin = Kin.load(); tz = -0.0154
print("att  yaw    side size_s size_l  tab_s  frame_mid  frame_w | tab-from-frame  outline-centre-from-frame (toward the tab) | entry a,b mm")
for n in (10, 12, 14, 16, 18, 20, 24, 26, 28, 30, 32, 34, 36):
    A = f"{W}/attempts/attempt_{n}/"
    try:
        S = json.load(open(A + "summary.json")); O = S["objects"]; dr = O["drawer"]; face = tz + dr["face_h"]
        q = np.load(A + "plans/02_look_drawer.npz")["q"][-1]; T = kin.fk(q)
        im = cv2.resize(cv2.imread(A + "wrist_drawer.jpg"), (1280, 720), interpolation=cv2.INTER_CUBIC)
        d = V.find_drawer(im, T, np.array(dr["c"]), face)
        t = V.find_tab(im, T, d, face)
        if not t.get("ok"): print(n, "tab not found", t.get("why")); continue
        sg = t["side"]; fm = 0.5 * sum(t["frame_s"]); fw = t["frame_s"][1] - t["frame_s"][0]
        e = ""
        if n >= 30:
            ev = json.load(open(W + "/rl/transitions/" + os.path.basename(S["transitions_file"]).replace(".npz", ".json")))["events"]
            ins = [x for x in ev if x[4] <= 0.091]
            e = np.round(np.array(ins[0][5][:2]) * 1000, 1).tolist() if ins else ""
        print(n, f"{d['yaw']:7.1f} {sg:+d}  {d['size'][0]*1000:5.1f} {d['size'][1]*1000:6.1f} {t['s']*1000:6.1f} {fm*1000:6.1f} {fw*1000:6.1f} | {sg*(t['s']-fm)*1000:6.1f}  {-sg*fm*1000:6.1f} | {e}")
    except Exception as ex:
        print(n, "error", repr(ex)[:90])
