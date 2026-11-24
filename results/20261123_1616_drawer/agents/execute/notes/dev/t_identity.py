import json, os, sys, tempfile, types
os.environ["RLPJ_TASK"] = "drawer_lying"
W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; sys.path.insert(0, W + "/programs")
src = open(W + "/programs/execute.py").read().split("with RobotSession(program")[0].replace("from rlpj.robot.session import Abort, RobotSession", "class Abort(Exception): pass")
E = types.ModuleType("ep"); E.__file__ = W + "/programs/execute.py"; exec(compile(src, "h", "exec"), E.__dict__)
d = tempfile.mkdtemp(dir=W + "/dev_out"); os.environ["RLPJ_ATTEMPT_DIR"] = d; os.environ["RLPJ_EPISODE"] = "-1"; os.environ["RLPJ_RL_MODE"] = "eval"
for what, want in (("rl episode 11", ("train", 11)), ("rl_eval episode 12", ("eval", 12)), ("Real test of commit x", ("eval", -1)), ("rl episode", ("eval", -1))):
    json.dump(dict(what=what, commit="x"), open(d + "/attempt.json", "w")); got = E.run_identity(); print(repr(what), got, "ok" if got == want else "WRONG")
os.remove(d + "/attempt.json"); print("no attempt.json", E.run_identity())
for m in (0.0910, 0.0944, 0.0985, 0.0712, 0.113): print("measured", m, "-> used", E.grasp_face(m), "h_g", round(E.grasp_face(m) + E.TIP_CLEAR + E.TIP_CLOSED, 4))
