import sys; W = "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute"; exec(open(W + "/notes/dev/t_simarm_head.py").read())
for ea, eb in ((0, 0), (0.007, 0.002)):
    o, h, _, e, s = run(ea, eb, seed=1); print("error", ea, eb, "end", h["end"], "decisions", h["decisions"], "stops", h["stops"], "final bottom", h["final_bottom"])
    for x in e.sim_log[:60]:
        ob = x["obs"]
        if x["k"] < 34 or x["k"] > 52: print(f" k{x['k']:2d} z {ob['z']*1000:6.1f} x {ob['x']*1000:6.1f} y {ob['y']*1000:6.1f} f {np.round(ob['f'],1).tolist()} step mm {np.round(np.array(x['step'])*[1000,1000,1,1000],1).tolist()} {x['status']:5s} bottom {x['bottom']*1000:5.1f}")
