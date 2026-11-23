import sys, numpy as np
sys.path.insert(0, "/home/r2r/robot_learning_final_pj/runs/20261123_1616_drawer/agents/execute/programs")
import vision as V
r = np.random.default_rng(0); y = 18.2; a, b = V.axes(y); c = np.array([0.45, 0.003])
for ts, tt_ in ((3.0, -1.0), (0.0, 0.0), (-2.0, 1.5)):
    s, t = r.uniform(-.036, .036, 4000), r.uniform(-.056, .056, 4000)
    z = .17 + np.tan(np.radians(ts)) * s + np.tan(np.radians(tt_)) * t + r.normal(0, .0012, 4000)
    k = (np.abs(s + .02) < .012) & (np.abs(t) < .03); z[k] += .02 * r.random(k.sum())               # fingers above the face
    P = np.c_[c[0] + s * a[0] + t * b[0], c[1] + s * a[1] + t * b[1], z]
    o = V.plane_tilt(P, c, y, (.033, .053), .17, .014, exclude=(-.02, 0, .016, .032))
    print((ts, tt_), {k: o.get(k) for k in ("ok", "n", "tilt_s_deg", "tilt_t_deg", "resid_mm")})
