"""Newton test 3: does add_mjcf accept a mocap hand + weld equality and slide/hinge joints with stiffness?"""
import numpy as np
import warp as wp

import newton

XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="box" size="0.2 0.2 0.01" pos="0 0 -0.01"/>
    <body name="hand" mocap="true" pos="0 0 0.1"/>
    <body name="carrier" pos="0 0 0.1">
      <freejoint/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
      <body name="drawer" pos="0 0 -0.04">
        <joint name="slip_y" type="slide" axis="0 1 0" stiffness="1500" damping="5" range="-0.01 0.01"/>
        <joint name="slip_yaw" type="hinge" axis="0 0 1" stiffness="3" damping="0.02" range="-10 10"/>
        <geom name="drawer_b" type="box" size="0.06 0.05 0.005" mass="0.3"/>
      </body>
    </body>
  </worldbody>
  <equality><weld body1="hand" body2="carrier" solref="0.01 1"/></equality>
</mujoco>
"""

if __name__ == "__main__":
    wp.init()
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    env = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(env)
    try:
        env.add_mjcf(XML)
    except Exception as e:
        print("add_mjcf failed:", repr(e)[:300]); raise SystemExit
    b.replicate(env, 4)
    model = b.finalize()
    print("bodies", model.body_labels if hasattr(model, "body_labels") else model.body_label)
    print("joints", model.joint_label, "types", model.joint_type.numpy())
    print("body flags", model.body_flags.numpy() if hasattr(model, "body_flags") else None)
    solver = newton.solvers.SolverMuJoCo(model)
    print("mj nmocap", solver.mj_model.nmocap, "neq", solver.mj_model.neq, "njnt", solver.mj_model.njnt)
    s0, s1, c = model.state(), model.state(), model.control()
    for _ in range(500):
        solver.step(s0, s1, c, None, 0.002); s0, s1 = s1, s0
    print("body_q after 1 s", np.round(s0.body_q.numpy()[:3, :3], 4))
