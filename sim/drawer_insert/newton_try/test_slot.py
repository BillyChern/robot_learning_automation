"""Newton feasibility test: hollow drawer pushed into a cabinet slot by a PD-driven hand, N worlds on the GPU."""
import sys
import time

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorContact

N_WORLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
L, W, H, TH = 0.12, 0.10, 0.065, 0.01        # drawer outer size and wall thickness (m)
CLEAR = 0.005                                # lateral clearance (m)
FLOOR = 0.02                                 # slot floor height (m)
MASS = 0.3


def cabinet(b, cfg):
    """Static slot: floor, two side walls (posts), roof, back stop; opening at x = 0, depth along +x."""
    hw = W / 2 + CLEAR / 2
    boxes = [((0.08, 0, FLOOR / 2), (0.09, 0.09, FLOOR / 2)),
             ((0.08, hw + 0.01, FLOOR + 0.045), (0.09, 0.01, 0.045)),
             ((0.08, -hw - 0.01, FLOOR + 0.045), (0.09, 0.01, 0.045)),
             ((0.08, 0, FLOOR + 0.08 + 0.01), (0.09, 0.09, 0.01)),
             ((L + 0.002 + 0.005, 0, FLOOR + 0.04), (0.005, 0.07, 0.04))]
    for i, (c, h) in enumerate(boxes):
        b.add_shape_box(-1, xform=wp.transform(c, wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2], cfg=cfg, label=f"cab{i}")


def drawer_shapes(b, body, cfg):
    """Bottom, front, back, two sides; body origin at the drawer centre."""
    parts = [((0, 0, -H / 2 + TH / 2), (L / 2, W / 2, TH / 2)),
             ((-L / 2 + TH / 2, 0, 0), (TH / 2, W / 2, H / 2)),
             ((L / 2 - TH / 2, 0, 0), (TH / 2, W / 2, H / 2)),
             ((0, W / 2 - TH / 2, 0), (L / 2 - TH, TH / 2, H / 2)),
             ((0, -W / 2 + TH / 2, 0), (L / 2 - TH, TH / 2, H / 2))]
    for i, (c, h) in enumerate(parts):
        b.add_shape_box(body, xform=wp.transform(c, wp.quat_identity()), hx=h[0], hy=h[1], hz=h[2], cfg=cfg, label=f"drawer{i}")


def build():
    env = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(env)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.4, density=0.0)
    cabinet(env, cfg)
    x0 = -L / 2 - 0.03                                       # drawer centre at start: back 3 cm in front of the opening
    z0 = FLOOR + H / 2 + 0.002
    grip = wp.vec3(x0 - L / 2 + TH / 2, 0.0, z0 + H / 2)     # hand at the top of the front wall
    I_h = wp.mat33(np.eye(3) * 1e-4)
    hand = env.add_link(xform=wp.transform(grip, wp.quat_identity()), mass=0.2, inertia=I_h, label="hand")
    D = newton.ModelBuilder.JointDofConfig
    ke, kd = 2e4, 300.0
    j0 = env.add_joint_d6(-1, hand, parent_xform=wp.transform(grip, wp.quat_identity()),
                          linear_axes=[D(axis=a, target_ke=ke, target_kd=kd, target_q=0.0) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))],
                          angular_axes=[D(axis=(0, 0, 1), target_ke=50.0, target_kd=2.0, target_q=0.0)], label="hand_drive")
    dI = wp.mat33(np.diag([MASS * (W**2 + H**2) / 12, MASS * (L**2 + H**2) / 12, MASS * (L**2 + W**2) / 12]))
    drw = env.add_link(xform=wp.transform((x0, 0, z0), wp.quat_identity()), mass=MASS, inertia=dI, lock_inertia=True, label="drawer")
    drawer_shapes(env, drw, cfg)
    j1 = env.add_joint_d6(hand, drw, child_xform=wp.transform(wp.vec3(-L / 2 + TH / 2, 0, H / 2), wp.quat_identity()),
                          linear_axes=[D(axis=(0, 1, 0), stiffness=1500.0, damping=5.0, limit_lower=-0.01, limit_upper=0.01)],
                          angular_axes=[D(axis=(0, 0, 1), stiffness=3.0, damping=0.02, limit_lower=-0.2, limit_upper=0.2)],
                          label="in_hand")
    env.add_articulation([j0, j1])
    b = newton.ModelBuilder()
    b.replicate(env, N_WORLDS)
    return b.finalize(), x0


def run(case, steps, model, x0, verbose=True):
    """case: 'aligned' | 'yaw5' | 'off6' ; returns final drawer x, max drive force."""
    solver = newton.solvers.SolverMuJoCo(model, njmax=200, nconmax=80, cone="elliptic", impratio=10)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    nd = model.joint_dof_count // N_WORLDS
    q = s0.joint_q.numpy().reshape(N_WORLDS, -1)
    tq = ctrl.joint_target_q.numpy().reshape(N_WORLDS, nd)
    if case == "yaw5":    # hand (and drawer) turned 5 deg
        q[:, 3] = np.radians(5.0); tq[:, 3] = np.radians(5.0)
    if case == "off6":
        tq[:, 1] = 0.006; q[:, 1] = 0.006
    s0.joint_q.assign(q.ravel()); newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    sensor = SensorContact(model, sensing_shapes="*drawer*", request_contact_attributes=False)
    cp = newton.CollisionPipeline(model, rigid_contact_max=solver.get_max_contact_count(), soft_contact_max=0)
    contacts = cp.contacts()
    obs = solver.observables(contacts=contacts, kinds=sensor.solver_observable_kinds)
    v, x_end = 0.02, 0.153                                  # hand speed (m/s), target advance (presses 1 mm on the back stop)
    fmax = np.zeros(N_WORLDS); t0 = time.time()
    def chunk():
        for _ in range(25):
            solver.step(s0, s1, ctrl, contacts, DT, observables=obs)
            s0.assign(s1)
    chunk()                                                 # warm-up (compiles kernels)
    with wp.ScopedCapture() as cap:
        chunk()
    graph = cap.graph
    s0.joint_q.assign(q.ravel()); s0.joint_qd.zero_(); newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    wp.synchronize(); t0 = time.time()
    for k in range(0, steps, 25):
        tq[:, 0] = min(v * k * DT, x_end); ctrl.joint_target_q.assign(tq.ravel())
        wp.capture_launch(graph)
        sensor.update(s0, contacts, observables=obs)
        f = np.linalg.norm(sensor.total_force.numpy().reshape(N_WORLDS, -1, 3).sum(1), axis=-1)
        fmax = np.maximum(fmax, f)
    wp.synchronize(); el = time.time() - t0
    jq = s0.joint_q.numpy().reshape(N_WORLDS, -1)
    bq = s0.body_q.numpy().reshape(N_WORLDS, 2, 7)
    dx = bq[:, 1, 0] - x0
    if verbose:
        print(f"{case}: hand q {np.round(jq[0, :4], 4)}, in-hand y {jq[0, 4]:+.4f} yaw {np.degrees(jq[0, 5]):+.2f} deg, "
              f"drawer advance {dx.mean():.4f} (min {dx.min():.4f} max {dx.max():.4f}), drawer z {bq[:, 1, 2].mean():.4f}, "
              f"finite {np.isfinite(bq).all()}, max contact F {fmax.mean():.1f} N, "
              f"{steps * N_WORLDS / el:.0f} env-steps/s ({el:.1f} s)")
    return dx, fmax


if __name__ == "__main__":
    wp.init()
    model, x0 = build()
    print("worlds", model.world_count, "bodies", model.body_count, "shapes", model.shape_count, "dofs", model.joint_dof_count)
    steps = int(9.0 / DT)
    for case in ("aligned", "yaw5", "off6"):
        run(case, steps, model, x0)
