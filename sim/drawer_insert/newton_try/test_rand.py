"""Newton test 2: per-world clearance / friction / yaw randomization, force-limited straight push, partial reset."""
import sys
import time

import numpy as np
import warp as wp

import newton
from newton.sensors import SensorContact
import test_slot as T

F_STOP = 65.0


def main(n):
    T.N_WORLDS = n
    model, x0 = T.build()
    rng = np.random.default_rng(0)
    clear = rng.uniform(0.003, 0.010, n)
    yaw = rng.uniform(-6, 6, n)
    mu = rng.uniform(0.2, 0.7, n)
    # per-world geometry: shapes 1, 2 are the side walls (10 shapes per world, cabinet first)
    st = model.shape_transform.numpy().reshape(n, 10, 7)
    hw = T.W / 2 + clear / 2
    st[:, 1, 1] = hw + 0.01; st[:, 2, 1] = -hw - 0.01
    model.shape_transform.assign(st.reshape(-1, 7))
    m = model.shape_material_mu.numpy().reshape(n, 10); m[:] = mu[:, None]; model.shape_material_mu.assign(m.ravel())
    solver = newton.solvers.SolverMuJoCo(model, njmax=200, nconmax=80, cone="elliptic", impratio=10)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    nd = model.joint_dof_count // n
    q = s0.joint_q.numpy().reshape(n, nd); tq = ctrl.joint_target_q.numpy().reshape(n, nd)
    q[:, 3] = tq[:, 3] = np.radians(yaw)
    s0.joint_q.assign(q.ravel()); newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    sensor = SensorContact(model, sensing_shapes="*drawer*", request_contact_attributes=False)
    cp = newton.CollisionPipeline(model, rigid_contact_max=solver.get_max_contact_count(), soft_contact_max=0)
    contacts = cp.contacts()
    obs = solver.observables(contacts=contacts, kinds=sensor.solver_observable_kinds)

    def chunk():
        for _ in range(25):
            solver.step(s0, s1, ctrl, contacts, T.DT, observables=obs)
            s0.assign(s1)
    chunk()
    with wp.ScopedCapture() as cap:
        chunk()
    s0.joint_q.assign(q.ravel()); s0.joint_qd.zero_(); newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    x_cmd = np.zeros(n); stopped = np.zeros(n, bool); fmax = np.zeros(n); t0 = time.time()
    for k in range(int(9.0 / (25 * T.DT))):
        x_cmd = np.where(stopped, x_cmd, np.minimum(x_cmd + 0.02 * 25 * T.DT, 0.153))
        tq[:, 0] = x_cmd; ctrl.joint_target_q.assign(tq.ravel())
        wp.capture_launch(cap.graph)
        sensor.update(s0, contacts, observables=obs)
        f = np.linalg.norm(sensor.total_force.numpy().reshape(n, -1, 3).sum(1), axis=-1)
        jq = s0.joint_q.numpy().reshape(n, nd)
        f_drive = T_ke * np.abs(x_cmd - jq[:, 0])           # hand drive force along x (what a torque estimate sees)
        fmax = np.maximum(fmax, f_drive)
        stopped |= f_drive > F_STOP
    el = time.time() - t0
    bq = s0.body_q.numpy().reshape(n, 2, 7)
    adv = bq[:, 1, 0] - x0
    ok = adv > 0.152 - 0.01
    print(f"{n} worlds, {el:.1f} s, finite {np.isfinite(bq).all()}, inserted {ok.mean():.2f}, force-stopped {stopped.mean():.2f}")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 6)):
        sel = (np.abs(yaw) >= lo) & (np.abs(yaw) < hi)
        print(f"  |yaw| {lo}-{hi} deg: n {sel.sum()}, inserted {ok[sel].mean():.2f}, mean adv {adv[sel].mean():.3f}, "
              f"fmax {np.median(fmax[sel]):.0f} N, clearance of inserted {1000 * clear[sel & ok].mean() if (sel & ok).any() else 0:.1f} mm")
    # partial reset: worlds 0..n/2 back to the start, the rest untouched
    jq = s0.joint_q.numpy().reshape(n, nd).copy(); keep = jq[n // 2:].copy()
    jq[: n // 2] = q[: n // 2]; s0.joint_q.assign(jq.ravel())
    jd = s0.joint_qd.numpy().reshape(n, nd); jd[: n // 2] = 0; s0.joint_qd.assign(jd.ravel())
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    tq[: n // 2, 0] = 0.0; ctrl.joint_target_q.assign(tq.ravel())
    wp.capture_launch(cap.graph)
    jq2 = s0.joint_q.numpy().reshape(n, nd)
    print(f"partial reset: reset worlds hand x {np.abs(jq2[: n // 2, 0]).max():.4f} (want ~0), "
          f"kept worlds max change {np.abs(jq2[n // 2:] - keep).max():.5f}")


T_ke = 2e4
if __name__ == "__main__":
    wp.init()
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1024)
