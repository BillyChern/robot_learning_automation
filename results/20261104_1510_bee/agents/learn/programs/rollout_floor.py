"""Policy loop of rlpj.vla.rollout with fingertip-floor handling (operator 2026-11-08 14:00).

Differences from the stock loop (everything else unchanged; all helpers are looked up in rlpj.vla.rollout at call
time, so the patches of rollout_smooth / rollout_v2 / rollout_v4 apply):
  - tip margin 8 mm with the gripper open (the session's own check); -2 mm once it is closed, as execute.py's lift (operator, ep 388);
  - a chunk that fails the model check ONLY on tip clearance (it passes with the tip check disabled) runs its longest
    prefix that passes (the arm holds at the floor); any other check failure aborts as before;
  - gripper: while open, a close anywhere in the chunk closes (in the demos the close comes right at the bottom of the
    descent, i.e. in the actions the floor cuts off); opening still uses the last executed action;
  - stop ("fail") after 3 chunks in a row held at the floor without a close. Never closes for the policy.
Never claims success."""
import argparse
import logging
import os
import time

import numpy as np

import rlpj.vla.rollout as R
import rollout_smooth as RS

EASE_RAMP = 0.4   # eased chunks as rollout_v3: ep 394 (full_v5) was stopped mid-carry by a 1.04 deg tracking deviation
                  # at t=0.21 s of a chunk started from rest (operator)


def _eased_segment(kin, q0_deg, targets_deg, dt, speed=R.NORMAL, name="policy"):
    return RS.EasedSegment(RS._stock(kin, q0_deg, targets_deg, dt, speed, name), ramp=EASE_RAMP)


R.make_policy_segment = _eased_segment

TIP_MARGIN = 0.008
CLOSED_TIP_MARGIN = -0.002   # gripper closed: as execute.py's LIFT_TIP_MARGIN (model puts closed tips 47 mm below
                             # the tool, real ~38 mm); 8 mm held every lift from the grasp height (ep 388, operator)
MAX_HELD = 3


def main(argv=None):
    from rlpj import config
    cfg = config.load()
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=cfg["hosts"]["policy_server"])
    p.add_argument("--port", type=int, default=cfg["hosts"]["policy_port"])
    p.add_argument("--instruction", default=cfg["task"]["instruction"])
    p.add_argument("--max-seconds", type=float, default=cfg["evaluation"]["max_policy_seconds"])
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--replan-every", type=int, default=10)
    p.add_argument("--fps", type=float, default=cfg["recording"]["fps"])
    p.add_argument("--settle-tol", type=float, default=0.5)
    p.add_argument("--dry-chunks", type=int, default=3)
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = R.log

    from rlpj.robot import calib, sim
    from rlpj.robot.kin import Kin
    from rlpj.robot.session import RobotSession

    motion = os.environ.get("RLPJ_MOTION") == "1"
    st = dict(chunks=0, steps=0, infer_ms=[], max_dev_deg=0.0, max_dev_mm=0.0, gripper_events=0, check_fail=0,
              floor_holds=0, stop_reason="", time_scale_max=1.0)
    client = R.PolicyClient(a.host, a.port, log=log.info)
    t_start = time.time()
    with RobotSession(program="rollout") as s:
        s.phase("policy")
        kin = getattr(s, "kin", None) or Kin.load()
        table_z = calib.table_z()
        fb = s.feedback()
        latch = R.GripperLatch(closed=np.nan_to_num(fb["gripper"]) / 100.0 > 0.5)
        s.decide(f"policy {a.host}:{a.port} chunk={a.chunk} replan={a.replan_every} tip_margin={TIP_MARGIN} "
                 f"metadata={client.metadata} motion={motion}")
        t_end = t_start + a.max_seconds
        held_run = 0
        try:
            while True:
                if time.time() >= t_end:
                    st["stop_reason"] = "time limit"
                    break
                if not motion and st["chunks"] >= a.dry_chunks:
                    st["stop_reason"] = "dry run done"
                    break
                side, _, _ = s.side.latest()
                wrist, _ = s.wrist.latest()
                fb = s.feedback()
                s.require(side is not None and wrist is not None, "no camera frame")
                obs = R.make_obs(side, wrist, R.state_from_feedback(fb["q"], fb["gripper"]), a.instruction)
                out = client.infer(obs)
                st["infer_ms"].append(client.last_ms)
                actions = np.asarray(out["actions"], float)
                n_exec = max(1, min(a.replan_every, a.chunk, len(actions)))
                # plan from joints read after inference: the arm can still be settling from the last chunk, and the
                # session refuses a plan whose start is > 0.5 deg off (ep 390: 0.75 deg after ~0.2 s inference; operator)
                fb = dict(fb, q=s.feedback()["q"])
                targets = R.actions_to_targets_deg(actions[:n_exec, :6], fb["q"])
                # check exactly as the session will: its commanded gripper value and its table height (ep 422: the
                # pre-check used the half-closed reading 0.56, the session the closed command, and refused the chunk)
                grip = float(getattr(s, "grip_frac", np.clip(np.nan_to_num(fb["gripper"]) / 100.0, 0, 1)))
                tz = max(float(getattr(s, "table_z", table_z)), table_z)
                if grip > 0.5:
                    # closed: never below -2 mm and never lower than the tips are now, so a lift from a grasp deeper
                    # than the demos' passes and pushing down never does (ep 421: 290 holds at the grasp; operator)
                    q_now = np.asarray(fb["q"], float)
                    _, now = sim.check(np.vstack([q_now, q_now]), kin, R.NORMAL, gripper_frac=grip, table_z=tz, tip_margin=-1.0)
                    margin = min(CLOSED_TIP_MARGIN, float(now["min_tip_above_table"]) - 0.0005)
                else:
                    margin = TIP_MARGIN

                def build(k):
                    sg = R.make_policy_segment(kin, fb["q"], targets[:k], 1.0 / a.fps, R.NORMAL, name=f"policy_{st['chunks']}")
                    return sg, sim.check(sg.q, kin, R.NORMAL, gripper_frac=grip, table_z=tz, tip_margin=margin)

                seg, (ok, rep) = build(n_exec)
                n_run, held = n_exec, False
                if not ok:
                    ok_notip, rep2 = sim.check(seg.q, kin, R.NORMAL, gripper_frac=grip, table_z=tz, tip_margin=-1.0)
                    if not ok_notip:
                        st["check_fail"] += 1
                        st["stop_reason"] = "model check failed"
                        s.require(False, f"policy chunk {st['chunks']} failed the model check: {R._short(rep)}")
                    held, n_run, seg = True, 0, None
                    for k in range(n_exec - 1, 0, -1):
                        sg, (ok_k, _) = build(k)
                        if ok_k:
                            seg, n_run = sg, k
                            break
                    st["floor_holds"] += 1
                    log.info(f"chunk {st['chunks']}: below the {margin * 1000:.0f} mm tip floor; running {n_run}/{n_exec} actions")
                if seg is not None:
                    st["time_scale_max"] = max(st["time_scale_max"], seg.scale)
                    log.info(f"chunk {st['chunks']}: infer {client.last_ms:.0f} ms, T={seg.T:.2f}s x{seg.scale:.2f}, "
                             f"n={n_run}, gripper last {actions[max(n_run, 1) - 1, 6]:.2f} max {actions[:, 6].max():.2f}")
                    if motion:
                        res = s.run_segment(seg, settle_tol=a.settle_tol, tip_margin=margin) or {}
                        st["max_dev_deg"] = max(st["max_dev_deg"], float(res.get("max_dev_deg", 0.0)))
                        st["max_dev_mm"] = max(st["max_dev_mm"], float(res.get("max_dev_mm", 0.0)))
                        if not res.get("ok", True):
                            st["stop_reason"] = f"tracking: {res.get('reason', '')}"
                            s.require(False, f"policy chunk aborted: {res.get('reason', '')}")
                st["chunks"] += 1
                st["steps"] += n_run
                gmax = float(np.clip(actions[:, 6].max(), 0, 1))
                if not latch.closed and gmax > latch.hi:
                    g_cmd = latch.update(gmax)
                else:
                    g_cmd = latch.update(actions[max(n_run, 1) - 1, 6])
                closed_now = g_cmd is not None and g_cmd > 0.5
                if g_cmd is not None:
                    st["gripper_events"] += 1
                    if motion:
                        try:
                            s.gripper(g_cmd, "policy_close" if g_cmd > 0.5 else "policy_open")
                        except Exception as e:
                            # the session refuses an open on a held object outside the release zone before anything
                            # moves; skip that command and let the policy go on (ep 490 ended 65 s in; operator)
                            if type(e).__name__ != "Abort" or "release refused" not in str(e):
                                raise
                            latch.closed = True
                            st["release_refused"] = st.get("release_refused", 0) + 1
                            s.decide(f"open refused outside the release zone; kept holding ({str(e)[:90]})")
                    else:
                        log.info(f"dry run: gripper -> {g_cmd:.2f}")
                # count only holds with the gripper open: after a close the demos wait at the bottom before the lift
                # (operator 14:25: ep 345 closed on the bee, then 3 quick held chunks aborted it)
                held_run = held_run + 1 if (held and not latch.closed and not closed_now) else 0
                if held_run >= MAX_HELD:
                    st["stop_reason"] = f"held at the tip floor {MAX_HELD} chunks in a row without a close"
                    s.decide(st["stop_reason"])
                    break
        finally:
            ms = np.asarray(st["infer_ms"]) if st["infer_ms"] else np.array([np.nan])
            s.summary(policy_server=f"{a.host}:{a.port}", instruction=a.instruction, chunks=st["chunks"],
                      steps=st["steps"], infer_ms_mean=round(float(np.nanmean(ms)), 1),
                      infer_ms_max=round(float(np.nanmax(ms)), 1), max_dev_deg=round(st["max_dev_deg"], 3),
                      max_dev_mm=round(st["max_dev_mm"], 2), gripper_events=st["gripper_events"],
                      check_fail=st["check_fail"], floor_holds=st["floor_holds"], release_refused=st.get("release_refused", 0),
                      time_scale_max=round(st["time_scale_max"], 2),
                      stop_reason=st["stop_reason"] or "interrupted", seconds=round(time.time() - t_start, 1))
            client.close()
        s.claim("fail")      # the orchestrator's verdict decides success
    return 0
