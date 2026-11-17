"""Rollout for the H=30 policies: rollout_v2 (eased chunks + release handling) with
  - chunk 30 / replan every 20 actions (1.33 s of motion per stop instead of 0.67 s),
  - ramp 0.4 s (eval 04 aborts came at t~0.28 s, right after the 0.15 s ramp: accelerating too fast).
Same arguments as rlpj.vla.rollout; --chunk/--replan-every given explicitly still win."""
import sys

import rollout_smooth as RS
import rollout_v2  # noqa: F401  (release handling + eased segments)
import rlpj.vla.rollout as R

RAMP = 0.4


def make_segment(kin, q0_deg, targets_deg, dt, speed=R.NORMAL, name="policy"):
    return RS.EasedSegment(RS._stock(kin, q0_deg, targets_deg, dt, speed, name), ramp=RAMP)


R.make_policy_segment = make_segment

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--chunk" not in argv:
        argv += ["--chunk", "30"]
    if "--replan-every" not in argv:
        argv += ["--replan-every", "20"]
    sys.exit(R.main(argv))
