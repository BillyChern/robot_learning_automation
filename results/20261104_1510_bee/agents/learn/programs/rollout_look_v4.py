"""Policy from the look pose for v4 policies (joint deltas, gripper-only state, H=30): rollout_look (scripted camera
move to the look pose) + rollout_v4 (eased chunks, release handling, chunk 30 / replan 20, targets = q_now + delta), policy loop from rollout_floor (8 mm tip-floor prefix handling)."""
import sys

import rollout_look  # noqa: F401  (look move before the policy loop; handles --look-only)
import rollout_v4    # noqa: F401  (patches rlpj.vla.rollout)
import rollout_floor

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--chunk" not in argv:
        argv += ["--chunk", "30"]
    if "--replan-every" not in argv:
        argv += ["--replan-every", "20"]
    sys.exit(rollout_floor.main(argv))
