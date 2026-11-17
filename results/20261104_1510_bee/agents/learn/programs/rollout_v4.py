"""Rollout for recipe_v4 policies (actions = joint deltas wrt the joints at inference time): rollout_v3 (chunk 30,
replan 20, 0.4 s eased chunks, release handling) with targets = q_now + delta. Same arguments as rlpj.vla.rollout."""
import sys

import numpy as np

import rollout_v3  # noqa: F401  (eased segments, release handling)
import rlpj.vla.rollout as R


def deltas_to_targets_deg(actions_rad, q_now_deg):
    d = np.rad2deg(np.asarray(actions_rad, float).reshape(-1, 6))
    return np.asarray(q_now_deg, float).reshape(1, 6) + d


R.actions_to_targets_deg = deltas_to_targets_deg

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--chunk" not in argv:
        argv += ["--chunk", "30"]
    if "--replan-every" not in argv:
        argv += ["--replan-every", "20"]
    sys.exit(R.main(argv))
