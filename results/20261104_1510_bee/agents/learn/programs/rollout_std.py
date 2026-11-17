"""The stock rlpj.vla.rollout, unchanged (passing it as --rollout makes the orchestrator pull this workspace to the
robot machine, which the ckpt/ shim for PyTorch checkpoints needs). Same arguments."""
import sys

import rlpj.vla.rollout as R

if __name__ == "__main__":
    sys.exit(R.main())
