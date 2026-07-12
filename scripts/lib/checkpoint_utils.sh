#!/bin/bash
# Shared helpers for deciding whether a run is already complete.

# Return 0 when CKPT_PATH exists and represents a finished run for TARGET_STEPS.
# Legacy checkpoints without train_step are treated as complete.
checkpoint_is_complete() {
    local ckpt="$1"
    local target="$2"
    if [ ! -f "${ckpt}" ]; then
        return 1
    fi
    "${PYTHON:-python}" - "${ckpt}" "${target}" <<'PY'
import sys
import os

ckpt_path, target = sys.argv[1], float(sys.argv[2])
try:
    import torch
except ImportError:
    sys.exit(0 if os.path.isfile(ckpt_path) else 1)
try:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
except Exception:
    sys.exit(1)
step = obj.get("train_step")
if step is None:
    sys.exit(0)
sys.exit(0 if int(step) >= int(target) else 1)
PY
}
