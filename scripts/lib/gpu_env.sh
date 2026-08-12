#!/bin/bash
# Resolve MuJoCo EGL + PyTorch to the same physical GPU.
#
# CUDA runtime ordinals and nvidia-smi indices can disagree on heterogeneous
# hosts. Bind CUDA by the physical GPU UUID selected with nvidia-smi, while EGL
# keeps the physical integer index. This makes GPU_ID unambiguous on every host.
#
# Usage (after setting GPU_ID):
#   source scripts/lib/gpu_env.sh
#   setup_gpu_env
#   # exports CUDA_VISIBLE_DEVICES, TORCH_DEVICE=cuda:0, MUJOCO_GL, MUJOCO_EGL_DEVICE_ID

# Use the interpreter from the active environment. Server-specific absolute
# paths break as soon as the same repository is used on another host.
PYTHON=${PYTHON:-$(command -v python || true)}
if [ -z "${PYTHON}" ]; then
    echo "[error] python is not available in PATH; activate the victim environment first" >&2
    return 1 2>/dev/null || exit 1
fi
export PYTHON

# dm_control 1.0.28 is only compatible with mujoco 3.3.x (see requirements.txt).
# Call before DMC / dmc_subtle jobs to fail fast instead of in worker subprocesses.
verify_dmc_stack() {
    "${PYTHON}" - <<'PY'
import sys

try:
    import mujoco
    from dm_control import suite
except ImportError as exc:
    print(f"[error] DMC stack import failed: {exc}")
    sys.exit(1)

ver = mujoco.__version__
if not ver.startswith("3.3."):
    print(f"[error] mujoco=={ver} is incompatible with dm_control 1.0.28")
    print("        Fix: pip install 'mujoco==3.3.0'")
    sys.exit(1)

try:
    import numpy as np

    env = suite.load("finger", "spin")
    env.reset()
    env.step(np.zeros(env.action_spec().shape, dtype=np.float32))
    frame = env.physics.render(height=64, width=64, camera_id=0)
    if frame.shape != (64, 64, 3):
        raise RuntimeError(f"unexpected render shape: {frame.shape}")
except Exception as exc:
    print(f"[error] DMC reset/step/EGL render smoke test failed: {exc}")
    sys.exit(1)

print(f"[env] DMC stack OK (mujoco {ver}, EGL render {frame.shape})")
PY
}

resolve_egl_gpu_id() {
    local cuda_visible="${1}"
    _CUDA_VISIBLE_DEVICES="${cuda_visible}" "${PYTHON:-python}" - <<'PY'
import os
import subprocess
import sys

cuda_vis = os.environ.get("_CUDA_VISIBLE_DEVICES", "0").split(",")[0]
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_vis

try:
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            text=True,
        )
        for line in out.strip().splitlines():
            idx, dev_name = [part.strip() for part in line.split(",", 1)]
            if dev_name == name:
                print(idx)
                sys.exit(0)
except Exception:
    pass

print(cuda_vis)
PY
}

setup_gpu_env() {
    GPU_ID=${GPU_ID:-0}
    local gpu_uuid
    gpu_uuid="$(
        nvidia-smi --id="${GPU_ID}" --query-gpu=uuid --format=csv,noheader \
            | head -n 1 | tr -d '[:space:]'
    )"
    if [ -z "${gpu_uuid}" ]; then
        echo "[error] cannot resolve physical GPU_ID=${GPU_ID} to a UUID" >&2
        return 1 2>/dev/null || exit 1
    fi
    export CUDA_VISIBLE_DEVICES=${gpu_uuid}
    export TORCH_DEVICE=cuda:0
    export MUJOCO_GL=egl
    export MUJOCO_EGL_DEVICE_ID=${GPU_ID}
    echo "[gpu] physical_index=${GPU_ID} uuid=${gpu_uuid} torch=${TORCH_DEVICE} egl=${MUJOCO_EGL_DEVICE_ID}"
}
