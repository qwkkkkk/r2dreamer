#!/bin/bash
# Resolve MuJoCo EGL + PyTorch to the same physical GPU.
#
# CUDA_VISIBLE_DEVICES and nvidia-smi ordinals can disagree on multi-GPU hosts.
# Match by device name so EGL rendering and torch share one card.
#
# Usage (after setting GPU_ID):
#   source scripts/gpu_env.sh
#   setup_gpu_env
#   # exports CUDA_VISIBLE_DEVICES, TORCH_DEVICE=cuda:0, MUJOCO_GL, MUJOCO_EGL_DEVICE_ID

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
    export CUDA_VISIBLE_DEVICES=${GPU_ID}
    export TORCH_DEVICE=cuda:0
    export MUJOCO_GL=egl
    export MUJOCO_EGL_DEVICE_ID
    MUJOCO_EGL_DEVICE_ID="$(resolve_egl_gpu_id "${GPU_ID}")"
    export MUJOCO_EGL_DEVICE_ID
}
