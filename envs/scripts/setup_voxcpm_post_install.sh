#!/bin/bash
# Phase 2 of the voxcpm two-phase install.
#
# Usage: setup_voxcpm_post_install.sh <env_prefix>
#
# nano-vllm-voxcpm's transitive dep flash-attn is sdist-only and does
# `import torch` in setup.py, so it must be compiled with
# PIP_NO_BUILD_ISOLATION=1 against the in-env torch (conda's pip block uses
# build isolation, so it cannot be in the yml). This script:
#
#   1. Checks that the CUDA 12.6 toolkit (nvcc) is available.
#   2. Installs nano-vllm-voxcpm (builds flash-attn, ~5-10 min).
#   3. Applies the low-VRAM KV-cache patch (required on < 12 GB GPUs).
#   4. Verifies the result.
#
# Prerequisite: CUDA 12.6 toolkit at /usr/local/cuda-12.6 (or $CUDA_HOME).
# See envs/voxcpm.yml header for install instructions.
set -e

prefix="${1:?usage: setup_voxcpm_post_install.sh <env_prefix>}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.6}"

if [ ! -x "$cuda_home/bin/nvcc" ]; then
    echo "[ERR] CUDA 12.6 toolkit not found at $cuda_home/bin/nvcc" >&2
    echo "      flash-attn cannot be compiled without it." >&2
    echo "      Install it (see envs/voxcpm.yml header) and re-run this script." >&2
    exit 1
fi

echo "Installing nano-vllm-voxcpm (builds flash-attn, ~5-10 min) ..."
conda run -p "$prefix" --no-capture-output \
    env PIP_NO_BUILD_ISOLATION=1 MAX_JOBS=4 \
        CUDA_HOME="$cuda_home" PATH="$cuda_home/bin:$PATH" \
        pip install nano-vllm-voxcpm

echo "Applying low-VRAM KV-cache patch ..."
conda run -p "$prefix" --no-capture-output \
    python "$script_dir/apply_voxcpm_lowvram_patch.py" "$prefix"

echo "Verifying voxcpm env (nano-vllm-voxcpm + low-VRAM patch) ..."
conda run -p "$prefix" --no-capture-output python - "$prefix" <<'PY'
import sys
from pathlib import Path
env = Path(sys.argv[1])

try:
    import nanovllm_voxcpm
    import flash_attn
    print(f"[OK] nanovllm_voxcpm + flash_attn {flash_attn.__version__} import")
except ImportError as e:
    print(f"[ERR] nano-vllm-voxcpm not importable: {e}", file=sys.stderr)
    sys.exit(1)

mr = env / "lib/python3.11/site-packages/nanovllm_voxcpm/engine/model_runner.py"
if not mr.is_file():
    print(f"[ERR] model_runner.py not found at {mr}", file=sys.stderr)
    sys.exit(1)
text = mr.read_text()
if "max(1, int(available_for_kv)" in text:
    print(f"[OK] low-VRAM KV-cache patch applied ({mr})")
else:
    print(f"[WARN] low-VRAM patch NOT applied. Run:", file=sys.stderr)
    print(f"       python envs/scripts/apply_voxcpm_lowvram_patch.py {env}",
          file=sys.stderr)
    print("       (required on < 12 GB VRAM)", file=sys.stderr)
PY

echo "voxcpm post-install complete."