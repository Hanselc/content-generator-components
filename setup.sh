#!/bin/bash

set -e

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$root/tools/comfy-ui/.git" ]; then
    rm -rf "$root/tools/comfy-ui"
    git clone https://github.com/Comfy-Org/ComfyUI.git "$root/tools/comfy-ui"
    # Allow parent repo to track user workflows/settings (upstream ignores /user/)
    sed -i '/^\/user\//d' "$root/tools/comfy-ui/.gitignore"
    git -C "$root" checkout -- tools/comfy-ui/
    echo "ComfyUI ready."
else
    echo "ComfyUI already present."
fi

# 2. Conda envs (workspace-local under .conda-envs/)
conda activate base 2>/dev/null || source "$(conda info --base)/etc/profile.d/conda.sh"

for env in comfyui moviepy voxcpm; do
    prefix="$root/.conda-envs/$env"
    if [ ! -d "$prefix" ]; then
        echo "Creating conda env '$env' at $prefix ..."
        conda env create -p "$prefix" -f "$root/envs/$env.yml"
    else
        echo "Updating conda env '$env' at $prefix ..."
        conda env update -p "$prefix" -f "$root/envs/$env.yml" --prune
    fi
done

# 3. Download VoxCPM2 weights and patch KV-cache max_length for low-VRAM GPUs.
#    The static KV cache is pre-allocated at init from config.json's max_length
#    (default 8192). Shrinking it to 2048 (~4x) saves a meaningful chunk of
#    VRAM on 8 GB cards. Pairs with VOXCPM_MAX_LEN=2000 in the server config.
#    NOTE: if you switch MODEL_NAME to a different model, re-run this block
#    against that model id.
echo "Preparing VoxCPM2 model (download + low-VRAM patch) ..."
conda run -p "$root/.conda-envs/voxcpm" --no-capture-output python - <<'PY'
from huggingface_hub import snapshot_download
import json
import os

path = snapshot_download("openbmb/VoxCPM2")
cfg_path = os.path.join(path, "config.json")
with open(cfg_path) as f:
    cfg = json.load(f)
if cfg.get("max_length", 0) > 2048:
    cfg["max_length"] = 2048
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Patched max_length -> 2048 in {cfg_path}")
else:
    print(f"max_length already <= 2048 in {cfg_path}")
PY

echo "Setup complete. Run: bash scripts/run-all.sh"
