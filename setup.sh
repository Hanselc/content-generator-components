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

# voxcpm needs a post-install step (phase 2: nano-vllm-voxcpm + low-VRAM patch)
# after the env is created. Track whether it was newly created so we can run
# that step after the generic loop.
voxcpm_prefix="$root/.conda-envs/voxcpm"
voxcpm_created=false

for env in comfyui moviepy voxcpm; do
    prefix="$root/.conda-envs/$env"
    if [ ! -d "$prefix" ]; then
        echo "Creating conda env '$env' at $prefix ..."
        conda env create -p "$prefix" -f "$root/envs/$env.yml"
        if [ "$env" = "voxcpm" ]; then
            voxcpm_created=true
        fi
    else
        echo "Updating conda env '$env' at $prefix ..."
        conda env update -p "$prefix" -f "$root/envs/$env.yml" --prune
    fi
done

# Phase 2 for voxcpm: build flash-attn + apply low-VRAM patch.
# Only needed once (after initial env creation). Re-running on an already-
# configured env is harmless (pip skips installed packages, patch is idempotent).
if [ "$voxcpm_created" = true ]; then
    bash "$root/envs/scripts/setup_voxcpm_post_install.sh" "$voxcpm_prefix"
else
    echo "voxcpm env already configured (post-install step skipped)."
fi

echo "Setup complete. Run: bash scripts/run-all.sh"