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

for env in comfyui moviepy; do
    prefix="$root/.conda-envs/$env"
    if [ ! -d "$prefix" ]; then
        echo "Creating conda env '$env' at $prefix ..."
        conda env create -p "$prefix" -f "$root/envs/$env.yml"
    else
        echo "Updating conda env '$env' at $prefix ..."
        conda env update -p "$prefix" -f "$root/envs/$env.yml" --prune
    fi
done

echo "Setup complete. Run: bash scripts/run-all.sh"