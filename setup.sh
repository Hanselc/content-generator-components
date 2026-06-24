#!/bin/bash

set -e

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$root/ComfyUI/.git" ]; then
    rm -rf "$root/ComfyUI"
    git clone https://github.com/Comfy-Org/ComfyUI.git "$root/ComfyUI"
    # Allow parent repo to track user workflows/settings (upstream ignores /user/)
    sed -i '/^\/user\//d' "$root/ComfyUI/.gitignore"
    git -C "$root" checkout -- ComfyUI/
    echo "ComfyUI ready."
else
    echo "ComfyUI already present."
fi

echo "Setup complete. Run: bash scripts/run-all.sh"