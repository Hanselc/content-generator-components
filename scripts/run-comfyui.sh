#!/bin/bash

set -e

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_prefix="$dir/../.conda-envs/comfyui"
root="$(cd "$dir/.." && pwd)"

exec stdbuf -oL -eL conda run -p "$env_prefix" --no-capture-output \
    bash -c "cd '$root/tools/comfy-ui' && exec python -u main.py --enable-manager --listen 0.0.0.0"