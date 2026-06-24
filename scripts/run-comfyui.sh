#!/bin/bash

set -e

exec stdbuf -oL -eL conda run -n comfyui --no-capture-output \
    bash -c 'cd /opt/tools/ComfyUI && exec python -u main.py --enable-manager --listen 0.0.0.0'
