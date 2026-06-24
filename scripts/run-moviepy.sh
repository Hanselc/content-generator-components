#!/bin/bash

set -e

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_prefix="$dir/../.conda-envs/moviepy"
root="$(cd "$dir/.." && pwd)"

exec stdbuf -oL -eL conda run -p "$env_prefix" --no-capture-output \
    bash -c "cd '$root/MoviePy' && exec python -u server.py"