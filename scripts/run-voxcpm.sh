#!/bin/bash

set -e

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$dir/_lib.sh"

env_prefix="$dir/../.conda-envs/voxcpm"
root="$(cd "$dir/.." && pwd)"
port=8788

# Stop any instance already listening on our port.
stop_existing_by_port "$port"

# Launch the service in its own process group so that signals can be
# delivered to conda run -> bash -c -> python as a whole (conda run
# does not forward TERM on its own).
setsid stdbuf -oL -eL conda run -p "$env_prefix" --no-capture-output \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash -c "cd '$root/tools/vox-cpm' && exec python -u server.py --port $port --listen 0.0.0.0" &
child=$!

# Forward INT/TERM to the whole child process group, then re-raise.
forward() {
    kill -TERM -"$child" 2>/dev/null || true
}
trap forward INT TERM

wait "$child"