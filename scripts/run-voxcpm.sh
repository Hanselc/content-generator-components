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
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here —
# it changes the CUDA allocator in a way that causes the Nano-vLLM warmup
# forward pass to OOM on 8 GB GPUs. The reference project (run.py) does not
# set it and works correctly.
setsid stdbuf -oL -eL conda run -p "$env_prefix" --no-capture-output \
    bash -c "cd '$root/tools/vox-cpm' && exec python -u server.py --port $port --listen 0.0.0.0" &
child=$!

# Forward INT/TERM to the whole child process group, then re-raise.
forward() {
    kill -TERM -"$child" 2>/dev/null || true
}
trap forward INT TERM

wait "$child"