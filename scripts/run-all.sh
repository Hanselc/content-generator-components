#!/bin/bash

set -e

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pids=()
names=()
cleanup_done=0
grace_seconds=30

cleanup() {
    # Second signal: force-kill everything immediately.
    if [[ $cleanup_done -gt 0 ]]; then
        echo "Force-killing all scripts..." >&2
        for pid in "${pids[@]}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        exit 130
    fi
    cleanup_done=1

    echo "Stopping all scripts (grace ${grace_seconds}s)..." >&2
    for pid in "${pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    # Give them a grace period to exit on their own.
    local waited=0
    while [[ $waited -lt $grace_seconds ]]; do
        local any_alive=0
        for pid in "${pids[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
        [[ $any_alive -eq 0 ]] && break
        sleep 1
        waited=$((waited + 1))
    done

    # Force-kill any survivors.
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force-killing pid $pid (did not stop in ${grace_seconds}s)" >&2
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    exit 130
}

trap cleanup INT TERM

run_script() {
    local script="$1"
    local name
    name="$(basename "$script")"
    local tag
    tag="$(printf '%s' "$name" | sed 's/^run-//; s/\.sh$//')"
    echo "Starting $name ..."
    bash "$script" > >(sed -u "s/^/[$tag] /") 2>&1 &
    pids+=($!)
    names+=("$name")
}

shopt -s nullglob
for script in "$dir"/run-*.sh; do
    [[ "$script" == "$dir/run-all.sh" ]] && continue
    run_script "$script"
done
shopt -u nullglob

if [[ ${#pids[@]} -eq 0 ]]; then
    echo "No run-*.sh scripts found." >&2
    exit 1
fi

echo
echo "Launched ${#pids[@]} script(s). Press Ctrl-C to stop them all."
echo

# Disable set -e for the wait loop: a signal-interrupted wait returns
# a value >128 and should be handled by the trap, not treated as a
# hard error. We also want to collect every child's exit status.
set +e
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[ok]   ${names[$i]}"
    else
        rc=$?
        if [[ $rc -gt 128 ]]; then
            # Interrupted by a signal; let the trap handle cleanup.
            exit $rc
        fi
        echo "[exit $rc] ${names[$i]}"
        failed=1
    fi
done

exit $failed