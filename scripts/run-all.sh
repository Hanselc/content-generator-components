#!/bin/bash

set -e

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pgids=()
names=()
cleanup_done=0
grace_seconds=30

cleanup() {
    # Second signal: force-kill everything immediately.
    if [[ $cleanup_done -gt 0 ]]; then
        echo "Force-killing all scripts..." >&2
        for pgid in "${pgids[@]}"; do
            kill -KILL -"$pgid" 2>/dev/null || true
        done
        exit 130
    fi
    cleanup_done=1

    echo "Stopping all scripts (grace ${grace_seconds}s)..." >&2
    # Signal the whole process group (negative pgid) so the signal
    # reaches conda run -> bash -c -> python directly, instead of
    # relying on conda run to forward it.
    for pgid in "${pgids[@]}"; do
        kill -TERM -"$pgid" 2>/dev/null || true
    done

    # Give them a grace period to exit on their own.
    local waited=0
    while [[ $waited -lt $grace_seconds ]]; do
        local any_alive=0
        for pgid in "${pgids[@]}"; do
            # Probe the process group as a whole; a dead leader alone
            # is not enough because conda run may exit before python.
            if kill -0 -"$pgid" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
        [[ $any_alive -eq 0 ]] && break
        sleep 1
        waited=$((waited + 1))
    done

    # Force-kill any survivors (whole group).
    for pgid in "${pgids[@]}"; do
        if kill -0 -"$pgid" 2>/dev/null; then
            echo "Force-killing pgid $pgid (did not stop in ${grace_seconds}s)" >&2
            kill -KILL -"$pgid" 2>/dev/null || true
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
    # setsid puts the child in its own session/process group so that
    # kill -- -<pgid> hits the wrapper + conda + python together.
    setsid bash "$script" > >(sed -u "s/^/[$tag] /") 2>&1 &
    local pid=$!
    pgids+=("$pid")
    names+=("$name")
}

shopt -s nullglob
for script in "$dir"/run-*.sh; do
    [[ "$script" == "$dir/run-all.sh" ]] && continue
    run_script "$script"
done
shopt -u nullglob

if [[ ${#pgids[@]} -eq 0 ]]; then
    echo "No run-*.sh scripts found." >&2
    exit 1
fi

echo
echo "Launched ${#pgids[@]} script(s). Press Ctrl-C to stop them all."
echo

# Disable set -e for the wait loop: a signal-interrupted wait returns
# a value >128 and should be handled by the trap, not treated as a
# hard error. We also want to collect every child's exit status.
set +e
failed=0
for i in "${!pgids[@]}"; do
    if wait "${pgids[$i]}"; then
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