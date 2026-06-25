#!/bin/bash
#
# Shared helpers for the run-*.sh service scripts.
# Source this file (do not execute it directly):
#   . "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
#

# Find PIDs of processes listening on a TCP port.
# Prints one PID per line on stdout. Returns 0 even if none found.
_port_pids() {
    local port="$1"
    if command -v ss >/dev/null 2>&1; then
        ss -ltnpH "sport = :$port" 2>/dev/null \
            | grep -oE 'pid=[0-9]+' \
            | grep -oE '[0-9]+' \
            | sort -u
    elif command -v fuser >/dev/null 2>&1; then
        fuser "$port/tcp" 2>/dev/null \
            | tr -s ' ' '\n' \
            | grep -E '^[0-9]+$' \
            | sort -u
    elif command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u
    fi
}

# Stop any process already listening on the given TCP port.
# Sends TERM, waits up to ${STOP_GRACE:-10}s, then KILLs survivors.
# Safe to call before launching a service.
stop_existing_by_port() {
    local port="$1"
    local grace="${STOP_GRACE:-10}"
    local self=$$
    local pids
    pids="$(_port_pids "$port")"
    [[ -z "$pids" ]] && return 0

    local to_kill=()
    local pid
    while read -r pid; do
        [[ -z "$pid" ]] && continue
        # Never kill ourselves or our own process group.
        [[ "$pid" == "$self" ]] && continue
        to_kill+=("$pid")
    done <<< "$pids"
    [[ ${#to_kill[@]} -eq 0 ]] && return 0

    echo "Stopping existing process(es) on port $port: ${to_kill[*]}" >&2
    for pid in "${to_kill[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done

    local waited=0
    while [[ $waited -lt $grace ]]; do
        local alive=0
        for pid in "${to_kill[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        [[ $alive -eq 0 ]] && break
        sleep 1
        waited=$((waited + 1))
    done

    for pid in "${to_kill[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force-killing pid $pid on port $port (did not stop in ${grace}s)" >&2
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}