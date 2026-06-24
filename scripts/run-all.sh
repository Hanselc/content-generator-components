#!/bin/bash

set -e

dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pids=()
names=()

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

failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[ok]   ${names[$i]}"
    else
        echo "[exit $?] ${names[$i]}"
        failed=1
    fi
done

exit $failed