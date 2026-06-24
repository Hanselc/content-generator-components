#!/bin/bash

set -e

exec stdbuf -oL -eL conda run -n moviepy --no-capture-output \
    bash -c 'cd /opt/tools/MoviePy && exec python -u server.py'