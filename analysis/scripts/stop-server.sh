#!/bin/sh
# Stop the MLX server, verified by process death — never by the port: uvicorn closes its listener before
# it waits on in-flight generation, which can hold tens of GiB wired long after the port reads free.
# SIGTERM (uvicorn drains), bounded wait, SIGKILL.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# paths.py resolves against the clone it is imported from, so read it from this script's own.
cd "$ROOT"
PIDFILE="$("$ROOT/../.venv/bin/python" -c 'from locus.analysis.mlx.paths import PIDFILE; print(PIDFILE)')"

echo "current timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [ ! -f "$PIDFILE" ]; then
    echo "no pidfile — no server to stop"
    exit 0
fi

PID="$(cat "$PIDFILE")"
if ! kill -0 "$PID" 2>/dev/null; then
    echo "pid $PID already dead — removing stale pidfile"
    rm -f "$PIDFILE"
    exit 0
fi

kill "$PID"
for _ in $(seq 1 15); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
done

if kill -0 "$PID" 2>/dev/null; then
    echo "pid $PID survived SIGTERM grace (in-flight generation) — SIGKILL"
    kill -9 "$PID"
    while kill -0 "$PID" 2>/dev/null; do sleep 1; done
fi

rm -f "$PIDFILE"
echo "server pid $PID dead, verified"
