#!/bin/sh
# Boot the MLX server detached and return when it serves — or fail loud with the boot log's tail.
# The server's own guards (the wired-limit sysctl, the one-resident-model pidfile) do the guarding;
# this owns only the mechanics: launch, log placement, readiness.
set -eu

[ $# -eq 1 ] || { echo "usage: start-server.sh <model-slug>   (slugs are the models.toml keys)" >&2; exit 1; }

# The package root the server runs from; the venv is the workspace's, above it. Where the server
# writes — the machine-level pidfile, the serve log under the deployment's data/ — is paths.py's.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/../.venv/bin/python"
# Everything below resolves against this script's own clone, whatever the caller's cwd.
cd "$ROOT"
# Host and port come from the slug's analysis card (base_url) — the same derivation boot uses.
HOST="$("$PY" -c "from locus.analysis.mlx.serve import bind_for; print(bind_for('$1')[0])")"
PORT="$("$PY" -c "from locus.analysis.mlx.serve import bind_for; print(bind_for('$1')[1])")"
PIDFILE="$("$PY" -c 'from locus.analysis.mlx.paths import PIDFILE; print(PIDFILE)')"
LOG="$("$PY" -c 'from locus.analysis.mlx.paths import serve_log; print(serve_log())')"
mkdir -p "$(dirname "$LOG")"

# Serving is judged by the pidfile changing hands to this boot AND the port answering — the port
# alone can answer because a prior resident server holds it, exactly the state the server's own
# guard refuses. Liveness of the boot is the process, never the port.
PRIOR="$(cat "$PIDFILE" 2>/dev/null || true)"

echo "current timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nohup uv run locus-mlx-serve "$1" >> "$LOG" 2>&1 &
BOOT=$!

# Model load is unbounded (a first run downloads weights), so wait as long as the boot lives.
while :; do
    CUR="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [ -n "$CUR" ] && [ "$CUR" != "$PRIOR" ] \
            && curl -sf "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
        echo "server pid $CUR serving on $HOST:$PORT (log: $LOG)"
        exit 0
    fi
    if ! kill -0 "$BOOT" 2>/dev/null; then
        echo "boot did not reach serving — $LOG tail:" >&2
        tail -20 "$LOG" >&2
        exit 1
    fi
    sleep 1
done
