#!/usr/bin/env bash
# Stand up the `locus` CLI from a clone. Idempotent — re-run any time to pick up dependency changes.
set -euo pipefail

echo "current timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

missing=""
command -v uv  >/dev/null 2>&1 || missing="$missing uv"
command -v bun >/dev/null 2>&1 || missing="$missing bun"
command -v sqlite3 >/dev/null 2>&1 || missing="$missing sqlite3"
[ -z "$missing" ] || { echo "required tools not on PATH:$missing — install them and re-run" >&2; exit 1; }

# Each run brings the runtimes to that day's latest. An install owned by an external package
# manager refuses its own self-update and says so; the run continues on the version present.
uv self update || true
bun upgrade || true

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$(uv tool dir --bin)"

# A plain sync removes extras; a deployment that opted into the local mlx server
# (`uv sync --extra mlx`) keeps it across re-runs.
EXTRAS=""
if "$REPO/.venv/bin/python" -c 'import mlx_vlm' >/dev/null 2>&1; then EXTRAS="--extra mlx"; fi
uv sync --project "$REPO" $EXTRAS
bun install --frozen-lockfile --cwd "$REPO/evidence/distill"
mkdir -p "$BIN_DIR"
ln -sf "$REPO/.venv/bin/locus" "$BIN_DIR/locus"
uv run --project "$REPO" python -m playwright install chromium

case ":$PATH:" in
  *":$BIN_DIR:"*) echo "locus -> $(command -v locus)" ;;
  *) uv tool update-shell; echo "locus linked at $BIN_DIR/locus — PATH updated by uv; resolves in a new shell" ;;
esac

# Wire the standard AWS chain to the deployment's bucket, for S3 tooling outside Locus
# (aws, rclone). The credential helper owns the profile — what it writes, what stops a write,
# and the one disposition line it always prints.
(cd "$REPO" && .venv/bin/locus-r2-credentials --write-profile)
