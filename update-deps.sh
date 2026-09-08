#!/bin/sh
# Dependency update, both package managers this repo uses. Every resolution — here or ad-hoc —
# is gated by the 48h supply-chain cooldown, declared per manager: [tool.uv] exclude-newer in
# pyproject.toml, install.minimumReleaseAge in each bun package's bunfig.toml. evidence/vendor/ is updated by hand.
set -e
cd "$(dirname "$0")"

uv lock --upgrade
# A plain sync removes extras; a deployment that opted into the local mlx server keeps it.
EXTRAS=""
if .venv/bin/python -c 'import mlx_vlm' >/dev/null 2>&1; then EXTRAS="--extra mlx"; fi
uv sync $EXTRAS

for pkg in evidence/distill recorder store; do
  (cd "$pkg" && bun update)
done

echo "python and bun dependencies updated under the 48h cooldown"
