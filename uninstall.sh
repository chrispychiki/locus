#!/usr/bin/env bash
# Undo install.sh — remove the `locus` symlink. The venv, node_modules, store/.env, and the
# deployment's data/ live inside the clone and go when you delete it. Nothing remote is touched.
set -euo pipefail

echo "current timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$(uv tool dir --bin)"
LINK="$BIN_DIR/locus"

if [ -L "$LINK" ]; then rm "$LINK"; echo "removed symlink $LINK"; else echo "no locus symlink at $LINK"; fi

# Remove the aws profile install.sh wrote — only if it points into this clone; a profile
# belonging to another deployment stays.
AWS_CONFIG="${AWS_CONFIG_FILE:-$HOME/.aws/config}"
if [ -f "$AWS_CONFIG" ]; then
  block="$(awk '/^\[profile locus\][[:space:]]*$/{f=1;next} /^\[/{f=0} f' "$AWS_CONFIG")"
  case "$block" in
    *"$REPO"*)
      # The config is the operator's file, not ours: back it up beside itself before the one
      # rewrite this repo ever does to it, and write through the original path so a symlinked
      # config and its permissions survive.
      cp -p "$AWS_CONFIG" "$AWS_CONFIG.locus-uninstall-backup"
      tmp="$(mktemp)"
      awk '/^\[profile locus\][[:space:]]*$/{skip=1;next} /^\[/{skip=0} !skip' "$AWS_CONFIG" > "$tmp"
      cat "$tmp" > "$AWS_CONFIG"
      rm "$tmp"
      echo "removed aws profile 'locus' from $AWS_CONFIG (backup: $AWS_CONFIG.locus-uninstall-backup)" ;;
  esac
fi

echo "left alone: the render browser in playwright's shared cache (~/Library/Caches/ms-playwright), and bun/uv."
