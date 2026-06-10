#!/usr/bin/env bash
# Backup the irreplaceable training state: the newest brain per family, the cognitive
# memory, and the run config. Checkpoint history and Obsidian notes are NOT included —
# notes regenerate from memory, and old step-snapshots are dead weight (see doom-cli prune).
#
#   ./scripts/backup_vault.sh                  # -> backups/hellmind-YYYYmmdd-HHMMSS.tar.gz
#   ./scripts/backup_vault.sh /Volumes/ext     # -> custom destination dir
set -euo pipefail

cd "$(dirname "$0")/.."
DEST="${1:-backups}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="${DEST}/hellmind-${STAMP}.tar.gz"
mkdir -p "$DEST"

# Newest checkpoint + _final per brain family (the only files resume ever loads).
BRAINS=$(ls -t vault/.checkpoints/*_final.zip 2>/dev/null | head -5)
LATEST=$(ls -t vault/.checkpoints/*_steps.zip 2>/dev/null | head -3)

tar -czf "$OUT" \
    --exclude='*.plateau_l4_*' \
    vault/.memory \
    .env \
    $BRAINS $LATEST 2>/dev/null

SIZE=$(du -h "$OUT" | cut -f1)
echo "backup: $OUT ($SIZE)"
echo "contents: cognitive memory + $(echo "$BRAINS $LATEST" | wc -w | tr -d ' ') brain files + .env"
