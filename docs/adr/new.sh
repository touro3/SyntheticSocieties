#!/usr/bin/env bash
# Bootstrap a new ADR. Usage: bash docs/adr/new.sh "Short title here"
set -euo pipefail

TITLE="${1:-}"
if [ -z "$TITLE" ]; then
  echo "Usage: $0 \"Short imperative title\"" >&2
  exit 1
fi

cd "$(dirname "$0")"

# Find the next numeric prefix
last=$(ls 0[0-9][0-9][0-9]-*.md 2>/dev/null | sed -E 's/^0*([0-9]+)-.*/\1/' | sort -n | tail -1 || echo 0)
next=$((10#${last:-0} + 1))
printf -v num "%04d" "$next"

slug=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+|-+$//g')
file="${num}-${slug}.md"
today=$(date -u +%F)

cp template.md "$file"
sed -i "s/^# NNNN. .*/# ${num}. ${TITLE}/" "$file"
sed -i "s/^- Date: YYYY-MM-DD/- Date: ${today}/" "$file"

echo "Created docs/adr/${file}"
echo "Don't forget to add it to docs/adr/README.md index."
