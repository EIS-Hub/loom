#!/usr/bin/env bash
# Count hand-written changed lines of a chunk against its base: source only (no tests, prose,
# lockfiles, generated json). Fails above the cap so a chunk stays readable in one sitting.
set -euo pipefail
base="${1:?base ref}"; cap="${2:-150}"
n=$(git diff --numstat "$base...HEAD" -- . ':!tests/**' ':!*.md' ':!*.json' ':!*.lock' \
    | awk '{ if ($1 != "-") { a += $1; d += $2 } } END { print a + d + 0 }')
echo "hand-written changed lines: $n (cap $cap)"
if [ "$n" -gt "$cap" ]; then
  echo "::error::this chunk has $n hand-written lines; split it, or Gabriel adds the oversize-approved label"
  exit 1
fi
