#!/bin/bash
set -e

git status --porcelain | grep -E '^(UU|AA)' | awk '{print $2}' > conflicts.txt
echo "Total conflicted files: $(wc -l < conflicts.txt)"

> identical_files.txt
> real_diff_files.txt

while IFS= read -r f; do
  git show ":2:$f" 2>/dev/null | tr -d '\r' > /tmp/ours_norm || continue
  git show ":3:$f" 2>/dev/null | tr -d '\r' > /tmp/theirs_norm || continue

  if diff -q /tmp/ours_norm /tmp/theirs_norm > /dev/null 2>&1; then
    echo "$f" >> identical_files.txt
  else
    echo "$f" >> real_diff_files.txt
  fi
done < conflicts.txt

echo
echo "=== Identical content (CRLF/whitespace-only diff): $(wc -l < identical_files.txt) files ==="
cat identical_files.txt
echo
echo "=== REAL content differences needing manual resolution: $(wc -l < real_diff_files.txt) files ==="
cat real_diff_files.txt
