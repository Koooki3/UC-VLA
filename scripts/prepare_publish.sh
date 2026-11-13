#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the PUBLIC release tree in a throw-away clone: rewritten author identity, withheld docs removed,
# dangling references patched, verification run. The working repository is never modified and NO
# remote is created or pushed. Usage: prepare_publish.sh OUTDIR "Name" "noreply@address"
set -euo pipefail
OUT="$1"; NAME="$2"; EMAIL="$3"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-$SRC/.venv/bin/python}"
OLD_EMAILS=$(git -C "$SRC" log --format='%ae%n%ce' | sort -u)
[ ! -e "$OUT" ] || { echo "OUTDIR exists" >&2; exit 1; }
git clone -q --no-local "$SRC" "$OUT"
cd "$OUT"
: > /tmp/ucvla_mailmap.$$
for e in $OLD_EMAILS; do
  case "$e" in *noreply@anthropic.com) continue;; esac
  echo "$NAME <$EMAIL> <$e>" >> /tmp/ucvla_mailmap.$$
done
: > /tmp/ucvla_replace.$$
for e in $OLD_EMAILS; do
  case "$e" in *noreply@anthropic.com) continue;; esac
  echo "$e==>[redacted-personal-email]" >> /tmp/ucvla_replace.$$
done
"$PY" -m git_filter_repo --force --mailmap /tmp/ucvla_mailmap.$$ --replace-text /tmp/ucvla_replace.$$ \
  --invert-paths --path docs/16_BENCHMARK_READINESS.md --path docs/20_MODEL_LICENSE_REVIEW.md \
  --path docs/22_ROBOMME_SAMPLE.md --path docs/30_BACKBONE_MATRIX.md --path docs/32_BACKBONE_DATA_LICENCE_AUDIT.md --path PAUSE_NOTE.md
rm -f /tmp/ucvla_mailmap.$$ /tmp/ucvla_replace.$$
mkdir -p provenance
awk '{print}' .git/filter-repo/commit-map | "$PY" -c "
import sys,json
m={}
for l in sys.stdin.read().splitlines()[1:]:
    o,n=l.split(); m[o]=n
json.dump(m,open('provenance/sha_map.json','w'),indent=1)"
# patch references to withheld documents
for f in $(git ls-files '*.md' '*.py' '*.json'); do
  [ "$f" = provenance/sha_map.json ] && continue
  sed -i -E 's#docs/(16|20|22|30|32)([_A-Z]*[A-Za-z_.]*)?#docs/\1 (withheld from the public release)#g' "$f" || true
done
sed -i -E '/docs\/(16|20|22)_[A-Z_]+\.md` \|/d; /^\| `docs\/(16|20|22)/d' README.md
"$PY" scripts/package_manifest.py >/dev/null
git add -A
GIT_AUTHOR_NAME="$NAME" GIT_AUTHOR_EMAIL="$EMAIL" GIT_COMMITTER_NAME="$NAME" GIT_COMMITTER_EMAIL="$EMAIL" \
  git commit -q -m "Publication preparation: withhold third-party licence summaries (docs 16/20/22), add sha_map

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
echo "authors/committers:"; git log --format='%an <%ae> | %cn <%ce>' | sort | uniq -c
echo "tags:"; git tag
"$PY" scripts/repo_scan.py --strict
if git log -p --all | grep -q -F -f <(printf "%s\n" $OLD_EMAILS | grep -v noreply@anthropic.com); then echo "HISTORY STILL CONTAINS OLD EMAIL" >&2; exit 1; else echo "history scan: old e-mail absent"; fi
"$PY" scripts/package_manifest.py --check
git remote -v | wc -l | sed 's/^/remotes: /'
