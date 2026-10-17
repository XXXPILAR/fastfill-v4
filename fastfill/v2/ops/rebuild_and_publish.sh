#!/bin/bash
# Rebuild the main training set with the current main, upload it to HF, and hand it to the server autopilot.
# Usage: rebuild_and_publish.sh <name>   (e.g. main-20261007b)
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
NAME=$1
MAIN=/Users/slian/Desktop/3D/Worldedge/OptiScene
REPO=${SRC:-$MAIN}; cd $REPO  # SRC: a worktree pinned to the commit to build from
test -z "$(git status --porcelain --untracked-files=no -- fastfill)" || { echo "fastfill has uncommitted changes"; exit 1; }
COMMIT=$(git rev-parse HEAD)
PY=/Users/slian/miniforge3/bin/python
R=$MAIN/.release/v3.2; E=$MAIN/.release/audits/2026-09-28/source-check
OUT=$MAIN/outputs/fastfill_v2/rebuild-$NAME
test ! -e $OUT || { echo "$OUT exists"; exit 1; }
mkdir -p $OUT/reports
s(){ echo "STEP $1 $(date +%H:%M:%S)"; }
s 1; $PY -m fastfill.v2.legacy_build --release-root $R --evidence-root $E --front-policy axis --workers 8 --output $OUT/bridge > $OUT/reports/1-bridge.log 2>&1
s 2; $PY -m fastfill.v2.legacy_verify --data-root $OUT/bridge --output $OUT/reports/legacy-verify.json > $OUT/reports/2-verify.log 2>&1
s 3; $PY -m fastfill.v2.review_data build --parent-root $OUT/bridge --output $OUT/reviewed > $OUT/reports/3-review.log 2>&1
     $PY -m fastfill.v2.review_data verify --parent-root $OUT/bridge --data-root $OUT/reviewed >> $OUT/reports/3-review.log 2>&1
s 4; $PY -m fastfill.v2.qualified_data --parent-root $OUT/reviewed --output $OUT/main --ir-root $R/ir --frozen-manifest $R/data/v3.2/MANIFEST.json --pin-current-parent > $OUT/reports/4-main.log 2>&1
s 5; $PY -m fastfill.v2.multisource_verify --full-condition --parent-root $OUT/reviewed --data-root $OUT/main --ir-root $R/ir --parent-manifest $OUT/reviewed/manifest.json --output $OUT/reports/main-verify.json > $OUT/reports/5-main-verify.log 2>&1
$PY -c "import json,sys; r=json.load(open('$OUT/reports/main-verify.json')); sys.exit(0 if r.get('ok') else 1)" || { echo "main verify failed"; exit 1; }
s 6-upload
REV=$($PY - <<PYEOF
from huggingface_hub import HfApi
info = HfApi().upload_folder(folder_path="$OUT", path_in_repo="rebuild-$NAME", repo_id="liantian/fastfill-v2", repo_type="dataset",
    allow_patterns=["main/*", "reports/*.json", "bridge/manifest.json", "reviewed/manifest.json"],
    commit_message="rebuild-$NAME: main view built by fastfill $COMMIT")
print(info.oid)
PYEOF
)
$PY - <<PYEOF
import hashlib, json
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
spec = {"name": "$NAME", "repo": "liantian/fastfill-v2", "revision": "$REV", "folder": "rebuild-$NAME/main", "commit": "$COMMIT",
        "sha256": {n: sha("$OUT/main/" + n) for n in ("train.jsonl", "validation.jsonl", "test.jsonl")}}
open("$OUT/NEXT_DATA.json", "w").write(json.dumps(spec, indent=2) + "\n")
print(json.dumps(spec))
PYEOF
test -z "${NO_HANDOFF:-}" || { echo "BUILT $NAME $REV (handoff held)"; exit 0; }
scp -q $OUT/NEXT_DATA.json yxd-dev:/home/jovyan/shanliantian/fastfill/data/.NEXT_DATA.json.tmp && ssh -o BatchMode=yes yxd-dev "mv /home/jovyan/shanliantian/fastfill/data/.NEXT_DATA.json.tmp /home/jovyan/shanliantian/fastfill/data/NEXT_DATA.json"
echo "PUBLISHED $NAME $REV"
