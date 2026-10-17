#!/bin/zsh
# Full download of the exported 3D room collections with `hf download`, one dataset after another (needs zsh).
#   fastfill/download_full.sh OUT_DIR [NAME ...]      (names as in fastfill/download.py SOURCES; default: all 22)
# Re-runnable: files already in OUT_DIR are skipped. A dataset is retried until `hf download` exits cleanly,
# because the Hub answers folders with tens of thousands of small files with transient 429/499 errors.
# Exits 1 on an unknown name or when a dataset still fails after 30 attempts.
OUT=${1:?usage: download_full.sh OUT_DIR [NAME ...]}; shift
OUT=${OUT:A}                                   # absolute before the cd below
HF=${HF_BIN:-hf}; PY=${PYTHON:-python}
cd "${0:A:h}/.." || exit 1
known=($($PY -c 'from fastfill.download import SOURCES; print(*SOURCES)')) || exit 1
for n in "$@"; do (( ${known[(Ie)$n]} )) || { echo "unknown dataset: $n (known: $known)"; exit 1; }; done
failed=()
$PY -c 'from fastfill.download import SOURCES
for n, (r, f) in SOURCES.items(): print(n, r, f or "-")' | while read name repo folder; do
  if (( $# )) && [[ ${@[(Ie)$name]} -eq 0 ]]; then continue; fi
  inc=(); [[ $folder != "-" ]] && inc=(--include "$folder/*")
  echo "== $name: $repo/$folder"
  ok=0
  for i in {1..30}; do
    $HF download $repo --repo-type dataset $inc --local-dir "$OUT/${repo/\//__}" --max-workers 4 >/dev/null && { ok=1; break; }
    echo "== $name: attempt $i failed, retrying in 60s"; sleep 60
  done
  if (( ok )); then echo "== $name: done"; else echo "== $name: FAILED after 30 attempts"; failed+=($name); fi
done
(( ${#failed} )) && { echo "== FAILED: $failed"; exit 1; }
echo "== FULL DOWNLOAD FINISHED"
