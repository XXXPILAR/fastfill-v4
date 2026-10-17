"""Download the exported 3D room collections.

Default = aggregate files only (files at most 1 level below each export folder: the merged
json/jsonl exports, README, kits). --full adds per-scene copies and raw sources (~90k small files).
Files are fetched one by one with per-file backoff and skipped when already present with the right
size, so an interrupted run resumes; the Hub rate-limits bursts of small files (HTTP 429/499).

Usage:
    python -m fastfill.download --out /Volumes/harddisk/3D_Room_Collections
    python -m fastfill.download --out DIR --only IL3D 3D-FRONT --full [--with-meshes]
"""
import argparse
import fnmatch
import os
import time
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

# name -> (repo_id, folder inside the repo or "" for the whole repo); from excel/3D_Room_Scene_Collection.xlsx
SOURCES = {
    "SAGE-10k": ("liantian/3D_Room_Scene_Collections", "SAGE10k"),
    "M3DLayout": ("imChuling/3D_Room_Collections", "M3DLayout_exported"),
    "InteriorGS": ("imChuling/3D_Room_Collections", "InteriorGS_exported"),
    "MetaScenes": ("imChuling/3D_Room_Collections", "MetaScenes_exported"),
    "IL3D": ("imChuling/3D_Room_Collections", "IL3D_exported"),
    "SceneSmith": ("imChuling/3D_Room_Collections", "SceneSmith_exported"),
    "MansionWorld": ("imChuling/3D_Room_Collections", "MansionWorld_exported"),
    "OptiScene": ("imChuling/3D_Room_Collections", "OptiScene_exported"),
    "InternScenes": ("imChuling/3D_Room_Collections", "InternScenes_exported"),
    "Structured3D": ("imChuling/3D_Room_Collections", "Structured3D_exported"),
    "3RScan": ("imChuling/3D_Room_Collections", "3RScan_exported"),
    "ARKitScenes": ("imChuling/3D_Room_Collections", "ARKitScenes_exported"),
    "ScanNet": ("imChuling/3D_Room_Collections", "ScanNet_exported"),
    "SceneCAD_Scan2CAD": ("imChuling/3D_Room_Collections", "SceneCAD_Scan2CAD_exported"),
    "SpatialLM": ("imChuling/3D_Room_Collections", "SpatialLM_exported"),
    "3D-FRONT": ("BillLin66/3D_Room_Collections", "3DFront_exported"),
    "ScanNet++": ("BillLin66/3D_Room_Collections", "ScanNet++_exported"),
    "SpatialGen": ("BillLin66/3D_Room_Collections", "SpatialGen_exported"),
    "ASE": ("BillLin66/3D_Room_Collections", "ASE_exported"),
    "HM3DSem": ("BillLin66/3D_Room_Collections", "HM3DSem_exported"),
    "HSSD-200": ("XXXpilar/hssd_clean", ""),
    "MultiScan": ("XXXpilar/multiscan-clean", ""),
}

# meshes, textures, images and per-vertex segment archives: not needed for text layout training
HEAVY = ["*.glb", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.obj", "*.ply", "*.npz", "*.tar", "*.mtl", "*.urdf",
         "*segments_anno.zip", "*keyframes.zip"]


def select(files, folder, full, with_meshes):
    """Pick repo paths for one export folder."""
    pre = f"{folder}/" if folder else ""
    # whole-repo sources (HSSD, MultiScan) keep their layout json one level deeper: scenes/<id>/scene.json
    max_depth = 2 if folder == "" else 1
    out = []
    for p in files:
        if not p.startswith(pre):
            continue
        rel = p[len(pre):]
        if not with_meshes and any(fnmatch.fnmatch(p, pat) for pat in HEAVY):
            continue
        if full or rel.count("/") <= max_depth:
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="subset of SOURCES names")
    ap.add_argument("--full", action="store_true", help="also per-scene/raw files deeper in the folder")
    ap.add_argument("--with-meshes", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    by_repo = {}
    for name in a.only or SOURCES:
        repo, folder = SOURCES[name]
        by_repo.setdefault(repo, []).append((name, folder))
    api = HfApi()
    failed_count = 0
    for repo, items in by_repo.items():
        # one listing per repo: listing a large repo takes minutes
        sizes = {f.path: f.size for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
                 if isinstance(f, RepoFile)}
        local = f"{a.out}/{repo.replace('/', '__')}"
        for name, folder in items:
            wanted = select(sizes, folder, a.full, a.with_meshes)
            todo = [p for p in wanted if not (os.path.isfile(f"{local}/{p}") and os.path.getsize(f"{local}/{p}") == sizes[p])]
            print(f"== {name}: {repo}/{folder} ({len(wanted)} files, {len(todo)} to fetch, "
                  f"{sum(sizes[p] for p in todo) / 1e9:.2f} GB)", flush=True)
            with ThreadPoolExecutor(a.workers) as ex:
                failed = [p for p, ok in zip(todo, ex.map(lambda p: fetch(repo, p, local), todo)) if not ok]
            failed_count += len(failed)
            print(f"== {name}: done" if not failed else f"== {name}: {len(failed)} files failed, rerun to resume",
                  flush=True)
    if failed_count:
        raise SystemExit(1)


def fetch(repo, path, local, tries=8):
    """One file with backoff: the Hub answers bursts of small-file requests with 429/499/5xx."""
    for k in range(tries):
        try:
            hf_hub_download(repo, path, repo_type="dataset", local_dir=local)
            return True
        except Exception as e:  # noqa: BLE001  network/HTTP errors of several libraries
            if k == tries - 1:
                print(f"   failed {path}: {str(e)[:120]}", flush=True)
                return False
            time.sleep(min(30 * 2 ** k, 600))


if __name__ == "__main__":
    main()
