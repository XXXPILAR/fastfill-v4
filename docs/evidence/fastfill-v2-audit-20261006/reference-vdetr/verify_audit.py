"""Reconcile actual review scopes and prove pinned tracked source is unchanged."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
REPO = PROJECT / "V-DETR"
HEAD = "9062d75fe2c91e5d4a771b5325483fc330a3e827"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    before = json.loads((HERE / "source-inventory-before.json").read_text())
    native = json.loads((HERE / "native-geometry/native-coverage.json").read_text())
    native_sources = {row["path"]: row for row in native["text_files"] if row.get("source_full_read")}
    # These 22 paths were fully read by the child; all remaining 31 paths were
    # fully displayed and read by the main reviewer. Four utility reads overlap.
    child_core = {p for p in native_sources if p.startswith("third_party/pointnet2/")}
    child_core.update({"utils/box_ops3d.py", "utils/box_util.py"})
    head_actual = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    assert head_actual == HEAD
    rows = []
    for row in before["sources"]:
        relative = str(Path(row["path"]).relative_to("V-DETR"))
        source = REPO / relative
        content = source.read_bytes()
        git_blob = subprocess.check_output(["git", "-C", str(REPO), "show", f"{HEAD}:{relative}"])
        after_hash = sha(content)
        assert after_hash == row["sha256"] == sha(git_blob), relative
        readers = []
        if relative not in child_core:
            readers.append("reference_interface: displayed entire source, then static reading")
        if relative in native_sources:
            assert native_sources[relative]["sha256"] == after_hash
            readers.append("vdetr_native_geometry: actual full read recorded in native-coverage.json")
        assert readers, relative
        rows.append({"path": relative, "lines": row["lines"], "sha256": after_hash,
                     "full_text_read": True, "read_ranges": [[1, row["lines"]]] if row["lines"] else [],
                     "empty_file_confirmed": row["lines"] == 0, "readers": readers,
                     "matches_git_HEAD_blob": True, "matches_initial_inventory": True})
    noncode = []
    for relative in ["README.md", "requirements.txt", "LICENSE", ".gitignore", "results/scannet_result.txt"]:
        source = REPO / relative
        content = source.read_bytes()
        git_blob = subprocess.check_output(["git", "-C", str(REPO), "show", f"{HEAD}:{relative}"])
        assert content == git_blob
        noncode.append({"path": relative, "sha256": sha(content), "lines": len(content.decode().splitlines()),
                        "full_text_read": True, "matches_git_HEAD_blob": True})
    report = {
        "head": HEAD, "scope": "53 pinned source paths; full static reading asserted only by actual review scope, hash equality checked independently",
        "summary": {"files": len(rows), "lines": sum(r["lines"] for r in rows),
                    "python_files": sum(r["path"].endswith(".py") for r in rows),
                    "native_files": sum(not r["path"].endswith(".py") for r in rows),
                    "main_reviewer_full_files": sum(r["path"] not in child_core for r in rows),
                    "main_reviewer_full_lines": sum(r["lines"] for r in rows if r["path"] not in child_core),
                    "child_core_full_files": len(child_core), "child_extra_duplicate_reads": 4,
                    "hash_mismatches": 0, "uncovered_sources": 0},
        "sources": rows, "additional_texts": noncode,
        "paper_reading": {"arxiv_v1": {"pages": 15, "extracted_text_pages_read": list(range(1, 16)),
                                        "visual_contact_sheets_inspected": [1, 2, 3, 4, 5], "render_scale_long_edge_px": 1000},
                          "repo_details_v2": {"pages": 1, "visual_pages_inspected": [1], "render_scale_long_edge_px": 2000},
                          "metrics_reproduced": False, "ICLR_final_version_audited": False},
        "upstream_git_status": subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip(),
        "execution_limits": "CPU-only selected Python/mathematical counterexamples; no native install/compile/GPU/full detector/data/physical validation",
    }
    assert report["summary"]["files"] == 53
    assert report["summary"]["lines"] == 10488
    assert report["summary"]["main_reviewer_full_files"] == 31
    (HERE / "source-read-coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"head": HEAD, "summary": report["summary"], "git_status": report["upstream_git_status"]}, indent=2))


if __name__ == "__main__":
    main()
