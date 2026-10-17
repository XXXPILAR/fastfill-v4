"""Standalone, standard-library checksum verification for a portable bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_path(root, name):
    if (not isinstance(name, str) or not name or "\\" in name or "\0" in name
            or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
            or not PurePosixPath(name).parts):
        raise ValueError("invalid bundle inventory path")
    target = root / name
    if any(path.is_symlink() for path in (target, *target.parents) if path != root and root in path.parents):
        raise ValueError(f"bundle inventory symlink is forbidden: {name}")
    return target


def _generated(path):
    return (path.parts[0] in {"outputs", "upload", "models"}
            or any(part in {"__pycache__", ".pytest_cache"} for part in path.parts)
            or path.suffix == ".pyc")


def verify_bundle(bundle_root):
    """Verify immutable inventory; generated runs/caches remain outside it.

    This establishes transfer/file integrity, not geometry-label truth or model
    quality. Dataset/source audits are separate reports in the bundle.
    """
    root = Path(bundle_root).resolve()
    manifest = json.loads((root / "BUNDLE_MANIFEST.json").read_text())
    if manifest.get("schema_version") != "fastfill.bundle.v1" or not isinstance(manifest.get("files"), dict):
        raise ValueError("unsupported bundle manifest")
    files, total = manifest["files"], 0
    for name, entry in files.items():
        path = _entry_path(root, name)
        if not path.is_file():
            raise FileNotFoundError(f"missing bundle file: {name}")
        if (not isinstance(entry, dict) or not isinstance(entry.get("bytes"), int)
                or isinstance(entry["bytes"], bool) or entry["bytes"] < 0
                or not isinstance(entry.get("sha256"), str) or len(entry["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in entry["sha256"])):
            raise ValueError(f"invalid bundle inventory metadata: {name}")
        if path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
            raise ValueError(f"bundle checksum/size mismatch: {name}")
        total += entry["bytes"]
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if _generated(relative) or str(relative) in {"BUNDLE_MANIFEST.json", "SHA256SUMS"}:
            continue
        if path.is_symlink() or (path.is_file() and relative.as_posix() not in files):
            raise ValueError(f"unlisted bundle file or symlink: {relative}")
    return {"ok": True, "files_checked": len(files), "bytes_checked": total,
            "scope": "immutable bundle checksum integrity; generated outputs/upload/models/caches excluded"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    result = verify_bundle(parser.parse_args(argv).bundle_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
