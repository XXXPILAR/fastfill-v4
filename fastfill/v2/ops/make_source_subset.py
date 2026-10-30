"""A dataset restricted to chosen sources: every row of <data>/{train,validation,test}.jsonl whose provenance.source is
listed, copied byte for byte in file order (so the subset of a sha256-pinned build is itself reproducible). Writes the
three files and manifest.json (inputs' and outputs' sha256, sources, rooms / objects / detailed descriptions per split and
source) into a temporary directory renamed to --output at the end; an existing --output is refused.

The default sources are the three whose object descriptions are detailed (SAGE-10k style): an object counts as detailed
when its description is not just the category and has at least 12 words naming at least two of colour, material,
shape/structure and style (88-95 % of these sources' objects; 16 % of MansionWorld's, 8.5 % of OptiScene's, 0 elsewhere).

    python fastfill/v2/ops/make_source_subset.py --data data/v4fixed --output data/v4detail
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

SPLITS = ("train", "validation", "test")
DETAILED_SOURCES = ("SAGE-10k", "IL3D_3dfront", "IL3D_synthetic")
ATTRIBUTES = tuple(set(words.split()) for words in (
    "white black gray grey brown beige red blue green yellow orange pink purple violet silver gold golden cream ivory tan "
    "navy teal turquoise maroon burgundy charcoal olive khaki dark light pale bright multicolored multicolor colorful",
    "wood wooden oak walnut pine teak birch maple mahogany cherry metal metallic steel iron brass chrome copper aluminum "
    "aluminium glass ceramic porcelain marble stone granite fabric leather velvet linen cotton wool plastic wicker rattan "
    "bamboo concrete upholstered upholstery woven mesh rubber paper cardboard terracotta acrylic lacquered",
    "rectangular rectangle square round rounded circular cylindrical cylinder oval curved flat tall low slim slender thin "
    "thick narrow wide compact tapered angular legs leg drawer drawers shelf shelves door doors frame top base cushion "
    "cushions armrest armrests arms backrest handle handles panel panels knob knobs pedestal tripod spherical l-shaped",
    "modern minimalist classic vintage rustic contemporary sleek industrial scandinavian traditional elegant ornate "
    "antique retro mid-century simple"))
WORD = re.compile(r"[a-z][a-z\-]*")


def detailed(obj: dict) -> bool:
    """At least 12 words and two attribute groups, and not merely the category."""
    norm = lambda s: re.sub(r"[_\s]+", " ", (s or "").strip().lower())  # noqa: E731
    text = norm(obj.get("description"))
    if not text or text == norm(obj.get("category")):
        return False
    words = WORD.findall(text)
    return len(words) >= 12 and sum(bool(group.intersection(words)) for group in ATTRIBUTES) >= 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True, help="directory with train/validation/test.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sources", nargs="+", default=list(DETAILED_SOURCES))
    a = p.parse_args(argv)
    if a.output.exists():
        raise SystemExit(f"{a.output} exists; remove it to rebuild")
    tmp = a.output.with_name(a.output.name + ".tmp")
    tmp.mkdir(parents=True, exist_ok=False)
    keep, seen = set(a.sources), set()
    manifest = {"selection": {"provenance.source": sorted(keep)}, "inputs": {}, "outputs": {}, "splits": {}}
    for split in SPLITS:
        source_file, target = a.data / f"{split}.jsonl", tmp / f"{split}.jsonl"
        counts = defaultdict(lambda: {"rooms": 0, "objects": 0, "objects_detailed": 0})
        with open(source_file, "rb") as stream, open(target, "wb") as out:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                source = row["provenance"]["source"]
                seen.add(source)
                if source not in keep:
                    continue
                out.write(line if line.endswith(b"\n") else line + b"\n")
                objects = row["condition"]["objects"]
                c = counts[source]
                c["rooms"] += 1
                c["objects"] += len(objects)
                c["objects_detailed"] += sum(map(detailed, objects))
        manifest["inputs"][f"{split}.jsonl"] = sha256(source_file)
        manifest["outputs"][f"{split}.jsonl"] = sha256(target)
        manifest["splits"][split] = {source: counts[source] for source in sorted(counts)}
        total = {k: sum(c[k] for c in counts.values()) for k in ("rooms", "objects", "objects_detailed")}
        print(f"{split}: {total['rooms']} rooms, {total['objects']} objects, "
              f"{100 * total['objects_detailed'] / max(total['objects'], 1):.1f} % detailed")
    if keep - seen:
        raise SystemExit(f"sources absent from {a.data}: {sorted(keep - seen)} (left in {tmp})")
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.rename(a.output)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
