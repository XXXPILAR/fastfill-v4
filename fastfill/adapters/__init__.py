"""Source adapters: exported dataset files -> IR rooms (see fastfill/scene.py).

Each module in this package defines
    SOURCE = "<name>"            # output file name <name>.jsonl
    def load(root) -> iterable   # yields IR room dicts; root = download dir of fastfill.download

    python -m fastfill.adapters --root /Volumes/harddisk/3D_Room_Collections --out /Volumes/harddisk/fastfill_ir [--only IL3D]
"""
import argparse
import importlib
import json
import os
import pkgutil


def registry():
    mods = {}
    for m in pkgutil.iter_modules(__path__):
        if m.name.startswith("_") or m.name == "unified":
            continue
        mod = importlib.import_module(f"{__name__}.{m.name}")
        if hasattr(mod, "SOURCE") and hasattr(mod, "load"):
            mods[mod.SOURCE] = mod
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, mod in sorted(registry().items()):
        if a.only and name not in a.only:
            continue
        n = 0
        tmp = f"{a.out}/{name}.jsonl.tmp"
        try:
            with open(tmp, "w") as f:
                for room in mod.load(a.root):
                    f.write(json.dumps(room, ensure_ascii=False) + "\n")
                    n += 1
        except FileNotFoundError as e:       # dataset not downloaded (or deleted): keep any existing IR file
            os.remove(tmp)
            print(f"{name}: skipped, input not found: {e.filename}", flush=True)
            continue
        if n == 0:                           # an empty input file must not wipe a good IR file
            os.remove(tmp)
            print(f"{name}: skipped, 0 rooms read; existing IR kept", flush=True)
            continue
        os.replace(tmp, f"{a.out}/{name}.jsonl")
        print(f"{name}: {n} rooms", flush=True)


if __name__ == "__main__":
    main()
