"""python -m fastfill.tools.ablation_check <base data dir> <constrained data dir>
The constrained build must differ from the base build only by the `constraints` key of some train rows:
same complete rows (including source, flags and all message metadata) after removing only that key from
the constrained user JSON; dev/test files byte-identical (one evaluation serves both)."""
import hashlib
from itertools import zip_longest
import json
import sys


def _normalized(row, remove_constraints=False):
    """Keep the entire row; normalize only the intended user JSON payload."""
    user = json.loads(row["messages"][1]["content"])
    if remove_constraints:
        user = {k: v for k, v in user.items() if k != "constraints"}
    normalized = {**row, "messages": [
        {**m, "content": user} if i == 1 else m for i, m in enumerate(row["messages"])]}
    # JSON booleans and numbers must not compare equal (Python's True == 1 does).
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def main(base, cons):
    diff = n_rows = n_cons = 0
    with open(f"{base}/train.jsonl", encoding="utf-8") as a, open(f"{cons}/train.jsonl", encoding="utf-8") as b:
        for lx, ly in zip_longest(a, b):
            assert lx is not None and ly is not None, "train row counts differ"
            x, y = json.loads(lx), json.loads(ly)
            n_rows += 1
            n_cons += "constraints" in json.loads(y["messages"][1]["content"])
            diff += _normalized(x) != _normalized(y, remove_constraints=True)
    same_eval = all(_sha256(f"{base}/{f}") == _sha256(f"{cons}/{f}")
                    for f in ("dev.jsonl", "test.jsonl", "dev_rooms.jsonl", "test_rooms.jsonl",
                              "dev_constrained_rooms.jsonl", "test_constrained_rooms.jsonl"))
    print(json.dumps({"train_rows": n_rows, "rows_differing_beyond_constraints": diff,
                      "constrained_rows_in_" + cons.rstrip("/").split("/")[-1]: n_cons, "eval_files_identical": same_eval}))
    assert diff == 0 and same_eval, "not a clean ablation"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
