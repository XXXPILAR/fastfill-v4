"""Self-check of stratify's selection-rule attribution on new-data rows (CPU only).

    python selftest_shift.py --rows runs/analysis/ablation-sample-500/new.jsonl --work runs/analysis/test/shift

Writes the labels as predictions with +0.5 m along x on every target whose selection_rule is not frozen_prep, scores
them with ``evaluate --predictions`` (full condition) and checks via stratify: report reproduced exactly; position error
0 on frozen_prep and 0.5 on each added rule; size and yaw error 0 everywhere.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import stratify as S

SHIFT = .5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    with open(a.rows) as src, open(a.work / "predictions.jsonl", "w") as out:
        for line in src:
            row = json.loads(line)
            rules = [(e or {}).get("selection_rule") or "frozen_prep" for e in row["provenance"]["field_evidence"]]
            objects = []
            for t, rule in zip(row["target"]["objects"], rules):
                position = [v if v is not None else 0. for v in t["bottom_center_m"]]
                size = [v if v else 1. for v in t["target_size_local_m"]]
                objects.append({"id": t["id"], "bottom_center_m": [position[0] + (rule != "frozen_prep") * SHIFT, *position[1:]],
                                "target_size_local_m": size, "yaw_rad": t["yaw_rad"] if t["yaw_rad"] is not None else 0.})
            out.write(json.dumps({"layout": {"schema_version": "fastfill.v2", "objects": objects}}) + "\n")
    subprocess.run([sys.executable, "-m", "fastfill.v2.evaluate", "--data", str(a.rows), "--predictions",
                    str(a.work / "predictions.jsonl"), "--output", str(a.work / "eval")], check=True, stdout=subprocess.DEVNULL)
    meta, rooms, checks = S.collect(a.work / "eval", "outcomes.jsonl", a.rows, 6)
    strata = S.aggregate(rooms, boot=200)
    assert checks["reproduces_report_exactly"], checks
    seen = {}
    for rule, entry in strata["rule"].items():
        m = entry["metrics"]
        seen[rule] = (m["bottom_center_error_m"]["mean"], m["bottom_center_error_m"]["n"])
        assert abs(m["bottom_center_error_m"]["mean"] - (0. if rule == "frozen_prep" else SHIFT)) < 1e-9, (rule, m)
        assert m["yaw_error_rad"]["mean"] < 1e-9 and m.get("log_size_error", {"mean": 0.})["mean"] < 1e-9, (rule, m)
    assert set(seen) == set(S.RULES), seen
    print(json.dumps({"ok": True, "rooms": len(rooms), "layouts": checks["requests_with_layout"],
                      "position_error_mean_and_objects_by_rule": seen,
                      "places": {k: v["objects"] for k, v in strata["place"].items()}}, indent=1))


if __name__ == "__main__":
    main()
