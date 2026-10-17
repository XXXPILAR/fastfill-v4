"""Tiny, offline regressions for DATA-QA-1/2/3 in the 2026-09-27 audit.

Run: python -m unittest discover -s fastfill/tests -p test_qa_repairs.py -v
No persisted dataset, trainer dependencies, or model downloads are needed.
"""
import argparse
import ast
import collections
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from fastfill import build
from fastfill.scene import canonical, messages, rot90
from fastfill.tools import ablation_check, qa_report


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILES = ("dev.jsonl", "test.jsonl", "dev_rooms.jsonl", "test_rooms.jsonl",
              "dev_constrained_rooms.jsonl", "test_constrained_rooms.jsonl")
ARGS = dict(reject_flagged=[], keep_duplicates=False, min_objects=1, max_vertices=0,
            sources=["S"], source_anchors={}, boundary_types=["polygon", "hull"],
            anchors=["floor", "object"], oob_tol=.1, hidden_max=.3,
            no_rot90=True, constraint_frac=0., field_dropout=0., desc_dropout=0.)


def write_rows(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def file_info(path):
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "lines": len(raw.splitlines())}


def source_room(uid="S:train"):
    def obj(id_, cat, pos, size, parent=None):
        return dict(id=id_, category=cat, size=size, pos=pos, yaw=0, parent=parent,
                    anchor="object" if parent else "floor", desc="A wooden furnishing beside a wall.")
    return dict(uid=uid, source="S", group="world_" + uid, room_type="livingroom",
                boundary=[[0, 0], [8, 0], [8, 6], [0, 6]], boundary_type="polygon", height=3,
                objects=[obj("a", "chair", [1, 1, 0], [.5, .5, 1]),
                         obj("b", "chair", [3, 1, 0], [.5, .5, 1]),
                         obj("m", "stool", [2, 1, 0], [.4, .4, .5]),
                         obj("t", "table", [5, 4, 0], [1, 1, 1]),
                         obj("l", "lamp", [5, 4, 1], [.2, .2, .5], "t"),
                         obj("w", "cabinet", [.25, 3, 0], [.5, 1, 1])],
                meta={"front_known": True})


def saved_row(raw, args, split="train"):
    """Fixture writer follows the documented build order, independently of QA."""
    room, why = build.prep(copy.deepcopy(raw), argparse.Namespace(**args))
    assert why is None, why
    cons, with_desc = None, True
    if split == "train":
        aug, s2, drop = (build.rng_for(raw["uid"], k) for k in ("aug", "s2", "dropout"))
        room = canonical(room if args["no_rot90"] else rot90(room, aug.randrange(4)))
        cons = build.extract_constraints(room, s2) if s2.random() < args["constraint_frac"] else None
        if drop.random() < args["field_dropout"]:
            room = {**room, "room_type": None}
        if drop.random() < args["field_dropout"]:
            room = {**room, "height": None}
        with_desc = drop.random() >= args["desc_dropout"]
    else:
        room = canonical(room)
    return {"uid": raw["uid"], "source": raw["source"], "flags": room["meta"]["flags"],
            "messages": messages(room, cons, with_desc=with_desc)}, room


def trainer_stub():
    # Audit technique: exercise the real selector on the original QA without importing torch/peft.
    tree = ast.parse((ROOT / "train.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "select")
    mod = types.ModuleType("fastfill.train")
    mod.collections, mod.hashlib = collections, hashlib
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "train.select", "exec"), mod.__dict__)
    return mod


class QARepairs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="fastfill-qa-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ir, self.data = self.root / "ir", self.root / "v3.2"
        self.ir.mkdir()
        self.data.mkdir()

    def fixture(self, **overrides):
        args = {**ARGS, **overrides}
        room = source_room()
        row, _ = saved_row(room, args)
        write_rows(self.ir / "S.jsonl", [room])
        write_rows(self.data / "train.jsonl", [row])
        for name in EVAL_FILES:
            write_rows(self.data / name, [])
        write_rows(self.data / "stats.json", [{"S": {"scanned": 1, "train_S1": 1}}])
        self.man = {"args": args, "files": {}, "ir_sha256": {"S.jsonl": file_info(self.ir / "S.jsonl")["sha256"]},
                    "code_sha256": {name: file_info(ROOT / name)["sha256"] for name in
                                    ("__init__.py", "build.py", "anchors.py", "scene.py", "split.py", "validate.py", "train.py")}}
        self.seal()
        return row

    def seal(self):
        self.man["files"] = {name: file_info(self.data / name) for name in ("train.jsonl", *EVAL_FILES, "stats.json")}
        write_rows(self.data / "MANIFEST.json", [self.man])

    def run_qa(self, options=False):
        args = ["qa_report", "--ir", str(self.ir), "--data", str(self.data)] if options else [
            "qa_report", str(self.ir), str(self.data)]
        with patch.object(sys, "argv", args), patch.dict(sys.modules, {"fastfill.train": trainer_stub()}), \
                contextlib.redirect_stdout(io.StringIO()):
            self.qa_status = qa_report.main()
        return json.loads((self.data / "QA.json").read_text())

    def ablation(self, original, changed, extra=False):
        base, cons = self.root / "base", self.root / "cons"
        base.mkdir(exist_ok=True)
        cons.mkdir(exist_ok=True)
        write_rows(base / "train.jsonl", [original])
        write_rows(cons / "train.jsonl", [changed] * (2 if extra else 1))
        for name in EVAL_FILES:
            write_rows(base / name, [])
            write_rows(cons / name, [])
        with contextlib.redirect_stdout(io.StringIO()):
            ablation_check.main(str(base), str(cons))

    def test_ablation_compares_every_metadata_field_and_message(self):
        row, _ = saved_row(source_room(), ARGS)
        variants = [{**row, "source": "other"}, {**row, "flags": {"oob_objects": 1}},
                    {**row, "weight": .5}, {**row, "messages": row["messages"] + [{"role": "user", "content": "extra"}]}]
        for field, value in (("role", "assistant"), ("name", "extra")):
            variants.append({**row, "messages": [row["messages"][0], {**row["messages"][1], field: value}, row["messages"][2]]})
        for i, changed in enumerate(variants):
            with self.subTest(variant=i), self.assertRaises(AssertionError):
                self.ablation(row, changed)

    def test_ablation_does_not_equate_boolean_and_numeric_metadata(self):
        row, _ = saved_row(source_room(), ARGS)
        with self.assertRaises(AssertionError):
            self.ablation({**row, "weight": True}, {**row, "weight": 1})

    def test_ablation_requires_byte_identical_nonempty_eval_files(self):
        row, _ = saved_row(source_room(), ARGS)
        self.ablation(row, row)
        write_rows(self.root / "base" / "dev.jsonl", [row])
        write_rows(self.root / "cons" / "dev.jsonl", [{**row, "source": "other"}])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(AssertionError):
            ablation_check.main(str(self.root / "base"), str(self.root / "cons"))

    def test_ablation_allows_only_user_constraints_and_json_whitespace(self):
        row, _ = saved_row(source_room(), ARGS)
        changed = copy.deepcopy(row)
        user = json.loads(changed["messages"][1]["content"])
        changed["messages"][1]["content"] = json.dumps({**user, "constraints": [["near", "chair_1", "chair_2", 2]]}, indent=2)
        self.ablation(row, changed)
        with self.assertRaises(AssertionError):
            self.ablation(row, changed, extra=True)
        with self.assertRaises(AssertionError):
            self.ablation(changed, changed)  # constraints in the base are not the permitted ablation

    def test_train_constraints_check_schema_references_and_saved_answer_truth(self):
        row = self.fixture()
        user = json.loads(row["messages"][1]["content"])
        cons = [["on", "lamp_1", "table_1"], ["faces", "chair_1", "chair_2"],
                ["near", "chair_1", "chair_2", 2], ["against_wall", "cabinet_1"],
                ["between", "stool_1", "chair_1", "chair_2"],
                ["near", "chair_1", "does_not_exist", .1], ["near", "chair_1", "chair_2", .1],
                ["on", "chair_1", "table_1"], ["bogus", "chair_1"], [], ["faces", "chair_1"],
                ["near", "chair_1", "chair_2", True], ["near", "chair_1", "chair_2", -1],
                ["near", "chair_1", "chair_2", float("nan")]]
        row["messages"][1]["content"] = json.dumps({**user, "constraints": cons})
        write_rows(self.data / "train.jsonl", [row])
        q = self.run_qa()
        self.assertEqual(q["constraint_reference_validation"]["train"], "5/14")
        self.assertEqual(q["constraint_failures"]["train"], {"schema": 6, "reference": 1, "false": 2})
        self.assertEqual(q["object_set_closure"], {"ok": 1})

    def test_constraint_container_and_unparseable_answer_are_reported(self):
        row = self.fixture()
        user = json.loads(row["messages"][1]["content"])
        for constraints, answer, reason in (({}, row["messages"][2]["content"], "schema"),
                                             ([["near", "chair_1", "chair_2", 2]], "broken", "answer")):
            with self.subTest(reason=reason):
                changed = copy.deepcopy(row)
                changed["messages"][1]["content"] = json.dumps({**user, "constraints": constraints})
                changed["messages"][2]["content"] = answer
                write_rows(self.data / "train.jsonl", [changed])
                q = self.run_qa()
                self.assertEqual(q["constraint_failures"]["train"], {reason: 1})
                self.assertEqual(q["constraint_reference_validation"]["train"], "0/1")

    def test_output_hashes_and_line_counts_include_stats_and_missing_files(self):
        self.fixture()
        (self.data / "dev_constrained_rooms.jsonl").unlink()
        (self.data / "stats.json").write_text('{"S":{"scanned":9}}\n\n')
        q = self.run_qa()
        differences = q["output files differing from MANIFEST (must be empty)"]
        self.assertEqual(set(differences), {"dev_constrained_rooms.jsonl", "stats.json"})
        self.assertIn("missing", differences["dev_constrained_rooms.jsonl"])
        self.assertEqual(set(differences["stats.json"]), {"sha256", "lines"})

    def test_target_corruption_never_reports_reproducible_even_with_updated_hash(self):
        row = self.fixture()
        answer = json.loads(row["messages"][2]["content"])
        answer["placements"][0]["pos"] = [6, 5, 0]
        row["messages"][2]["content"] = json.dumps(answer)
        write_rows(self.data / "train.jsonl", [row])
        q = self.run_qa()
        self.assertIn("train.jsonl", q["output files differing from MANIFEST (must be empty)"])
        self.assertEqual(q["leakage (all must be 0)"]["rows not reproducible from IR"], 1)
        self.seal()
        q = self.run_qa()
        self.assertEqual(q["output files differing from MANIFEST (must be empty)"], {})
        self.assertEqual(q["SFT reproducibility"]["mismatches"], {"assistant": 1})

    def test_replay_detects_user_system_role_and_metadata_changes(self):
        row = self.fixture()
        for component in ("user", "system", "role", "metadata"):
            with self.subTest(component=component):
                changed = copy.deepcopy(row)
                if component == "user":
                    user = json.loads(changed["messages"][1]["content"])
                    changed["messages"][1]["content"] = json.dumps({**user, "room_type": "wrong"})
                elif component == "system":
                    changed["messages"][0]["content"] = "wrong system"
                elif component == "role":
                    changed["messages"][1]["role"] = "assistant"
                else:
                    changed["flags"] = {"oob_objects": 1}
                write_rows(self.data / "train.jsonl", [changed])
                q = self.run_qa()
                self.assertEqual(q["leakage (all must be 0)"]["rows not reproducible from IR"], 1)
                self.assertEqual(q["SFT reproducibility"]["mismatches"], {"user" if component == "role" else component: 1})

    def test_manifest_settings_and_named_cli_reproduce_all_splits(self):
        self.fixture(no_rot90=False, constraint_frac=1., field_dropout=.5, desc_dropout=1.)
        raw = [source_room()]
        for split in ("dev", "test"):
            room = source_room("S:" + split)
            raw.append(room)
            row, prepared = saved_row(room, self.man["args"], split)
            write_rows(self.data / (split + ".jsonl"), [row])
            write_rows(self.data / (split + "_rooms.jsonl"), [prepared])
        write_rows(self.ir / "S.jsonl", raw)
        self.man["ir_sha256"]["S.jsonl"] = file_info(self.ir / "S.jsonl")["sha256"]
        self.seal()
        q = self.run_qa(options=True)
        self.assertEqual(q["rows"], {"train": 1, "dev": 1, "test": 1})
        self.assertEqual(q["SFT reproducibility"]["matched"], 3)
        self.assertEqual(q["leakage (all must be 0)"]["rows not reproducible from IR"], 0)
        self.assertEqual(q["build"], str(self.data))

    def test_trainer_drift_is_disclosed_without_suppressing_build_drift(self):
        self.fixture()
        self.man["code_sha256"].update({"train.py": "old trainer", "scene.py": "old scene"})
        excluded = ["review_old/audit.py", "tests/test_old.py", ".hidden/tool.py"]
        self.man["code_sha256"].update(dict.fromkeys(excluded, "old excluded artifact"))
        self.seal()
        q = self.run_qa()
        self.assertEqual(q["build code differing from MANIFEST (must be empty)"], ["scene.py"])
        self.assertIn("train.py", q["other code changed since the build (does not affect the data)"])
        self.assertEqual(q["legacy MANIFEST code entries outside build provenance scope"], sorted(excluded))
        self.assertTrue(set(excluded).isdisjoint(q["other code changed since the build (does not affect the data)"]))
        self.assertEqual(q["SFT reproducibility"]["matched"], 1)

    def test_missing_ir_and_prep_rejection_cannot_count_as_reproduced(self):
        self.fixture()
        for raw, reason in (([], "missing_ir"), ([{**source_room(), "meta": {"front_known": False}}], "prep")):
            with self.subTest(reason=reason):
                write_rows(self.ir / "S.jsonl", raw)
                q = self.run_qa()
                self.assertEqual(q["leakage (all must be 0)"]["rows not reproducible from IR"], 1)
                self.assertEqual(q["SFT reproducibility"]["mismatches"], {reason: 1})

    def test_empty_train_reports_empty_percentiles(self):
        self.fixture()
        write_rows(self.data / "train.jsonl", [])
        self.seal()
        q = self.run_qa()
        self.assertEqual(q["train objects per row"], dict.fromkeys(("p50", "p90", "p99", "max")))
        self.assertEqual(q["SFT reproducibility"]["matched"], 0)

    def test_missing_stats_still_writes_provenance_discrepancy(self):
        self.fixture()
        (self.data / "stats.json").unlink()
        q = self.run_qa()
        self.assertEqual(q["output files differing from MANIFEST (must be empty)"]["stats.json"], {"missing": True})
        self.assertIsNone(q["rejections_by_source"])

    def test_fixed_references_extra_messages_and_false_geometry_remain_failures(self):
        row = self.fixture()
        user = json.loads(row["messages"][1]["content"])
        user["fixed"] = [{"id": "column_1", "size": [.3, .3, 3], "pos": [7, 5, 0], "yaw": 0}]
        user["constraints"] = [["near", "chair_1", "column_1", 100]]
        row["messages"][1]["content"] = json.dumps(user)
        target = json.loads(row["messages"][2]["content"])
        target["placements"][0]["pos"] = [20, 20, 0]
        row["messages"][2]["content"] = json.dumps(target)
        row["messages"].append({"role": "assistant", "content": "extra"})
        write_rows(self.data / "train.jsonl", [row])
        q = self.run_qa()
        self.assertEqual(q["constraint_failures"]["train"], {"reference": 1})
        self.assertEqual(q["train rows used by default"]["reference_valid"], "0/1")
        self.assertEqual(q["train reference inside walls within 1 mm"]["raw"], "0/1")
        self.assertEqual(q["SFT reproducibility"]["mismatches"]["extra_messages"], 1)
        row["flags"] = {"overlapping_furniture": 1, "fixed_collision": 1, "oob_objects": 0}
        write_rows(self.data / "train.jsonl", [row])
        self.assertEqual(self.run_qa()["train rows used by default"]["left_out"], {"flag:fixed_collision": 1})

    def test_eval_constraint_validation_and_summary_match_existing_metrics(self):
        from fastfill.evaluate import score, summarize
        self.fixture()
        _, room = saved_row(source_room("S:dev"), ARGS, "dev")
        write_rows(self.data / "dev_rooms.jsonl", [room])
        write_rows(self.data / "dev_constrained_rooms.jsonl", [
            {**room, "constraints": [["faces", "chair_1", "chair_2"], ["near", "chair_1", "missing", 1], ["unknown"]]}])
        q = self.run_qa()
        self.assertEqual(q["constraint_reference_validation"]["dev"], "1/3")
        self.assertEqual(q["constraint_failures"]["dev"], {"reference": 1, "schema": 1})
        expected = summarize([score(room, None)])["in_dist"]
        actual = q["reference legality raw vs repaired"]["dev/in_dist"]
        self.assertEqual(actual, {k: expected[k] for k in actual})

    def test_actual_builder_roundtrip_with_manifest_settings(self):
        write_rows(self.ir / "S.jsonl", [source_room()])
        args = ["build", "--ir", str(self.ir), "--out", str(self.data), "--sources", "S",
                "--dev", "0", "--test", "0", "--constraint_frac", "1", "--field_dropout", "1", "--desc_dropout", "1"]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            build.main()
        q = self.run_qa(options=True)
        self.assertEqual(q["SFT reproducibility"]["matched"], 1)
        self.assertEqual(q["SFT reproducibility"]["failed"], 0)
        self.assertEqual(q["MANIFEST defaults used"], {})
        self.assertEqual(q["output files differing from MANIFEST (must be empty)"], {})
        self.assertEqual(q["build code differing from MANIFEST (must be empty)"], [])

    def test_data_files_are_streamed_and_hashing_handles_unterminated_last_line(self):
        import builtins
        row = self.fixture()
        (self.data / "train.jsonl").write_text(json.dumps(row))  # No trailing newline.
        self.seal()
        original_open = builtins.open

        class Bounded:
            def __init__(self, file):
                self.file = file

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.file.close()

            def __iter__(self):
                return iter(self.file)

            def read(self, size=-1):
                if not 0 < size <= 1024 * 1024:
                    raise AssertionError("unbounded dataset read")
                return self.file.read(size)

            def readlines(self, *args):
                raise AssertionError("materialized dataset")

        def bounded_open(path, mode="r", *args, **kwargs):
            file = original_open(path, mode, *args, **kwargs)
            return Bounded(file) if str(path).endswith(".jsonl") and "r" in mode else file

        with patch("builtins.open", bounded_open):
            q = self.run_qa()
            self.ablation(row, row)
        self.assertEqual(q["output files differing from MANIFEST (must be empty)"], {})
        self.assertEqual(q["SFT reproducibility"]["matched"], 1)

    def rotated_fixed_row(self, fixed_xy, chair_xy, chair_size):
        room = {"boundary": [[0, 0], [4, 0], [4, 4], [0, 4]], "boundary_type": "polygon", "height": 3,
                "fixed": [{"id": "column_1", "category": "column", "size": [2, .1, 2],
                           "pos": [*fixed_xy, 0], "yaw": math.pi / 2}],
                "objects": [{"id": "chair_1", "category": "chair", "size": chair_size,
                             "pos": [*chair_xy, 0], "yaw": 0, "parent": None}]}
        return {"uid": "S:rotated", "source": "S", "flags": {}, "messages": messages(room)}

    def test_rotated_fixed_yaw_is_radians_for_collision_geometry(self):
        row = self.rotated_fixed_row([2, 2], [2, 2.8], [.2, .2, 1])
        original = copy.deepcopy(row)
        user, room, _, layout, error = qa_report._layout(row)
        self.assertIsNone(error)
        # A vertical 2m column crosses this chair; 90 radians would miss it.
        self.assertEqual(qa_report.check(layout)["fixed_blocking"], [("chair_1", "column_1")])
        self.assertAlmostEqual(room["fixed"][0]["yaw"], math.pi / 2)
        self.assertEqual(user["fixed"][0]["yaw"], 90)
        self.assertEqual(row, original)

    def test_rotated_fixed_blocks_repair_and_changes_reported_containment(self):
        row = self.rotated_fixed_row([.5, 2.8], [.2, 2], [.5, .5, 1])
        # The chair starts 5cm outside the left wall, touching the column's x=.45 edge.
        # Every allowed move inside would increase overlap with the vertical column.
        q = qa_report._train_metrics(iter([row]))
        self.assertEqual(q["train reference inside walls within 1 mm"], {"raw": "0/1", "repaired": "0/1"})

    def test_mandatory_failures_exclude_trainer_drift_and_geometry_statistics(self):
        self.fixture()
        self.man["code_sha256"]["train.py"] = "old trainer"
        self.seal()
        q = self.run_qa()
        self.assertEqual(self.qa_status, 0)
        self.assertEqual(q["mandatory_failures"], [])
        informational = {**q, "train reference inside walls within 1 mm": {"raw": "0/1", "repaired": "0/1"},
                         "train rows used by default": {"rows": 1, "reference_valid": "0/1"}}
        self.assertEqual(qa_report._mandatory_failures(informational), [])
        mandatory = {
            "build code differing from MANIFEST (must be empty)": ["scene.py"],
            "IR files differing from MANIFEST (must be empty)": ["S.jsonl"],
            "output files differing from MANIFEST (must be empty)": {"train.jsonl": {"missing": True}},
            "output files missing from MANIFEST (must be empty)": ["stats.json"],
            "object_set_closure": {"ok": 1, "missing_ids": 1},
            "fixed_boxes_answered (must be 0)": 1,
            "constraint_failures": {"train": {"false": 1}, "dev": {}, "test": {}},
            "leakage (all must be 0)": {"train&dev worlds": 1},
            "SFT reproducibility": {"matched": 0, "failed": 1},
        }
        for field, value in mandatory.items():
            with self.subTest(field=field):
                self.assertIn(field, qa_report._mandatory_failures({**q, field: value}))

    def test_cli_writes_report_before_failing_for_target_corruption(self):
        row = self.fixture()
        command = [sys.executable, "-B", "-m", "fastfill.tools.qa_report", "--ir", str(self.ir), "--data", str(self.data)]
        clean = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True)
        self.assertEqual(clean.returncode, 0, clean.stderr)
        row["messages"][2]["content"] = row["messages"][2]["content"].replace('"pos":[5,4,0]', '"pos":[6,5,0]')
        write_rows(self.data / "train.jsonl", [row])
        changed = subprocess.run(command, cwd=ROOT.parent, capture_output=True, text=True)
        self.assertEqual(changed.returncode, 1, changed.stderr)
        q = json.loads((self.data / "QA.json").read_text())
        self.assertIn("output files differing from MANIFEST (must be empty)", q["mandatory_failures"])
        self.assertIn("SFT reproducibility", q["mandatory_failures"])
        self.assertEqual(json.loads(changed.stdout)["mandatory_failures"], q["mandatory_failures"])


if __name__ == "__main__":
    unittest.main()
