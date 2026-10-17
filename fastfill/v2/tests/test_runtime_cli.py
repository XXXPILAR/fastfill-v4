"""The offline runtime CLI executes complete JSON transactions with local fixtures."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastfill.v2.serve import asset_from_dict, main
from fastfill.v2.tests.test_runtime import condition, layout


class RuntimeCLITests(unittest.TestCase):
    def test_complete_request_commits_only_in_memory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            request_path, prediction_path, catalog_path = (root / name for name in ("condition.json", "prediction.json", "catalog.json"))
            request_path.write_text(json.dumps(condition()))
            prediction_path.write_text(json.dumps(layout()))
            catalog_path.write_text(json.dumps({"assets": [{"ref": "desk", "category": "desk", "actual_size_local_m": [1, 1, .75]}]}))
            output = StringIO()
            with redirect_stdout(output):
                status = main(["--condition", str(request_path), "--prediction", str(prediction_path),
                               "--catalog", str(catalog_path), "--commit-in-memory"])
            report = json.loads(output.getvalue())
            self.assertEqual(status, 0)
            self.assertTrue(report["committed"])
            self.assertEqual(report["offline_host_snapshot"]["commits"], 1)
            self.assertEqual(report["raw_model_output"], layout())

    def test_source_data_output_is_refused_before_input_reads(self):
        stderr = StringIO()
        with redirect_stderr(stderr):
            status = main(["--condition", "missing", "--prediction", "missing", "--catalog", "missing",
                           "--output", "/Volumes/harddisk/3D_Room_Collections/runtime-test.json"])
        self.assertEqual(status, 2)
        self.assertIn("read-only", stderr.getvalue())

    def test_existing_output_is_preserved_before_input_reads(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("preserved")
            with redirect_stderr(StringIO()):
                status = main(["--condition", "missing", "--prediction", "missing", "--catalog", "missing", "--output", str(output)])
            self.assertEqual(status, 2)
            self.assertEqual(output.read_text(), "preserved")

    def test_failed_scene_is_written_as_failure_with_no_commit(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            request_path, prediction_path, catalog_path, report_path = (root / name for name in ("condition.json", "prediction.json", "catalog.json", "report.json"))
            request_path.write_text(json.dumps(condition()))
            prediction_path.write_text(json.dumps(layout([{**layout()["objects"][0], "bottom_center_m": [.1, 2, 0]}])))
            catalog_path.write_text(json.dumps([{"ref": "desk", "category": "desk", "actual_size_local_m": [1, 1, .75]}]))
            status = main(["--condition", str(request_path), "--prediction", str(prediction_path), "--catalog", str(catalog_path),
                           "--output", str(report_path), "--commit-in-memory", "--max-asset-retries", "0"])
            report = json.loads(report_path.read_text())
            self.assertEqual(status, 2)
            self.assertFalse(report["committed"])
            self.assertEqual(report["offline_host_snapshot"]["commits"], 0)

    def test_malformed_asset_metadata_is_rejected(self):
        for asset in ({"ref": "desk", "category": "desk", "actual_size_local_m": [1, 1, 0]},
                      {"ref": "desk", "category": "desk", "actual_size_local_m": [1, 1, 1], "capabilities": "work_surface"},
                      {"ref": "desk", "category": "desk", "actual_size_local_m": [1, 1, 1], "semantic_front_local": [0, 1, 0]},
                      {"ref": "desk", "category": "desk", "actual_size_local_m": [1, 1, 1], "surprise": True}):
            with self.subTest(asset=asset), self.assertRaises(ValueError):
                asset_from_dict(asset)


if __name__ == "__main__":
    unittest.main()
