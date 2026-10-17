"""Bounded runtime audit regressions; synthetic layouts and handler sinks, never sockets."""
import ast
import copy
import io
import json
import math
from pathlib import Path
import runpy
import unittest
from unittest.mock import Mock
import warnings

from shapely.geometry import Polygon

from fastfill import serve, validate
from fastfill.interface import placements_from_text, request_to_room
from fastfill.scene import messages


def request():
    return {
        "room": {"boundary_xy": [[0, 0], [4, 0], [4, 4], [0, 4]]},
        "objects_to_place": [
            {"id": "a", "category": "box", "size_xyz_m": [.5, .5, .3]},
            {"id": "b", "category": "table", "size_xyz_m": [1, 1, .7]},
        ],
    }


def model_text(supported=True):
    box = {"id": "box_1", "pos": [2, 2, .7] if supported else [1, 1, 0], "yaw": 0}
    return json.dumps({"placements": [
        {"id": "table_1", "pos": [2, 2, 0], "yaw": 0},
        {**box, **({"on": "table_1"} if supported else {})},
    ]})


def one_box_text(x, y=2):
    return json.dumps({"placements": [{"id": "box_1", "pos": [x, y, 0], "yaw": 0}]})


class RequestRepairs(unittest.TestCase):
    def test_anchors_are_enforced_and_reported_without_new_model_protocol(self):
        for anchor in ("floor", "object"):
            for supported in (False, True):
                with self.subTest(anchor=anchor, supported=supported):
                    req = request()
                    req["objects_to_place"][0]["anchor"] = anchor
                    original = copy.deepcopy(req)
                    generate = Mock(return_value=model_text(supported))
                    status, out = serve.answer(req, generate)
                    expected = supported == (anchor == "object")
                    self.assertEqual(status, 200 if expected else 422)
                    validation = (out if status == 200 else out["error"])["validation"]
                    for phase in ("raw", "repaired"):
                        report = next(c for c in validation[phase]["constraints"]
                                      if c["constraint"] == ["anchor", "a", anchor])
                        self.assertEqual(report["holds"], expected)
                        self.assertTrue(report["hard"])
                        self.assertFalse(report["model_conditioned"])
                        self.assertFalse(report["trained"])
                    prompt = json.loads(generate.call_args.args[0][1]["content"])
                    self.assertNotIn("constraints", prompt)
                    self.assertTrue(all("anchor" not in o for o in prompt["objects"]))
                    self.assertEqual(req, original)

    def test_object_anchor_still_requires_geometric_support(self):
        req = request()
        req["objects_to_place"][0]["anchor"] = "object"
        status, out = serve.answer(req, lambda _: model_text().replace('[2, 2, 0.7]', '[1, 1, 0.7]'))
        self.assertEqual(status, 422)
        self.assertEqual(out["error"]["validation"]["repaired"]["support_fail"], ["a"])

    def test_soft_keepout_preserves_floor_and_reports_satisfaction(self):
        polygon = [[0, 0], [1, 0], [1, 4], [0, 4]]
        for x, expected in ((.5, False), (1.25, True), (2, True)):
            with self.subTest(x=x):
                req = {**request(), "objects_to_place": request()["objects_to_place"][:1],
                       "layout_constraints": [{"type": "keepout", "hard": False, "polygon_xy": polygon}]}
                room, ctx = request_to_room(req)
                self.assertEqual(room["boundary"], req["room"]["boundary_xy"])
                self.assertEqual(ctx["floor"].area, 16)
                self.assertEqual((ctx["x0"], ctx["y0"]), (0, 0))
                self.assertEqual(room["constraints"], [])
                status, out = serve.answer(req, lambda _: one_box_text(x))
                self.assertEqual(status, 200)
                self.assertEqual(out["placements"][0]["position_m"], [x, 2, 0])
                for phase in ("raw", "repaired"):
                    v = out["validation"][phase]
                    self.assertTrue(v["ok"])
                    report = next(c for c in v["constraints"] if c["constraint"] == ["keepout", polygon])
                    self.assertEqual(report["holds"], expected)
                    self.assertFalse(report["hard"])
                    self.assertFalse(report["model_conditioned"])

    def test_full_room_soft_keepout_is_not_rejected_before_generation(self):
        req = request()
        req["layout_constraints"] = [{"type": "keepout", "hard": False,
                                      "polygon_xy": req["room"]["boundary_xy"]}]
        status, out = serve.answer(req, lambda _: model_text())
        self.assertEqual(status, 200)
        self.assertFalse(out["validation"]["repaired"]["constraints"][0]["holds"])

    def test_mixed_keepouts_report_in_request_frame_and_only_cut_hard_geometry(self):
        req = {**request(), "room": {"floor_z": 3,
               "boundary_xy": [[10, 20], [14, 20], [14, 24], [10, 24]]},
               "objects_to_place": request()["objects_to_place"][:1]}
        hard = {"type": "keepout", "polygon_xy": [[10, 20], [11, 20], [11, 24], [10, 24]]}
        soft = {"type": "keepout", "hard": False,
                "polygon_xy": [[12, 20], [13, 20], [13, 24], [12, 24]]}
        req["layout_constraints"] = [hard, soft]
        room, ctx = request_to_room(req)
        hard_room, hard_ctx = request_to_room({**req, "layout_constraints": [hard]})
        self.assertEqual(room["boundary"], hard_room["boundary"])
        self.assertTrue(ctx["floor"].equals(hard_ctx["floor"]))
        out = placements_from_text(one_box_text(12.5 - ctx["x0"]), room, ctx)
        self.assertEqual(out["placements"][0]["position_m"], [12.5, 22, 3])
        self.assertTrue(out["validation"]["repaired"]["ok"])
        report = next(c for c in out["validation"]["repaired"]["constraints"] if not c["hard"])
        self.assertEqual(report["constraint"], ["keepout", soft["polygon_xy"]])
        self.assertFalse(report["holds"])

    def test_hard_keepout_still_excludes_floor(self):
        req = {**request(), "objects_to_place": request()["objects_to_place"][:1],
               "layout_constraints": [{"type": "keepout", "polygon_xy": [[0, 0], [1, 0], [1, 4], [0, 4]]}]}
        _, ctx = request_to_room(req)
        self.assertAlmostEqual(ctx["floor"].area, 12)
        status, out = serve.answer(req, lambda _: one_box_text(.5 - ctx["x0"]))
        self.assertEqual(status, 422)
        self.assertEqual(out["error"]["validation"]["repaired"]["outside_1mm"], ["a"])

    def test_keepout_hard_flag_and_polygon_are_validated(self):
        for hard in ("false", 0, None, []):
            with self.subTest(hard=hard):
                req = {**request(), "layout_constraints": [{"type": "keepout", "hard": hard,
                       "polygon_xy": [[0, 0], [1, 0], [1, 4], [0, 4]]}]}
                with self.assertRaises(ValueError):
                    request_to_room(req)
        for polygon in ([], [[0, 0], [1, 0], [2, 0]], [[0, 0], [2, 2], [0, 2], [2, 0]]):
            for hard in (False, True):
                with self.subTest(polygon=polygon, hard=hard):
                    with self.assertRaises(ValueError):
                        request_to_room({**request(), "layout_constraints": [
                            {"type": "keepout", "hard": hard, "polygon_xy": polygon}]})

    def test_floor_id_must_be_a_string_before_generation(self):
        for floor_id in (["invalid"], {}, None, 1, True):
            with self.subTest(floor_id=floor_id):
                req = request()
                req["room"]["floor_id"] = floor_id
                generate = Mock(return_value=model_text())
                status, out = serve.answer(req, generate)
                self.assertEqual(status, 422)
                self.assertEqual(out["error"]["type"], "invalid_request_error")
                generate.assert_not_called()
        req = request()
        req["room"]["floor_id"] = "room-floor"
        status, out = serve.answer(req, lambda _: model_text())
        self.assertEqual(status, 200)
        self.assertEqual(out["placements"][1]["support_parent"], "room-floor")

    def test_windows_singular_and_plural_have_the_same_collision_policy(self):
        results = []
        for category in ("window", "windows", "sliding windows"):
            req = request()
            req["room"]["fixed_geometry"] = [{"id": "fx", "category": category,
                "size_xyz_m": [1, .1, 1.2], "position_m": [2, 2, .4]}]
            status, out = serve.answer(req, lambda _: model_text())
            self.assertEqual(status, 200)
            results.append(out["validation"]["repaired"])
        self.assertTrue(all(v["fixed_collisions"] == [["b", "fx"], ["a", "fx"]] for v in results))
        self.assertTrue(all(v["fixed_blocking"] == [] for v in results))

    def test_structural_wall_band_matches_training_and_respects_explicit_kind(self):
        from fastfill.build import _keep_fixed

        cases = [("door", None, True), ("windows", None, True), ("cabinet", None, False),
                 ("gate", "structure", True), ("door", "generic", False)]
        for category, kind, kept in cases:
            for x in (-.18, -.4):
                with self.subTest(category=category, kind=kind, x=x):
                    fx = {"id": "fx", "category": category, "size_xyz_m": [.07, 1.13, 2.21],
                          "position_m": [x, .67, 0], "yaw_rad": math.pi,
                          **({"fixed_kind": kind} if kind else {})}
                    req = request()
                    req["room"]["fixed_geometry"] = [fx]
                    room, ctx = request_to_room(req)
                    expected = kept and x == -.18
                    self.assertEqual(bool(room["fixed"]), expected)
                    training = _keep_fixed([{"id": "fx", "category": category, "size": fx["size_xyz_m"],
                        "pos": fx["position_m"], "yaw": math.pi,
                        "fixed_kind": kind or ("structure" if kept else "generic")}], [],
                        Polygon(req["room"]["boundary_xy"]))
                    self.assertEqual(len(room["fixed"]), len(training))
                    self.assertEqual(ctx["unsupported"], [] if expected else [{"id": "fx", "reason": "fixed_ignored"}])


class SupportTolerance(unittest.TestCase):
    def test_audited_raw_and_rounded_support_examples(self):
        # Minimal captured geometry only: portable regression fixtures, without audit-directory dependencies.
        cases = [{'label': 'layout_d0d0e745:geometry',
          'parent': {'category': 'shelf',
                     'size': [0.24, 0.68, 0.99],
                     'pos': [0.19, 3.85, 0.0],
                     'yaw': 0.0,
                     'id': 'p',
                     'parent': None},
          'child': {'category': 'bottle',
                    'size': [0.18, 0.26, 0.2],
                    'pos': [0.2, 3.46, 0.99],
                    'yaw': 6.178465552059927,
                    'id': 'c',
                    'parent': 'p'}},
         {'label': 'layout_d0d0e745:raw_geometry',
          'parent': {'category': 'shelf',
                     'size': [0.23705480407808122, 0.6822268177839373, 0.9850224800665455],
                     'pos': [3.85, 7.813972597960959, 0.0],
                     'yaw': 4.71238898038469,
                     'id': 'p',
                     'parent': None},
          'child': {'category': 'bottle1',
                    'size': [0.18238561321686464, 0.2597533206243067, 0.1978236176908561],
                    'pos': [3.4638687850577643, 7.798155180754215, 0.9749873957139794],
                    'yaw': 4.607920657487361,
                    'id': 'c',
                    'parent': 'p'}},
         {'label': 'layout_b4156653:geometry',
          'parent': {'category': 'bookshelf',
                     'size': [0.22, 0.65, 1.02],
                     'pos': [0.18, 4.45, 0.0],
                     'yaw': 0.0,
                     'id': 'p',
                     'parent': None},
          'child': {'category': 'bookshelfbook',
                    'size': [0.15, 0.32, 0.32],
                    'pos': [0.34, 4.36, 1.02],
                    'yaw': 1.3788101090755203,
                    'id': 'c',
                    'parent': 'p'}},
         {'label': 'layout_b4156653:raw_geometry',
          'parent': {'category': 'bookshelf',
                     'size': [0.22219992770050281, 0.6485300549974972, 1.0189980105172654],
                     'pos': [4.45, 3.821400036149748, 0.0],
                     'yaw': 4.71238898038469,
                     'id': 'p',
                     'parent': None},
          'child': {'category': 'bookshelfbook_5',
                    'size': [0.15457270169708442, 0.31913111738012023, 0.32005607231815225],
                    'pos': [4.356301125825175, 3.6611010282046643, 0.9947451596630095],
                    'yaw': 6.091614619437346,
                    'id': 'c',
                    'parent': 'p'}}]
        for case in cases:
            with self.subTest(case=case['label']):
                child, parent = case['child'], case['parent']
                self.assertEqual(validate.check({"objects": [parent, child]})["support_fail"], [])
                self.assertTrue(validate.support_ok(child, parent))

    def test_inclusive_xy_height_and_floor_tolerances_with_epsilon(self):
        parent = {"id": "p", "category": "table", "pos": [2, 2, 0], "size": [1, 1, 1], "yaw": 0}
        for delta, expected in ((0, True), (5e-10, True), (1e-8, False), (1e-5, False)):
            for axis in ("xy", "top", "bottom", "floor", "sunk"):
                with self.subTest(delta=delta, axis=axis):
                    offset = .05 + delta
                    pos = [2.5 + offset, 2, 1] if axis == "xy" else [2, 2, 1 + offset] if axis == "top" \
                        else [2, 2, 1 - offset] if axis == "bottom" else [2, 2, offset if axis == "floor" else -offset]
                    child = {"id": "c", "category": "box", "size": [.1, .1, .1], "pos": pos, "yaw": 0,
                             "parent": "p" if axis in ("xy", "top", "bottom") else None}
                    failures = validate.check({"objects": [parent, child]})["support_fail"]
                    self.assertEqual(failures, [] if expected else ["c"])
                    self.assertEqual(validate.support_ok(child, parent if child["parent"] else None), expected)

    def test_xy_tolerance_is_euclidean_including_rounded_buffer_corners(self):
        parent = {"pos": [0, 0, 0], "size": [1, 1, 1], "yaw": 0}
        for distance, expected in ((.05, True), (.050001, False)):
            angle = math.pi / 64  # between Shapely buffer vertices
            xy = [.5 + distance * math.cos(angle), .5 + distance * math.sin(angle)]
            self.assertEqual(validate.support_contains(parent, xy), expected)

    def test_missing_parent_still_fails(self):
        child = {"id": "c", "category": "box", "size": [.1, .1, .1], "pos": [1, 1, 0],
                 "yaw": 0, "parent": "missing"}
        self.assertEqual(validate.check({"objects": [child]})["support_fail"], ["c"])

    def test_repair_uses_the_same_inclusive_support_boundary(self):
        req = request()
        req["objects_to_place"][0]["size_xyz_m"] = [.1, .1, .2]
        room, ctx = request_to_room(req)
        text = json.dumps({"placements": [
            {"id": "table_1", "pos": [3.4, 2, 0], "yaw": 0},
            {"id": "box_1", "pos": [4, 2, .7], "yaw": 0, "on": "table_1"}]})
        out = placements_from_text(text, room, ctx)
        self.assertTrue(out["validation"]["repaired"]["ok"])
        self.assertAlmostEqual(out["placements"][0]["position_m"][0], 3.95)


class HttpRepairs(unittest.TestCase):
    def dispatch(self, data, length=None, path="/v1/chat/completions"):
        generate = Mock(return_value=model_text())
        handler = serve.make_handler(generate, "ff-test", "/ckpt/ff-test").__new__(
            serve.make_handler(generate, "ff-test", "/ckpt/ff-test"))
        handler.path = path
        handler.headers = {} if length is None else {"Content-Length": length}
        handler.rfile = Mock(wraps=io.BytesIO(data))
        handler.wfile = io.BytesIO()
        handler.send_response = Mock()
        handler.send_header = Mock()
        handler.end_headers = Mock()
        handler.do_POST()
        return handler, generate, handler.send_response.call_args.args[0], json.loads(handler.wfile.getvalue())

    def body(self, content=None):
        return json.dumps({"messages": [{"role": "user", "content": json.dumps(request()) if content is None else content}]}).encode()

    def test_invalid_content_lengths_are_rejected_before_read(self):
        for length in (None, "", "-1", "0", "abc", "1.5"):
            with self.subTest(length=length):
                handler, generate, status, out = self.dispatch(self.body(), length)
                self.assertEqual(status, 400)
                handler.rfile.read.assert_not_called()
                generate.assert_not_called()
                self.assertEqual(out["error"]["type"], "invalid_request_error")
                self.assertEqual(out["error"]["model"], "ff-test")
                self.assertEqual(out["error"]["checkpoint"], "/ckpt/ff-test")

    def test_maximum_content_length_is_inclusive_and_oversize_is_413(self):
        limit = getattr(serve, "MAX_REQUEST_BYTES", 1024 * 1024)
        handler, generate, status, out = self.dispatch(b"", str(limit + 1))
        self.assertEqual(status, 413)
        handler.rfile.read.assert_not_called()
        generate.assert_not_called()
        self.assertEqual(out["error"]["type"], "invalid_request_error")
        self.assertEqual(out["error"]["checkpoint"], "/ckpt/ff-test")
        body = self.body()
        padded = body + b" " * (limit - len(body))
        handler, _, status, _ = self.dispatch(padded, str(limit))
        self.assertEqual(status, 200)
        handler.rfile.read.assert_called_once_with(limit)

    def test_deep_outer_and_inner_json_receive_structured_400(self):
        nested = "[" * 10000 + "]" * 10000
        for body in (nested.encode(), self.body(nested)):
            with self.subTest(inner=body.startswith(b"{")):
                _, generate, status, out = self.dispatch(body, str(len(body)))
                self.assertEqual(status, 400)
                generate.assert_not_called()
                self.assertEqual(out["error"]["type"], "invalid_request_error")
                self.assertEqual(out["error"]["model"], "ff-test")

    def test_normal_http_response_and_errors_keep_the_existing_envelope(self):
        for content, expected in ((None, 200), ("null", 422), ("{bad json", 400)):
            body = self.body(content)
            _, generate, status, out = self.dispatch(body, str(len(body)))
            self.assertEqual(status, expected)
            if status == 200:
                generate.assert_called_once()
                result = json.loads(out["choices"][0]["message"]["content"])
                self.assertEqual(result["checkpoint"], "/ckpt/ff-test")
                self.assertEqual(result["model"], "ff-test")
                self.assertTrue(result["validation"]["repaired"]["ok"])
            else:
                self.assertEqual(out["error"]["model"], "ff-test")


class EmbeddedChecks(unittest.TestCase):
    def test_scene_validate_interface_selfchecks(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for name in ("scene", "validate", "interface"):
                with self.subTest(module=name):
                    runpy.run_module("fastfill." + name, run_name="__main__")

    def test_serve_answer_selfchecks_without_binding(self):
        root = ast.parse(Path(serve.__file__).read_text())
        main = next(n for n in root.body if isinstance(n, ast.If)
                    and ast.unparse(n.test) == "__name__ == '__main__'")
        nodes = []
        for node in main.body[0].orelse:
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "HTTPServer"
                   for n in ast.walk(node)):
                break
            nodes.append(node)
        self.assertTrue(any(isinstance(n, ast.Assert) for n in nodes))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), serve.__file__, "exec"), dict(vars(serve)))


if __name__ == "__main__":
    unittest.main()
