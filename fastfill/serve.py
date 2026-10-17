"""OpenAI-compatible endpoint that speaks the WorldEdge request (scope doc Appendix B, fields in interface.py).

    python -m fastfill.serve --model outputs/ff-v1-merged --port 8001
    # EmbodiedGen side: WORLDEDGE_FASTFILL_URL=http://<host>:8001/v1 WORLDEDGE_FASTFILL_MODEL=ff-v1-merged
    python -m fastfill.serve                        # self-check (no model)

The last user message is the request JSON (the body's "model" field is not read). The reply's content is
placements_from_text's JSON ({"placements": [{"id", "position_m", "yaw_rad", "support_parent", "support_surface"?}], "raw_placements": [...],
"validation": {"raw", "repaired"}, "error": null, "unsupported": [...]}) plus "model" (--name, default the checkpoint
directory's name) and "checkpoint" (its absolute path): EmbodiedGen caches that content under its own
WORLDEDGE_FASTFILL_MODEL, so set that to the served name, and a cached layout still says which checkpoint made it.
A request with nothing to place (no objects, or all anchored to a wall or ceiling) is answered 200 with no placements
and its unsupported list, without calling the model (training never has an empty room).
HTTP 422 and no layout when the request is rejected (a prompt of --max_len tokens or more included: prompt_too_long),
the answer does not parse (a missing or extra object included), or the repaired layout still fails the canonical
checks: FastFill places every object validly or fails back to the caller, it never drops one. Every error body carries
model and checkpoint; a 422 after generation also the model's text (raw_text) and the unsupported list.
Greedy decoding, one request at a time.
The JSON body requires a positive Content-Length of at most MAX_REQUEST_BYTES (1 MiB). Missing/invalid lengths
and malformed or excessively nested JSON receive 400; oversized bodies receive 413 before any body read.
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from fastfill.interface import placements_from_text, request_to_room
from fastfill.scene import messages, prompt_text

MAX_REQUEST_BYTES = 1024 * 1024


class PromptTooLong(ValueError):
    """Raised by generate() for a prompt of max_len tokens or more (vLLM cannot start on it): answered with 422."""


def answer(req, generate):
    """Request dict -> (HTTP status, body). generate(model messages) -> model text, or raises PromptTooLong."""
    try:
        room, ctx = request_to_room(req)
    except ValueError as e:
        return 422, {"error": {"message": str(e), "type": "invalid_request_error"}}
    if not room["objects"]:                         # the empty answer is the only valid one: not generated
        return 200, placements_from_text('{"placements":[]}', room, ctx)
    try:
        text = generate(messages(room, room["constraints"], with_target=False))
    except PromptTooLong as e:
        return 422, {"error": {"message": str(e), "type": "invalid_request_error", "unsupported": ctx["unsupported"]}}
    out = placements_from_text(text, room, ctx)
    if out["error"]:
        return 422, {"error": {"message": f"model answer rejected: {out['error']}", "type": "invalid_layout",
                               "raw_text": text, "unsupported": out["unsupported"]}}
    if not out["validation"]["repaired"]["ok"]:     # every object must be placed validly; else the caller decides
        return 422, {"error": {"message": "layout fails validation even after repair", "type": "invalid_layout",
                               "raw_placements": out["raw_placements"], "validation": out["validation"],
                               "raw_text": text, "unsupported": out["unsupported"]}}
    return 200, out


def make_handler(generate, name, checkpoint):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if not self.path.endswith("/chat/completions"):
                return self.reply(404, {"error": {"message": "POST .../v1/chat/completions"}})
            try:
                length = int(self.headers["Content-Length"])
                if length <= 0:
                    raise ValueError("Content-Length must be positive")
                if length > MAX_REQUEST_BYTES:
                    return self.reply(413, {"error": {"message": f"body exceeds {MAX_REQUEST_BYTES} bytes",
                                                      "type": "invalid_request_error"}})
                body = json.loads(self.rfile.read(length))
                req = json.loads(body["messages"][-1]["content"])
            except (ValueError, KeyError, IndexError, TypeError, RecursionError) as e:
                return self.reply(400, {"error": {"message": f"bad body: {e!r}", "type": "invalid_request_error"}})
            try:
                status, out = answer(req, generate)
            except Exception as e:      # a bug or a dead model: say so instead of dropping the connection
                return self.reply(500, {"error": {"message": repr(e), "type": "server_error"}})
            if status == 200:
                content = json.dumps({**out, "model": name, "checkpoint": checkpoint})
                out = {"id": f"fastfill-{time.time_ns()}", "object": "chat.completion", "created": int(time.time()),
                       "model": name, "choices": [{"index": 0, "finish_reason": "stop",
                                                   "message": {"role": "assistant", "content": content}}],
                       "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
            self.reply(status, out)

        def reply(self, status, obj):
            if status != 200:
                obj["error"].update(model=name, checkpoint=checkpoint)
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged checkpoint dir")
    ap.add_argument("--name", help="model name echoed in replies (default: the checkpoint directory's name)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--max_len", type=int, default=40960, help="model sequence limit; the answer may use what the prompt leaves")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=a.max_len)
    tok = llm.get_tokenizer()

    def generate(msgs):
        p = prompt_text(tok, msgs)
        n = len(tok(p)["input_ids"])
        if n >= a.max_len:                  # vLLM raises on such a prompt: refuse it as a bad request instead
            raise PromptTooLong(f"prompt_too_long: {n} tokens, limit {a.max_len}")
        sp = SamplingParams(temperature=0, max_tokens=a.max_len - n)
        return llm.generate([p], sp, use_tqdm=False)[0].outputs[0].text
    ckpt = os.path.abspath(a.model) if os.path.isdir(a.model) else a.model      # a dir, or a hub id as given
    name = a.name or os.path.basename(os.path.normpath(ckpt))
    print(f"FastFill endpoint: http://{a.host}:{a.port}/v1  model {name} = {ckpt}", flush=True)
    HTTPServer((a.host, a.port), make_handler(generate, name, ckpt)).serve_forever()


if __name__ == "__main__":
    if sys.argv[1:]:
        main()
    else:                                           # self-check: answer() and the HTTP wrapper, a stub model
        import threading
        import urllib.error
        import urllib.request
        calls = []

        def stub(text):
            def gen(msgs):
                calls.append(json.loads(msgs[1]["content"]))
                if text is None:
                    raise PromptTooLong("prompt_too_long: 50000 tokens, limit 40960")
                return text
            return gen
        req = {"room": {"boundary_xy": [[0, 0], [4, 0], [4, 4], [0, 4]], "height_m": 2.8,
                        "fixed_geometry": [{"id": "col_1", "category": "column", "size_xyz_m": [0.3, 0.3, 2.8],
                                            "position_m": [3.5, 3.5, 0]}]},
               "objects_to_place": [{"id": "table_1", "category": "table", "size_xyz_m": [1.4, 0.8, 0.75]},
                                    {"id": "cup_1", "category": "cup", "size_xyz_m": [0.08, 0.08, 0.1]},
                                    {"id": "clock_1", "category": "clock", "size_xyz_m": [0.3, 0.05, 0.3], "anchor": "wall"}]}
        good = '{"placements":[{"id":"table_1","pos":[2,2,0],"yaw":0},{"id":"cup_1","on":"table_1","pos":[2,2,0.75],"yaw":0}]}'
        clock = [{"id": "clock_1", "reason": "unsupported_anchor"}]
        st, out = answer(req, stub(good))
        assert st == 200 and len(calls) == 1 and out["validation"]["repaired"]["ok"] and out["unsupported"] == clock, out
        # nothing placeable: no objects, or only wall / ceiling items -> 200, empty placements, the model never called
        for objs, unsup in (([], []), (req["objects_to_place"][2:], clock)):
            st, out = answer({**req, "objects_to_place": objs}, stub("never"))
            assert st == 200 and len(calls) == 1 and out["placements"] == [] and out["unsupported"] == unsup, out
            assert out["error"] is None and out["validation"]["repaired"]["ok"], out
        # 422 after generation keeps the model's text and the unsupported list: bad JSON, a missing object, a layout
        # that fails after repair (table through the column), a prompt over the limit (no text)
        on_col = good.replace("[2,2,0]", "[3.4,3.4,0]").replace("[2,2,0.75]", "[3.4,3.4,0.75]")
        for text, kind in (("{not json", "invalid_layout"), (good.split(',{"id":"cup_1"')[0] + "]}", "invalid_layout"),
                           (on_col, "invalid_layout"), (None, "invalid_request_error")):
            st, out = answer(req, stub(text))
            e = out["error"]
            assert st == 422 and e["type"] == kind and e["unsupported"] == clock and e.get("raw_text") == text, out
        assert out["error"]["message"].startswith("prompt_too_long")
        assert answer(req, stub(on_col))[1]["error"]["validation"]["repaired"]["fixed_blocking"] == [["table_1", "col_1"]]
        # over HTTP: every reply names the served model and checkpoint (200 in the content EmbodiedGen caches)
        texts = [good, "{not json"]
        srv = HTTPServer(("127.0.0.1", 0), make_handler(lambda m: texts.pop(0), "ff-test", "/ckpt/ff-test"))
        threading.Thread(target=srv.serve_forever, daemon=True).start()

        def post(obj):
            body = json.dumps({"model": "fastfill", "messages": [{"role": "user", "content": json.dumps(obj)}]}).encode()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_port}/v1/chat/completions", body) as f:
                    return f.status, json.load(f)
            except urllib.error.HTTPError as f:
                return f.code, json.load(f)
        st, rep = post(req)
        content = json.loads(rep["choices"][0]["message"]["content"])
        assert st == 200 and rep["model"] == "ff-test" and (content["model"], content["checkpoint"]) == ("ff-test", "/ckpt/ff-test")
        st, rep = post(req)
        assert st == 422 and rep["error"]["model"] == "ff-test" and rep["error"]["raw_text"] == "{not json", rep
        assert post({"room": {}})[0] == 422 and post({"room": {}})[1]["error"]["checkpoint"] == "/ckpt/ff-test"
        srv.shutdown()
        print("serve.py self-check ok")
