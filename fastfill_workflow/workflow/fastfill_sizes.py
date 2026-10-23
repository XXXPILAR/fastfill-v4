"""Step ② of the draft workflow: one size per object from a FastFill checkpoint, predicted in this process.

One forward pass on the request's condition (as given, before the workflow fills room defaults: the input the model
was trained on). Only ``target_size_local_m`` is kept; positions and yaws are discarded. The size head is a linear
layer, so the size does not depend on the decode: argmax, spread and remask1L give identical sizes (checked on all
4541 test300 objects of the base arm), and this is the size the earlier test300 runs used.

    sizer = FastFillSizer("runs/main4-cell05-v4fixed-e5-base/model-step-6552", device="cuda")
    sizes = sizer(condition)          # {object id: [depth, width, height]}

A room the model cannot take (over its object or context budget) raises ``ValueError``; the pipeline then falls back
to the train catalog and records it.
"""

from __future__ import annotations

import time
from pathlib import Path


class FastFillSizer:
    def __init__(self, checkpoint: str | Path, *, device: str = "cuda"):
        from fastfill.v2.batch import load_tokenizer
        from fastfill.v2.io import load_checkpoint_config
        from fastfill.v2.model import load_model

        start = time.perf_counter()
        self.checkpoint, self.device = Path(checkpoint), device
        self.model = load_model(self.checkpoint, device=device).eval()
        backbone = self.model.config.backbone
        self.tokenizer = load_tokenizer("tiny" if backbone == "tiny" else str(self.checkpoint.parent / "tokenizer"),
                                        local_files_only=True)
        self.max_length = load_checkpoint_config(self.checkpoint)["max_length"] or 4096
        self.load_s = round(time.perf_counter() - start, 1)

    def __call__(self, condition: dict) -> dict[str, list[float]]:
        from fastfill.v2.evaluate import predict_layout

        layout = predict_layout(self.model, self.tokenizer, condition, max_length=self.max_length,
                                device=self.device, grid_decode="spread")
        return {box["id"]: [float(v) for v in box["target_size_local_m"]] for box in layout["objects"]}

    def describe(self) -> dict:
        return {"checkpoint": str(self.checkpoint), "device": self.device, "max_length": self.max_length,
                "load_s": self.load_s}
