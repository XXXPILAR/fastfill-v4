"""Actual Accelerator training: every flushed window averages its microbatches."""
from pathlib import Path
from datetime import timedelta
import json
import os
import sys

import pytest
import torch

from fastfill.v2 import train
from fastfill.v2.tests.test_active_objective_training import _data, _row, _yaw_config


class _ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.yaw_logits_head = torch.nn.Parameter(torch.zeros(()))

    def forward(self, **kwargs):
        return self.yaw_logits_head

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True)
        torch.save(self.state_dict(), directory / "scalar.pt")


class _ScalarCriterion:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def __call__(self, prediction, batch):
        values = [self.coefficients[int(row["scene_id"])] for row in batch["provenance"]]
        coefficient = sum(values) / len(values)
        loss = prediction * coefficient
        zero = prediction * 0.
        active = sum(value != 0. for value in values)
        return {"loss": loss, "yaw_cls": loss, "counts": {"yaw": active},
                "active_objective_count_local": active,
                "term_sums": {**dict.fromkeys(train.TERMS, zero.detach()), "yaw_cls": loss.detach()},
                "term_counts": {**dict.fromkeys(train.TERMS, 0), "yaw_cls": active},
                **{key: zero for key in ("position", "size", "yaw_reg", "box", "collision", "boundary")}}


def _patch_scalar_training(patches, coefficients, observed, *, scaled=False):
    # Keep actual preflight, Accelerator.prepare/backward, accumulation, clipping,
    # logging and optimizer dispatch. Replace only model/objective with a known
    # constant derivative and use lr=0 to observe gradients without changing it.
    original_loader = train.DataLoader
    patches.setattr(train, "DataLoader", lambda rows, **kw:
                    original_loader(rows, **{**kw, "shuffle": False}))
    patches.setattr(train, "build_model", lambda config: _ScalarModel())
    patches.setattr(train, "GeometryCriterion", lambda config: _ScalarCriterion(coefficients))

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            observed.append(float(self.param_groups[0]["params"][0].grad.detach()))
            return super().step(closure)

    patches.setattr(train.torch.optim, "AdamW", lambda params, **kw: RecordingSGD(list(params), lr=0.))
    if scaled:
        import accelerate
        original_accelerator = accelerate.Accelerator

        class CpuScaledAccelerator(original_accelerator):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Real torch AMP scaler on CPU exercises scaled backward,
                # Accelerate unscale/clip and GradScaler optimizer dispatch.
                self.scaler = torch.amp.GradScaler("cpu", init_scale=16.)

            def unscale_gradients(self, optimizer=None):
                # Accelerate enables fp16 unscale on accelerator devices only;
                # adapt that device gate to the real CPU GradScaler test.
                optimizers = self._optimizers if optimizer is None else [optimizer]
                for prepared in optimizers:
                    self.scaler.unscale_(prepared.optimizer)

        patches.setattr(accelerate, "Accelerator", CpuScaledAccelerator)


def _run_scalar(tmp_path, monkeypatch, coefficients, *, accumulation=2, clip=100., scaled=False, steps=None):
    observed = []
    _patch_scalar_training(monkeypatch, coefficients, observed, scaled=scaled)
    config = _yaw_config()
    updates = sum(any(coefficients[i:i + accumulation])
                  for i in range(0, len(coefficients), accumulation))
    config["training"] = {**config["training"], "gradient_accumulation_steps": accumulation,
                          "steps": updates if steps is None else steps, "clip_grad_norm": clip}
    rows = [_row(str(index)) for index in range(len(coefficients))]
    logs = train.run_training(config, _data(tmp_path, rows), tmp_path / "run")
    manifest = json.loads((tmp_path / "run/run_manifest.json").read_text())
    return observed, logs, manifest


@pytest.mark.parametrize("count,accumulation,expected", [
    (3, 2, [1., 1.]), (4, 2, [1., 1.]), (3, 1, [1., 1., 1.]),
    (1, 3, [1.]), (5, 3, [1., 1.]),
])
def test_actual_run_training_averages_complete_and_tail_windows(tmp_path, monkeypatch, count, accumulation, expected):
    observed, logs, manifest = _run_scalar(tmp_path, monkeypatch, [1.] * count, accumulation=accumulation)
    assert observed == pytest.approx(expected)
    assert len(logs) == manifest["steps_completed"] == len(expected)


def test_actual_tail_counts_empty_label_microbatches_in_declared_mean(tmp_path, monkeypatch):
    observed, logs, _ = _run_scalar(tmp_path, monkeypatch, [2., 0., 2., 2., 0.], accumulation=3)
    # Both windows count every microbatch, including those with no labels.
    assert observed == pytest.approx([4. / 3., 1.])
    assert [record["active_objective_count"] for record in logs] == [2, 1]


def test_actual_tail_retains_prior_supervision_and_counts_empty_last_batch(tmp_path, monkeypatch):
    observed, logs, _ = _run_scalar(tmp_path, monkeypatch, [1., 1., 1., 2., 0.], accumulation=3)
    assert observed == pytest.approx([1., 1.])
    assert [record["active_objective_count"] for record in logs] == [3, 1]


def test_actual_empty_complete_window_does_not_reweight_later_tail(tmp_path, monkeypatch):
    observed, logs, manifest = _run_scalar(tmp_path, monkeypatch, [0., 0., 1.])
    assert observed == pytest.approx([1.])
    assert logs[0]["active_objective_count"] == 1
    assert manifest["skipped_no_objective_windows"] == 1


@pytest.mark.parametrize("scaled", [False, True])
def test_actual_tail_rescale_precedes_clipping_and_amp_unscale(tmp_path, monkeypatch, scaled):
    observed, _, _ = _run_scalar(tmp_path, monkeypatch, [.8, .8, .8], clip=.75, scaled=scaled)
    assert observed == pytest.approx([.75, .75], rel=2e-6)


def test_amp_overflow_does_not_count_as_completed_optimizer_step(tmp_path, monkeypatch):
    # The first finite scalar loss has an overflowing *scaled* derivative, so
    # GradScaler skips its optimizer update. The following window is finite and
    # must become completed step 1; an attempted overflow is not a training step.
    observed, logs, manifest = _run_scalar(
        tmp_path, monkeypatch, [1e38, 1.], accumulation=1, scaled=True, steps=1)
    assert observed == pytest.approx([1.])
    assert [record["step"] for record in logs] == [1]
    assert manifest["steps_completed"] == 1
    assert manifest["skipped_gradient_overflow_windows"] == 1


def test_actual_partial_window_is_flushed_each_epoch(tmp_path, monkeypatch):
    observed, logs, _ = _run_scalar(tmp_path, monkeypatch, [1., 1., 3.], steps=4)
    # Each epoch ends with its own one-microbatch mean. No tail is combined
    # with the next epoch or used to reweight the previous complete window.
    assert observed == pytest.approx([1., 3., 1., 3.])
    assert [record["step"] for record in logs] == [1, 2, 3, 4]


def _distributed_tail_worker(rank, rendezvous, directory, mode):
    from torch import distributed as dist
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2",
                      OMP_NUM_THREADS="1", MASTER_ADDR="127.0.0.1", MASTER_PORT="29514")
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    patches = pytest.MonkeyPatch()
    try:
        if sys.platform == "darwin":
            patches.setattr(torch._C, "_get_accelerator", lambda: torch.device("cpu"))
        root = Path(directory)
        coefficients = {"all_valid": [1.] * 6, "peer_empty": [0., 2.] * 3,
                        "tail_one_rank_empty": [1., 1., 1., 1., 2., 0.],
                        "empty_tail": [1., 1., 1., 1., 0., 0.]}[mode]
        observed = []
        _patch_scalar_training(patches, coefficients, observed)
        if rank == 0:
            _data(root, [_row(str(index)) for index in range(6)])
        dist.barrier()
        config = _yaw_config()
        config["training"] = {**config["training"], "gradient_accumulation_steps": 2,
                              "steps": 2, "clip_grad_norm": 100.}
        logs = train.run_training(config, root / "train.jsonl", root / "run")
        # Real prepared loader has three microbatches per rank: one complete
        # K=2 window and one single-batch tail. DDP averages rank derivatives.
        assert observed == pytest.approx([1., 1.])
        assert [record["accumulation_microbatches"] for record in logs] == (
            [2, 2] if mode == "empty_tail" else [2, 1])
        expected_counts = {"all_valid": [4, 2], "peer_empty": [2, 1],
                           "tail_one_rank_empty": [4, 1], "empty_tail": [4, 4]}[mode]
        assert [record["active_objective_count"] for record in logs] == expected_counts
        if rank == 0:
            manifest = json.loads((root / "run/run_manifest.json").read_text())
            assert manifest["world_size"] == 2
            assert manifest["skipped_no_objective_windows"] == int(mode == "empty_tail")
        (root / f"rank-{rank}.json").write_text(json.dumps({
            "mode": mode, "optimizer_gradients": observed,
            "microbatches": [record["accumulation_microbatches"] for record in logs],
            "global_objective_counts": expected_counts}, indent=2) + "\n")
    finally:
        patches.undo()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(),
                    reason="requires Gloo")
@pytest.mark.parametrize("mode", ["all_valid", "peer_empty", "tail_one_rank_empty", "empty_tail"])
def test_actual_two_rank_complete_and_tail_means(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")  # CPU Gloo ranks: with one visible GPU, rank 1 would ask for cuda:1
    torch.multiprocessing.spawn(_distributed_tail_worker,
        args=("file://" + str(tmp_path / "rendezvous"), str(tmp_path), mode), nprocs=2, join=True)
    assert all((tmp_path / f"rank-{rank}.json").exists() for rank in range(2))
