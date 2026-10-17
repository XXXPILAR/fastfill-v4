"""Integration seams of the 2026-10-06 audit fixes: verifier reads validity groups, loader migrates legacy rows,
the data CLI defaults to the axis policy, and every training config loads under the strict section validation."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fastfill.v2 import train
from fastfill.v2.io import read_samples
from fastfill.v2.legacy_verify import verify_selected_dataset
from fastfill.v2.losses import LossConfig
from fastfill.v2.model import ModelConfig
from fastfill.v2.tests.test_legacy_build import build, miniature_release, rows
from fastfill.v2.tests.test_legacy_verify import mutate_sample, replace_rows

CONFIGS = sorted(p for p in (Path(__file__).resolve().parents[1] / "configs").glob("*.json") if p.name != "direct_request.json")


def corpus(tmp_path):
    release, evidence = miniature_release(tmp_path)
    output = tmp_path / "dataset"
    build(release, evidence, output)  # default front policy: axis
    return output


def errors(report):
    return [error["message"] for error in report["errors"]]


def test_verifier_reads_groups_from_validity_and_needs_complete_position_only(tmp_path):
    output = corpus(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["exchangeable_group_counts"] == {"exchangeable_groups": 3, "exchangeable_members": 6}
    assert verify_selected_dataset(output)["passed"]

    def missing_size(row):  # size-incomplete members stay exchangeable (position-only cost)
        row["validity"]["size"][0] = [False, True, True]
    mutate_sample(output, "train", missing_size)
    manifest["valid_label_counts"]["size"] -= 1
    manifest["valid_label_counts"]["full_geometry"] -= 1
    (output / "manifest.json").write_text(json.dumps(manifest))
    report = verify_selected_dataset(output)
    assert report["passed"], report["errors"]

    def missing_position(row):
        row["validity"]["position"][0] = [True, False, True]
    mutate_sample(output, "validation", missing_position)
    assert any("complete position labels" in message for message in errors(verify_selected_dataset(output)))


def test_verifier_recomputes_group_yaw_and_symmetry_manifest_keys(tmp_path):
    output = corpus(tmp_path)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())

    def drop_group(row):  # a constraint on one member leaves no unreferenced pair, so no group is legal
        row["condition"]["constraints"].append({"type": "against_wall", "object_id": row["condition"]["objects"][0]["id"]})
        row["validity"]["exchangeable_group"] = [None, None]
    mutate_sample(output, "test", drop_group)
    assert "exchangeable_group_counts differs from independent streaming counts" in errors(verify_selected_dataset(output))
    manifest["exchangeable_group_counts"] = {"exchangeable_groups": 2, "exchangeable_members": 4}
    path.write_text(json.dumps(manifest))
    assert verify_selected_dataset(output)["passed"]

    for key, value, text in (("yaw_policy", "something else", "yaw_policy differs"),
                             ("descriptions", "category_only_no_asset_or_pose_text", "descriptions must be")):
        broken = {**manifest, key: value}
        path.write_text(json.dumps(broken))
        assert any(text in message for message in errors(verify_selected_dataset(output))), key
    path.write_text(json.dumps(manifest))

    def bad_order(row):
        row["validity"]["yaw_symmetry_order"][1] = 3
    mutate_sample(output, "train", bad_order)
    assert any("yaw_symmetry_order entries must be 1, 2 or 4" in message for message in errors(verify_selected_dataset(output)))


def test_verifier_requires_validity_group_list_and_source_descriptions(tmp_path):
    output = corpus(tmp_path)

    def legacy_shape(row):
        del row["validity"]["exchangeable_group"]
        row["provenance"]["descriptions"] = "category_only_no_asset_or_pose_text"
    mutate_sample(output, "train", legacy_shape)
    messages = errors(verify_selected_dataset(output))
    assert any("validity exchangeable_group must have exactly one entry per target" in m for m in messages)


def test_read_samples_migrates_legacy_condition_groups_once(tmp_path):
    output = corpus(tmp_path)
    sample = rows(output / "train.jsonl")[0]
    legacy = deepcopy(sample)
    groups = legacy["validity"].pop("exchangeable_group")
    legacy["condition"]["objects"] = [{**o, "exchangeable_group": g} for o, g in zip(legacy["condition"]["objects"], groups)]
    path = tmp_path / "legacy.jsonl"
    replace_rows(path, [legacy])
    migrated, = read_samples(path, training=True)
    assert migrated["validity"]["exchangeable_group"] == groups
    assert all("exchangeable_group" not in o for o in migrated["condition"]["objects"])
    assert migrated["condition"]["objects"] == sample["condition"]["objects"]


def test_data_cli_defaults_to_axis_front_policy(tmp_path):
    from fastfill.v2.data import main
    result = {"samples_written": 1, "split_samples": {"train": 1}}
    with patch("fastfill.v2.legacy_build.build_selected_dataset", return_value=result) as selected, redirect_stdout(StringIO()):
        main(["--output", str(tmp_path / "out")])
    assert selected.call_args.kwargs["front_policy"] == "axis"
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path / "out2"), "--front-policy", "unknown"])


def test_multiscan_adapter_puts_groups_in_validity_not_condition():
    from fastfill.v2.tests.test_data import multiscan_sample, region, row
    sample = multiscan_sample(region(), [row(), row("2")], {"scene_id": "house"})
    assert sample["validity"]["exchangeable_group"] == ["anonymous_chair", "anonymous_chair"]
    assert all("exchangeable_group" not in o for o in sample["condition"]["objects"])


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_every_training_config_loads_under_strict_sections(path):
    config = json.loads(path.read_text())
    assert not set(config) - set(train.CONFIG_SECTIONS)
    ModelConfig(**config.get("model", {}))
    loss = LossConfig(**config.get("loss", {}))
    assert loss.position_type in ("l1", "smooth_l1") and loss.size_type in ("l1", "smooth_l1")
    training = train._training_config(config.get("training", {}))
    optimizer = train._optimizer_config(config, training)
    assert optimizer["warmup_steps"] < training["steps"] or training["steps"] == 1
    assert train._augmentation_config(config)["rotate90"] is True
    assert train._validation_config(config)["exclude_flags"] == ["oob_objects", "fixed_collision", "overlapping_furniture"]


def test_cohort_completeness_accepts_legacy_rows_with_groups_in_objects(tmp_path):
    from fastfill.v2.cohort import _complete
    sample = rows(corpus(tmp_path) / "train.jsonl")[0]
    assert _complete(sample)
    legacy = deepcopy(sample)
    groups = legacy["validity"].pop("exchangeable_group")
    legacy["condition"]["objects"] = [{**o, "exchangeable_group": g} for o, g in zip(legacy["condition"]["objects"], groups)]
    assert _complete(legacy)
    assert "exchangeable_group" in legacy["condition"]["objects"][0]  # the caller's row is untouched
