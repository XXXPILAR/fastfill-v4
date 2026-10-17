from copy import deepcopy
import pytest

from fastfill.v2.schema import validate_condition
from fastfill.v2.tests.test_execution import sample


@pytest.mark.parametrize("targets", [None, ["b"], ["b", "b"], ["b", "missing"], [["b"], "c"], [1, "c"], ["a", "c"]])
def test_between_rejects_malformed_or_invalid_reference_list(targets):
    condition = deepcopy(sample()["condition"])
    condition["objects"] += [{"id": "b", "category": "chair", "description": "chair"},
                             {"id": "c", "category": "chair", "description": "chair"}]
    condition["constraints"] = [{"type": "between", "object_id": "a", "target_ids": targets}]
    with pytest.raises(ValueError, match="between"):
        validate_condition(condition)


def test_between_and_against_wall_validate_without_rewriting_condition():
    condition = deepcopy(sample()["condition"])
    condition["objects"] += [{"id": "b", "category": "chair", "description": "chair"},
                             {"id": "c", "category": "chair", "description": "chair"}]
    condition["constraints"] = [{"type": "between", "object_id": "a", "target_ids": ["b", "c"]},
                                {"type": "against_wall", "object_id": "b", "tolerance_m": .1}]
    before = deepcopy(condition)
    validate_condition(condition)
    assert condition == before
