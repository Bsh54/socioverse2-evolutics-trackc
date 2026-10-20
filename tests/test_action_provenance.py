"""Action provenance sources — EVOLUTICS, Track C (third platform fix).

A decision model records how each action was produced via Action.source. The schema's Literal
omitted "error" (a request that raised/failed and fell back to the rule), so a model that tried to
log a failed request crashed on Action construction instead — hiding request failures from any
provenance audit. This pins that all five provenance sources are valid, "error" included.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from socioverse.schemas.runtime import Action


def _action(source):
    return Action(agent_id="a1", step=1, kind="choose", payload={"choice": 0}, source=source)


def test_all_provenance_sources_are_valid():
    for s in ("llm", "rule", "fallback", "replay", "error"):
        assert _action(s).source == s


def test_error_source_no_longer_crashes_action_construction():
    # before the fix this raised ValidationError, crashing the run on the first request failure
    assert _action("error").source == "error"


def test_unknown_source_is_still_rejected():
    with pytest.raises(ValidationError):
        _action("totally-made-up")
