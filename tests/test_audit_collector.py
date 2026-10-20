"""The reusable validity-audit collector (socioverse/diagnostics.py) — EVOLUTICS, Track C.

Pins that the drop-in collector computes the three diagnostics from the per-step actions, in a
study-agnostic way, and that it can wrap a study's own collector without dropping its metrics.
"""
from __future__ import annotations

import math

from socioverse.abc.trajectory import MetricCollector
from socioverse.diagnostics import AuditCollector
from socioverse.engine.registry import register
from socioverse.schemas.runtime import Action


def _acts(choices, *, source="llm", step=1):
    return [Action(agent_id=f"a{i}", step=step, kind="choose",
                   payload={"choice": c}, source=source) for i, c in enumerate(choices)]


def test_provenance_flags_a_silent_fallback():
    c = AuditCollector()
    m = c.collect(None, _acts([0, 0, 0, 1], source="fallback"), 1)
    assert m["dx_llm"] == 0 and m["dx_fallback"] == 4
    assert m["dx_llm_share"] == 0.0 and m["dx_nonreal_share"] == 1.0   # the phantom-result signal


def test_dispersion_flags_homogenization():
    c = AuditCollector()
    allsame = c.collect(None, _acts([1, 1, 1, 1]), 1)
    assert allsame["dx_top_share"] == 1.0 and allsame["dx_entropy"] == 0.0   # collapsed population
    mixed = c.collect(None, _acts([0, 0, 1, 1]), 2)
    assert mixed["dx_top_share"] == 0.5
    assert mixed["dx_entropy"] == round(math.log(2), 4)      # two equal classes -> ln 2


def test_liveness_churn_tracks_changes_across_steps():
    c = AuditCollector()
    c.collect(None, _acts([0, 0, 0, 0], step=1), 1)          # establish the previous step
    m = c.collect(None, _acts([0, 0, 1, 1], step=2), 2)      # two of four agents changed
    assert m["dx_churn"] == 0.5
    frozen = c.collect(None, _acts([0, 0, 1, 1], step=3), 3)  # nobody moved
    assert frozen["dx_churn"] == 0.0


def test_replay_is_counted_as_provenance():
    c = AuditCollector()
    m = c.collect(None, _acts([0, 1, 0, 1], source="replay"), 1)
    assert m["dx_replay"] == 4 and m["dx_llm_share"] == 0.0


def test_wraps_inner_collector_and_keeps_its_metrics():
    @register("collector", "test.inner_demo")
    class _Inner(MetricCollector):
        def collect(self, env, actions, t):
            return {"my_metric": len(actions)}

        def columns(self):
            return ["my_metric"]

    c = AuditCollector(inner_ref="test.inner_demo")
    m = c.collect(None, _acts([1, 1]), 1)
    assert m["my_metric"] == 2 and "dx_entropy" in m          # study metric kept, audit added
    cols = c.columns()
    assert cols[:1] == ["my_metric"] and "dx_churn" in cols and "dx_top_share" in cols
