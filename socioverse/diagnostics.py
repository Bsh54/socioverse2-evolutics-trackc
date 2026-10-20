"""A reusable validity-audit collector for any SocioVerse2 study — EVOLUTICS, Track C.

The shipped evaluation habit is a single pairwise agreement number; we showed (detector evaluation)
that it misses ~78% of injected faults, and that three silent failure modes each produce reassuring
aggregates: a run where the LLM never executed (a silent rule/parse fallback) still scores perfect
agreement; a frozen population sits at maximum entropy; a branch that lost its replay source
cold-runs unpaired. This module turns the diagnostics that catch those into a DROP-IN collector any
`study.yaml` can enable, with no change to the study's own code or metrics.

Enable it by importing this module (so its `@register` runs) and pointing the run config at it:

    import socioverse.diagnostics            # registers "audit.diagnostics"
    sim.collector_ref  = "audit.diagnostics"
    sim.collector_args = {"inner_ref": "<study.collector>", "inner_args": {...}}  # optional

With `inner_ref` the study's own collector still runs and its metrics are kept; the audit columns
(prefixed ``dx_``) are added alongside. Without it, the audit columns are the whole metrics table.

The diagnostics are computed from the per-step ``actions`` the engine already produces — each
``Action`` carries a ``source`` (``llm`` / ``rule`` / ``fallback`` / ``error`` / ``replay``) — so
the collector is study-agnostic:

  provenance  dx_llm_share, dx_nonreal_share   -> catches a run the LLM never really drove
  dispersion  dx_entropy, dx_top_share         -> catches homogenization / a collapsed population
  liveness    dx_churn                         -> catches a frozen population (needs the flip probe
                                                  to separate a true freeze from equilibrium)

None of these is sufficient alone — that is the point (see the detector evaluation); declared
together they cover the faults the single agreement metric misses.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

from .abc.trajectory import MetricCollector
from .engine.registry import register, resolve
from .schemas.runtime import Action

# Provenance sources are exactly the Action schema's Literal: llm / rule / fallback / replay.
# "fallback" = the LLM ran but its output was unusable so the rule was substituted — the signal
# that a run reporting healthy aggregates may not have been driven by the model at all.
_AUDIT_COLUMNS = [
    "dx_n_actions", "dx_llm", "dx_rule", "dx_fallback", "dx_replay",
    "dx_llm_share", "dx_nonreal_share", "dx_entropy", "dx_top_share", "dx_churn",
]


def _signature(a: Action) -> str:
    """A canonical, study-agnostic label for an action: kind + sorted payload."""
    try:
        payload = json.dumps(a.payload, sort_keys=True, default=str)
    except Exception:
        payload = str(getattr(a, "payload", ""))
    return f"{a.kind}|{payload}"


@register("collector", "audit.diagnostics")
class AuditCollector(MetricCollector):
    """Validity diagnostics for any study; optionally wraps the study's own collector.

    Args (all optional, from ``collector_args``):
      inner_ref   registry key of the study's own collector to keep running (its metrics are
                  merged in; audit columns are added with a ``dx_`` prefix).
      inner_args  constructor args for that inner collector.
    """

    def __init__(self, inner_ref: str | None = None, inner_args: dict | None = None):
        self._inner = resolve("collector", inner_ref)(**(inner_args or {})) if inner_ref else None
        self._prev: dict[str, str] = {}          # agent_id -> last step's action signature (churn)

    def collect(self, env: Any, actions: list[Action], t: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self._inner is not None:
            out.update(self._inner.collect(env, actions, t))   # keep the study's own metrics

        acts = list(actions or [])
        n = len(acts)
        src = Counter(getattr(a, "source", "rule") or "rule" for a in acts)
        sig = Counter(_signature(a) for a in acts)

        entropy = 0.0
        for c in sig.values():
            p = c / n if n else 0.0
            if p > 0:
                entropy -= p * math.log(p)
        top_share = (max(sig.values()) / n) if n else 0.0

        cur = {a.agent_id: _signature(a) for a in acts}
        shared = [aid for aid in cur if aid in self._prev]
        churn = (sum(cur[aid] != self._prev[aid] for aid in shared) / len(shared)) if shared else 0.0
        self._prev = cur

        nonreal = src.get("fallback", 0)
        out.update({
            "step": t,
            "dx_n_actions": n,
            "dx_llm": src.get("llm", 0),
            "dx_rule": src.get("rule", 0),
            "dx_fallback": src.get("fallback", 0),
            "dx_replay": src.get("replay", 0),
            "dx_llm_share": round(src.get("llm", 0) / n, 4) if n else 0.0,
            "dx_nonreal_share": round(nonreal / n, 4) if n else 0.0,
            "dx_entropy": round(entropy, 4),
            "dx_top_share": round(top_share, 4),
            "dx_churn": round(churn, 4),
        })
        return out

    def columns(self) -> list[str]:
        inner = self._inner.columns() if self._inner is not None else []
        return list(inner) + _AUDIT_COLUMNS
