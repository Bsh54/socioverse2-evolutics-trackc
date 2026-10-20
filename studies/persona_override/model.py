"""persona_override — a FROM-SCRATCH SocioVerse2 study (Path B).

Research question. When an LLM agent is given a persistent persona trait — the standard
"silicon sampling" construction — does the trait override the social mechanism (the agent
answers its own preference and stops reading its neighbours), and do aggregate metrics
reveal it?

The model is a k-option neighbour-choice process on a ring, the smallest setting that
isolates persona vs. environment:
  - P: a FIXED pool of ring agents with persistent ids (po-000, ...). Each carries one
    persistent persona_trait (an option) and an initial choice.
  - E (local physical): each agent observes its two ring neighbours' current choices. E
    owns the mutable choice state and folds decisions back in (apply), synchronously or in
    agent-id order (asynchronous). An optional INTERVENTION flips the neighbour view shown
    to a subset of agents from a chosen step, so a frozen agent can be tested for whether
    it still responds to changed neighbour information (the liveness probe the challenge
    reviewer asked for: freezing alone does not prove an agent stopped reading neighbours).
  - B: the rule is "follow the neighbour majority; on a tie keep your current choice" and
    ignores the persona. The LLM behaviour is prompted with the neighbours and — when
    use_persona is on — the agent's trait, then asked to choose. Any reply that does not
    parse falls back to the identical rule, and the action records its source
    (llm / fallback / rule) so a run can be audited for silent fallback.

One registered decision ref (po.decision) carries both modes (llm_kind scripted|openai),
as in opinion_diffusion. Metrics are the four diagnostics the study evaluates:
option shares, choice entropy, churn (share of agents who changed since the previous step),
and persona-trait alignment; plus a per-step fallback audit (llm / fallback counts and the
parse-failure rate) read from the action sources.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from typing import Any, Callable

from socioverse.abc import (
    DecisionModel,
    EnvironmentProvider,
    MetricCollector,
    PopulationProvider,
)
from socioverse.engine import register
from socioverse.schemas import (
    Action,
    EnvironmentBundle,
    EnvironmentLayer,
    InteractionStructure,
    Observation,
    Persona,
    PopulationBundle,
    PropagationMode,
    ScheduledEvent,
    SimulationConfig,
    StudySpec,
)

_log = logging.getLogger(__name__)

OPTION_LABELS = ["A", "B", "C", "D", "E", "F"]


def agent_id(i: int) -> str:
    return f"po-{i:03d}"


def ring_neighbors(i: int, n: int) -> list[int]:
    return [(i - 1) % n, (i + 1) % n]


def ring_edges(n: int) -> list[tuple[str, str]]:
    return [(agent_id(i), agent_id((i + 1) % n)) for i in range(n)]


def _seeded_initial(i: int, n: int, k: int, seed: int) -> int:
    """Deterministic initial choice for agent i: a fixed pseudo-random spread over k options."""
    h = (hash((seed, "choice", i)) & 0x7FFFFFFF)
    return h % k


def _seeded_trait(i: int, n: int, k: int, seed: int) -> int:
    """Deterministic persona trait for agent i, balanced across options where possible."""
    # round-robin by a seed-permuted index so traits are balanced but seed-dependent
    perm = (i + seed) % n
    return perm % k


# --- P ------------------------------------------------------------------------------------------

@register("population", "po.pop")
class RingPopulationProvider(PopulationProvider):
    def __init__(self, bundle: PopulationBundle):
        self.bundle = bundle
        a = bundle.provider_args
        self.n = int(a.get("n_agents", 20))
        self.k = int(a.get("n_options", 2))

    def build(self, seed: int) -> list[Persona]:
        out = []
        for i in range(self.n):
            c0 = _seeded_initial(i, self.n, self.k, seed)
            trait = _seeded_trait(i, self.n, self.k, seed)
            out.append(Persona(
                agent_id=agent_id(i),
                group_key=OPTION_LABELS[trait],
                attributes={"persona_trait": trait, "choice0": c0},
                init_state={"choice": c0, "persona_trait": trait},
            ))
        return out

    def neighbors(self, aid: str) -> list[str]:
        i = int(aid.split("-")[1])
        return [agent_id(j) for j in ring_neighbors(i, self.n)]


# --- E ------------------------------------------------------------------------------------------

@register("environment", "po.env")
class RingEnvironmentProvider(EnvironmentProvider):
    def __init__(self, bundle: EnvironmentBundle):
        self.bundle = bundle
        a = bundle.provider_args
        self.n = int(a.get("n_agents", 20))
        self.k = int(a.get("n_options", 2))
        self.async_update = bool(a.get("async_update", False))
        # optional liveness intervention: from flip_step, the first flip_count agents are shown
        # neighbours fixed to (own_choice + 1) % k — i.e. a neighbourhood that disagrees with them.
        self.flip_step = a.get("flip_step", None)
        self.flip_count = int(a.get("flip_count", 0))
        self.choices: dict[str, int] = {}
        self.traits: dict[str, int] = {}
        self._t = 0

    def reset(self, seed: int) -> None:
        self.choices = {agent_id(i): _seeded_initial(i, self.n, self.k, seed) for i in range(self.n)}
        self.traits = {agent_id(i): _seeded_trait(i, self.n, self.k, seed) for i in range(self.n)}
        self._t = 0

    def advance_to(self, t: int) -> list[ScheduledEvent]:
        self._t = t
        fired: list[ScheduledEvent] = []
        if self.flip_step is not None and t == int(self.flip_step):
            for ev in self.bundle.scheduled_events:
                if ev.at_step == t:
                    fired.append(ev)
        return fired

    def _flipped(self, aid: str) -> bool:
        if self.flip_step is None or self._t < int(self.flip_step):
            return False
        i = int(aid.split("-")[1])
        return i < self.flip_count

    def observe_batch(self, agent_ids: list[str], t: int, round_idx: int = 0) -> list[Observation]:
        out = []
        for aid in agent_ids:
            i = int(aid.split("-")[1])
            if self._flipped(aid):
                opp = (self.choices[aid] + 1) % self.k           # neighbours that disagree
                neigh = [opp, opp]
            else:
                neigh = [self.choices[agent_id(j)] for j in ring_neighbors(i, self.n)]
            out.append(Observation(
                agent_id=aid, step=t,
                local_physical={"own_choice": self.choices[aid],
                                "neighbor_choices": neigh,
                                "persona_trait": self.traits[aid],
                                "n_options": self.k},
            ))
        return out

    def apply(self, actions: list[Action]) -> None:
        # synchronous by default (all see the pre-step state via observe_batch); asynchronous
        # update is handled in the decision model, which re-reads choices in agent order.
        for a in actions:
            if a.kind == "choose":
                self.choices[a.agent_id] = int(a.payload["choice"])

    def agent_state(self, aid: str) -> dict:
        return {"choice": self.choices[aid], "persona_trait": self.traits[aid]}

    def snapshot(self) -> dict:
        c = Counter(self.choices.values())
        tot = sum(c.values()) or 1
        shares = {f"share_{OPTION_LABELS[o]}": round(c.get(o, 0) / tot, 4) for o in range(self.k)}
        return shares


# --- B ------------------------------------------------------------------------------------------

def _rule_choice(own: int, neigh: list[int]) -> int:
    """Follow the neighbour majority; on a tie keep the current choice. Persona is ignored."""
    if not neigh:
        return own
    c = Counter(neigh)
    top = max(c.values())
    winners = [o for o, n in c.items() if n == top]
    return own if (len(winners) != 1 or own in winners and top == c.get(own, 0)) else winners[0]


def _rule_choice_strict(own: int, neigh: list[int]) -> int:
    """Majority; tie -> keep own. (Clear reimplementation; ties = no single max.)"""
    if not neigh:
        return own
    c = Counter(neigh)
    top = max(c.values())
    winners = [o for o, n in c.items() if n == top]
    return winners[0] if len(winners) == 1 else own


class _OpenAILLM:
    def __init__(self, model: str | None = None, temperature: float = 0.5, max_tokens: int = 60):
        from socioverse.external_events import _load_dotenv
        _load_dotenv(None)
        self.model = model or os.environ.get("SV_LLM_MODEL", "gpt-4o-mini")
        self.temperature, self.max_tokens = temperature, max_tokens
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ.get("SV_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                base_url=os.environ.get("SV_LLM_BASE_URL", "https://api.openai.com/v1"))
        return self._client

    def __call__(self, prompt: str) -> str:
        # newer OpenAI families (gpt-5/gpt-6) reject the legacy max_tokens and a non-default
        # temperature; they use max_completion_tokens. Keep the legacy params for gpt-4/others.
        m = self.model.lower()
        newer = any(x in m for x in ("gpt-5", "gpt-6"))
        kwargs: dict = {"model": self.model,
                        "messages": [{"role": "user", "content": prompt}]}
        if newer:
            kwargs["max_completion_tokens"] = max(self.max_tokens, 256)
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = self.temperature
        r = self._ensure().chat.completions.create(**kwargs)
        return r.choices[0].message.content or ""


# persona phrasings on an increasing-strength gradient, for the adversarial robustness test:
# if the override already fires at "weak"/"moderate", it is not an artefact of an authoritarian
# prompt; if it needs "strong"/"identity", the finding is bounded and we say so.
PERSONA_MODES = {
    "none": None,
    "weak": "- You have a slight preference for option {t}.",
    "moderate": "- You generally tend to choose option {t}.",
    "strong": "- Your personal long-standing preference is option {t}.",
    "identity": "- You are the kind of person who always chooses option {t}.",
}


def build_choice_prompt(ob: Observation, persona_mode: str) -> str:
    k = ob.local_physical["n_options"]
    labels = OPTION_LABELS[:k]
    own = OPTION_LABELS[ob.local_physical["own_choice"]]
    neigh = [OPTION_LABELS[o] for o in ob.local_physical["neighbor_choices"]]
    lines = [
        f"You are one person choosing between options {', '.join(labels)}.",
        f"- Your current choice: {own}",
        f"- Your two neighbours currently chose: {neigh}",
    ]
    tmpl = PERSONA_MODES.get(persona_mode)
    if tmpl:
        trait = OPTION_LABELS[ob.local_physical["persona_trait"]]
        lines.append(tmpl.format(t=trait))
    lines += [
        "Choose the option you will go with for the next round.",
        'Output one line of JSON only: {"choice": "<one of ' + "/".join(labels) + '>", '
        '"reason": "<at most 8 words>"}',
    ]
    return "\n".join(lines)


_CHOICE_RE = re.compile(r'"choice"\s*:\s*"?([A-Fa-f])"?')
_REASON_RE = re.compile(r'"reason"\s*:\s*"([^"]*)"')


def parse_choice_reply(resp: str, k: int) -> dict | None:
    if not resp:
        return None
    m = _CHOICE_RE.search(resp)
    if m is None:
        m = re.search(r'\b([A-F])\b', resp)      # last-ditch: a bare option letter
    if m is None:
        return None
    letter = m.group(1).upper()
    if letter not in OPTION_LABELS[:k]:
        return None
    r = _REASON_RE.search(resp)
    return {"choice": OPTION_LABELS.index(letter), "reason": (r.group(1) if r else "").strip()}


@register("decision", "po.decision")
class NeighbourChoiceDecision(DecisionModel):
    def __init__(self, llm_kind: str = "scripted", use_persona: bool = True,
                 persona_mode: str | None = None,
                 model: str | None = None, temperature: float = 0.5, max_tokens: int = 60,
                 max_workers: int = 20, retries: int = 4, n_options: int = 2):
        self.llm_kind = llm_kind
        self.retries = max(1, int(retries))
        # persona_mode wins when given; else map the use_persona bool to strong/none (back-compat)
        self.persona_mode = persona_mode or ("strong" if use_persona else "none")
        if self.persona_mode not in PERSONA_MODES:
            raise ValueError(f"persona_mode must be one of {tuple(PERSONA_MODES)}")
        self.k = int(n_options)
        self.max_workers = max(1, int(max_workers))
        self.llm = _OpenAILLM(model=model, temperature=temperature,
                              max_tokens=max_tokens) if llm_kind == "openai" else None

    def _ask(self, ob: Observation) -> tuple[str | None, bool]:
        """Return (reply, errored). Retries transient gateway errors with backoff so a 502/429
        under concurrency does not masquerade as a model parse failure. errored=True means the
        request never succeeded (distinct from a reply that failed to parse)."""
        import time
        prompt = build_choice_prompt(ob, self.persona_mode)
        for attempt in range(self.retries):
            try:
                return self.llm(prompt), False
            except Exception as exc:
                if attempt == self.retries - 1:
                    _log.warning("LLM call failed for %s after %d tries: %s",
                                 ob.agent_id, self.retries, exc)
                    return None, True
                time.sleep(0.5 * (attempt + 1))
        return None, True

    def decide_batch(self, obs: list[Observation], memories: dict[str, Any]) -> list[Action]:
        if self.llm_kind == "openai" and len(obs) > 1:
            from concurrent.futures import ThreadPoolExecutor
            self.llm._ensure()
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(obs))) as ex:
                replies = list(ex.map(self._ask, obs))
        elif self.llm_kind == "openai":
            replies = [self._ask(ob) for ob in obs]
        else:
            replies = [(None, False)] * len(obs)

        actions = []
        for ob, (resp, errored) in zip(obs, replies):
            own = ob.local_physical["own_choice"]
            neigh = ob.local_physical["neighbor_choices"]
            rule_new = _rule_choice_strict(own, neigh)
            if self.llm_kind == "openai":
                if errored:                                   # request never succeeded
                    choice, source, reason = rule_new, "error", ""
                else:
                    dec = parse_choice_reply(resp, self.k)
                    if dec is not None:
                        choice, source, reason = dec["choice"], "llm", dec.get("reason", "")
                    else:                                     # succeeded but did not parse
                        choice, source, reason = rule_new, "fallback", ""
            else:
                choice, source, reason = rule_new, "rule", ""
            actions.append(Action(
                agent_id=ob.agent_id, step=ob.step, kind="choose",
                payload={"choice": int(choice), "reason": reason},
                source=source, rationale=[f"rule->{OPTION_LABELS[rule_new]}"],
            ))
        return actions


# --- metrics (the four diagnostics + the fallback audit) ----------------------------------------

@register("collector", "po.collector")
class DiagnosticCollector(MetricCollector):
    def __init__(self):
        self._prev: dict[str, int] | None = None

    def collect(self, env: Any, actions: list[Action], t: int) -> dict[str, Any]:
        choices = dict(env.choices)
        traits = dict(env.traits)
        n = len(choices) or 1
        c = Counter(choices.values())
        # entropy of the choice distribution (nats)
        ent = -sum((v / n) * math.log(v / n) for v in c.values() if v)
        # churn vs previous step
        if self._prev is None:
            churn = 0.0
        else:
            churn = sum(1 for a in choices if self._prev.get(a) != choices[a]) / n
        self._prev = choices
        # trait alignment
        align = sum(1 for a in choices if choices[a] == traits.get(a)) / n
        # fallback audit from action sources: llm (parsed) / fallback (parse-fail) / error (request)
        src = Counter(a.source for a in actions)
        llm_n, fb_n, err_n = src.get("llm", 0), src.get("fallback", 0), src.get("error", 0)
        attempted = llm_n + fb_n + err_n            # all LLM-mode decisions this step
        succeeded = llm_n + fb_n                     # requests that returned a reply
        out = {
            "choice_entropy": round(ent, 4),
            "churn": round(churn, 4),
            "trait_alignment": round(align, 4),
            "llm_decisions": llm_n,
            "rule_fallbacks": fb_n,
            "request_errors": err_n,
            "parse_failure_rate": round(fb_n / succeeded, 4) if succeeded else 0.0,
            "request_error_rate": round(err_n / attempted, 4) if attempted else 0.0,
        }
        tot = sum(c.values()) or 1
        for o in range(env.k):
            out[f"share_{OPTION_LABELS[o]}"] = round(c.get(o, 0) / tot, 4)
        return out

    def columns(self) -> list[str]:
        return ["choice_entropy", "churn", "trait_alignment", "llm_decisions",
                "rule_fallbacks", "request_errors", "parse_failure_rate", "request_error_rate"]


# --- artifact factory ---------------------------------------------------------------------------

def make_bundles(
    study_id: str = "persona_override",
    *,
    n_agents: int = 20,
    n_options: int = 2,
    n_steps: int = 10,
    seed: int = 42,
    llm_kind: str = "scripted",
    use_persona: bool = True,
    persona_mode: str | None = None,
    async_update: bool = False,
    flip_step: int | None = None,
    flip_count: int = 0,
    model: str | None = None,
    max_workers: int = 20,
) -> tuple[StudySpec, EnvironmentBundle, PopulationBundle, SimulationConfig]:
    k = int(n_options)
    metrics = (["choice_entropy", "churn", "trait_alignment", "llm_decisions",
                "rule_fallbacks", "request_errors", "parse_failure_rate", "request_error_rate"]
               + [f"share_{OPTION_LABELS[o]}" for o in range(k)])
    study = StudySpec(
        study_id=study_id,
        title="Persona override in neighbour-choice dynamics",
        research_question="Does a persistent persona trait make an LLM agent stop reading its "
                          "neighbours, and do aggregate metrics reveal it?",
        hypothesis="Without a persona the LLM follows the neighbour-majority rule; with a persona "
                   "it locks to its trait (trait_alignment -> 1, churn -> 0) while option shares and "
                   "entropy can still look healthy.",
        study_type="longitudinal", n_steps=n_steps, seed=seed, metrics=metrics,
        domain="opinion-dynamics",
        tags=["opinion-dynamics", "persona", "evaluation", "diagnostics", "from-scratch"],
        legacy_simulator="from_scratch",
        provider_refs=["po.env", "po.pop", "po.decision", "po.collector"],
        adjustable_params=["n_agents", "n_options", "n_steps", "use_persona", "async_update",
                           "flip_step", "flip_count", "llm_kind"],
        status="demo",
    )
    env_args: dict[str, Any] = {"n_agents": n_agents, "n_options": k, "async_update": async_update}
    scheduled: list[ScheduledEvent] = []
    if flip_step is not None:
        env_args["flip_step"] = int(flip_step)
        env_args["flip_count"] = int(flip_count)
        scheduled.append(ScheduledEvent(at_step=int(flip_step), target_layer="neighbor_choices",
                                        op="set", property_name="flipped", value=float(flip_count),
                                        note=f"flip neighbour view for the first {flip_count} agents"))
    env_bundle = EnvironmentBundle(
        study_id=study_id, provider_ref="po.env", provider_args=env_args,
        layers=[EnvironmentLayer(name="neighbor_choices", modality="physical", scope="local",
                                 dynamics="endogenous",
                                 description="ring neighbours' current choices")],
        scheduled_events=scheduled,
    )
    pop_bundle = PopulationBundle(
        study_id=study_id, provider_ref="po.pop",
        provider_args={"n_agents": n_agents, "n_options": k},
        interaction=InteractionStructure(kind="explicit_network", edges=ring_edges(n_agents)),
        propagation=PropagationMode.CONTAGION,
    )
    sim_config = SimulationConfig(
        study_id=study_id, n_steps=n_steps, seed=seed,
        decision_ref="po.decision",
        decision_args={"llm_kind": llm_kind, "use_persona": use_persona,
                       "persona_mode": persona_mode or ("strong" if use_persona else "none"),
                       "n_options": k, "model": model},
        collector_ref="po.collector", interaction_rounds=1,
    )
    return study, env_bundle, pop_bundle, sim_config
