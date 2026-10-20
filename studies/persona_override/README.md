# persona_override

**Does a persistent persona trait make an LLM agent stop reading its neighbours, and do
aggregate metrics reveal it?**

A from-scratch SocioVerse2 study (Path B). Agents sit on a ring; each has a persistent
persona trait (one of the options) and updates its choice each step. The rule follows the
neighbour majority (tie → keep current) and ignores the persona. The LLM behaviour is
prompted with the neighbours and — when persona is on — the agent's trait. Four diagnostics
are recorded every step, plus a fallback audit.

## Reproduce

Rule mode is deterministic and needs **no API key**:

```bash
# from the SocioVerse2 repo root, in its venv
python studies/persona_override/run.py --seeds 0 1 2 3 4            # free, offline baseline
```

Live-LLM path (reads `SV_LLM_API_KEY`, `SV_LLM_BASE_URL`, `SV_LLM_MODEL` from `.env`):

```bash
python studies/persona_override/run.py --llm --seeds 0 1 2 3 4      # persona ON (default)
python studies/persona_override/run.py --llm --no-persona --seeds 0 1 2 3 4
python studies/persona_override/run.py --llm --options 4 --seeds 0 1 2 3 4
python studies/persona_override/run.py --llm --async --seeds 0 1 2 3 4
# the liveness probe: let it freeze, then show the first 5 agents disagreeing neighbours
python studies/persona_override/run.py --llm --flip-step 4 --flip-count 5 --seeds 0 1 2 3 4
```

Each run prints a per-seed summary and writes `results/<tag>.json` with the full per-step
trajectory and the fallback audit (`llm_decisions`, `rule_fallbacks`, `parse_failure_rate`).
`--emit-artifacts` (re)writes `study.yaml` and the `environment/` `population/` `simulation/`
JSON so the directory matches the standard `studies/<study>/` format.

## The conditions (2×2 ablation + controls)

| condition | persona in prompt | options | what it shows |
|---|---|---|---|
| rule | — | any | healthy neighbour-majority baseline (free) |
| LLM, no persona | no | 2, 4 | the LLM follows the rule; the population evolves |
| LLM, persona | yes | 2 | trait_alignment → 1, churn → 0; `B = f(P,E)` collapses to `B = f(P)` |
| LLM, persona | yes | 4 | freezes at maximum entropy ln 4 — a healthy-looking aggregate |
| LLM, persona, async | yes | 2 | synchronous oscillation gone, still freezes |
| LLM, persona, flip | yes | 2 | frozen agents shown disagreeing neighbours **do not move** — the override is real, not equilibrium |

## Diagnostics

- `choice_entropy` — entropy of the option distribution (nats).
- `churn` — share of agents who changed choice since the previous step (0 = frozen).
- `trait_alignment` — share whose current choice equals their persona trait.
- fallback audit — `llm_decisions`, `rule_fallbacks`, `parse_failure_rate`: a run that
  silently falls back to the rule is otherwise indistinguishable from real agreement, so
  the audit is what makes the live-LLM trajectory interpretable.

## What to read from a run

A healthy run keeps churn > 0 while the distribution is still moving, and ends where the
neighbour rule would. A persona-captured run reaches `trait_alignment ≈ 1` and `churn = 0`
within ~2 steps while option shares and entropy still look normal — which is the point: no
aggregate metric flags it. Only the liveness probe (`--flip-*`) separates a captured agent
from a legitimately converged one, by changing the neighbour information and checking the
output does not respond.

## Replication at scale (200 agents)

The effect is not a small-population artefact. At **200 agents, 6 steps, 3 seeds** on
`openai/gpt-4o-mini` (0 fallbacks, 0 parse failures), saved under `results/`:

| condition | final churn | trait alignment |
|---|---|---|
| persona ON (`scale200_llm_persona`) | **0.000 ± 0.000** | **1.000 ± 0.000** |
| persona OFF (`scale200_llm_nopersona`) | 0.895 ± 0.008 | 0.460 ± 0.004 |

Same pattern as the 20-agent ablation, held at 10× the population with near-zero variance:
with the persona the agents lock to their trait and stop responding to neighbours; without it
they keep churning near chance. (`scale200_scripted` is the free deterministic baseline.)

## Notes

- Stylised model, no real-world anchor: the claim is mechanistic (persona vs. neighbour
  information), not an effect size.
- Branch-replay eligible (single interaction round per step), so the flip intervention can
  be run as a counterfactual branch of a frozen parent on the hosted platform.
