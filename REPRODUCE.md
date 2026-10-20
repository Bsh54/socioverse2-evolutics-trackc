# Reproduction — EVOLUTICS, Track C (SocioVerse Challenge 2026)

**Contribution.** A reliability/validity auditing method for LLM social simulation, demonstrated on
SocioVerse2: a diagnostic suite that catches silent validity failures aggregate metrics miss, and —
delivered as a reusable drop-in collector (`socioverse/diagnostics.py`, enable with one config
line on any study), plus two platform fixes with regression tests — a reproducible robustness gap in
the platform's counterfactual **branching** guarantee (a branch whose parent has no archived run to
replay silently cold-ran unpaired, mislabelled as a branch and unaudited) and a fork encoding bug
that silently corrupted non-ASCII artifacts. Scope is
stated honestly in README-EVOLUTICS.md §2: a degraded/off-happy-path state (not a routine `git
clone`, which drops the whole gitignored snapshot directory and is already caught), with the
scientific harm specific to real-LLM branches (rule-mode cold runs reproduce the trajectory
deterministically).

Everything in this file runs **with no API key, deterministically, free** (`llm_kind="scripted"` /
rule mode). This secures the eliminatory *reproducibility & engineering* dimension on its own; the
live-LLM figures are reported separately with full cost/provenance (see "Live-LLM results" below).

## 1. Environment

- Python 3.11+ (developed and verified on **3.14.8**, Windows 11). All dependencies ship wheels for
  3.14; no compiler needed.
- Install (from this checkout):
  ```bash
  python -m pip install -e ".[dev,viz,abm,llm]"
  ```
- **Always set `PYTHONUTF8=1`.** `reproduce.py` sets it itself, but export it in your shell too:
  ```bash
  export PYTHONUTF8=1        # bash          $env:PYTHONUTF8="1"   # PowerShell
  ```
  Why: `skills/sv_workspace.py` `fork_study` reads JSON without an explicit encoding, so on a
  non-UTF-8 Windows locale (cp1252) any non-ASCII content in a forked study is silently corrupted.
  The flag fixes it. (This is itself a portability finding; the shipped Chinese-language reports
  would be mangled without it.)

## 2. One command

```bash
cd SocioVerse2
python reproduce.py
```

Runs four stages, each printing `PASS`/`FAIL`; the script exits non-zero if any stage fails.
Wall-clock ~4–6 min, entirely offline.

| # | stage | what it establishes | cost |
|---|---|---|---|
| 1 | branch-integrity regression tests | the defect is real (a test that fails before the fix) and the fix is correct (passes after), via the repo's own pytest | free |
| 2 | branch-replay defect demonstrator | the silent degradation and the fix, offline, using the repo's own functions | free |
| 3 | detector evaluation | the method's quantitative core | free |
| 4 | persona-override ring (scripted) | the mechanistic companion | free |
| 5 | cost of the silent branch degradation | the resource-cost face: a correct branch's replay saving vs the budget the degraded branch silently re-pays | free |
| 6 | reusable audit collector (drop-in) | the diagnostics as a collector any `study.yaml` enables, shown on a study it was not written for | free |
| 7 | audit across the platform's studies | the same collector run unchanged over three shipped validation studies | free |

## 3. Expected results (verified 2026-10-20, this checkout)

**Stage 1 — branch-integrity tests.** All selected tests pass (one upstream path-format test
deselected — see §6). Includes the three regression tests for our fix:
`test_missing_parent_trajectory_store_downgrades` (facet 1 — replay source absent),
`test_parent_trajectory_too_short_for_fork_step_downgrades` (facet 2 — replay source too short), and
`test_valid_branch_still_resolves_when_parent_trajectory_present` (guards against over-correction).

**Stage 2 — branch-replay defect + fix.** With the parent trajectory present, a branch resolves its
warm start `{source: v1, resume_from: 2}`. When the parent version has a snapshot directory but no
archived run to replay (its trajectory store absent or empty — e.g. branching a version that was not
run to completion, which `create_branch` does not prevent), the as-shipped code returned `None`
**silently** and cold-ran the branch unpaired, still labelled a branch; after the fix,
`check_branch_invariant` catches it, **downgrades** the branch to a version, records
`downgraded_from_branch` with the reason, and `resolve_warm_start` returns `None` for that recorded
reason. Self-checks assert the fixed behaviour.

**Stage 3 — detector evaluation** (4 ABM tasks × 10 seeds × 12 steps; 60 healthy + 92 faulty
labelled cases):

| detector | FP rate | FN rate |
|---|---|---|
| agreement (the field-standard single metric) | 0.00 | **0.78** |
| distribution (max per-step TV) | 0.00 | 0.51 |
| liveness (churn vs reference) | 0.00 | 0.45 |
| provenance (LLM share) | 0.00 | 0.67 |
| **combined (dist ∨ live ∨ prov)** | 0.00 | **0.01** |

The single agreement metric misses 78% of injected faults; the combined suite misses 1% (the
residual is `frozen_at_eq`, behaviourally identical to healthy convergence — the case that motivates
the live-LLM liveness probe). No false positives on healthy trajectories.

**Stage 4 — persona-override ring, rule mode, 3 seeds.** Deterministic rule trajectory; prints
per-seed final churn / alignment / entropy and the fallback audit (0 fallbacks in rule mode).

**Stage 5 — resource cost of the silent branch degradation.** Counts the engine's actual
agent-decision calls (the replay path computes none) across a cold run, a correct branch, and the
degraded branch. For 100 agents × 12 steps forking at step 7: cold/degraded = 1200 decisions,
correct branch = 600 (pays only steps 7–12). The correct branch SAVES 600 decisions (50% of the
run); the defect silently RE-PAYS exactly those 600 (`saving == leak == N × resume_from`). The fix
does not change the counts — it makes the degraded case an audited downgrade instead of a silent
re-charge, so the budget is not spent unknowingly. (Illustrative $ at a small-model per-call rate;
the decision-call counts are the model-agnostic, exact figures.)

**Stage 6 — reusable audit collector (drop-in).** Enables `audit.diagnostics` on
`opinion_diffusion` (a study it was not written for), wrapping that study's own collector: the
study's metrics are kept and the `dx_*` validity columns are added, with one config line and no
study-code change.

**Stage 7 — audit across the platform's own studies.** Runs the same collector unchanged over
`opinion_diffusion`, `campus_dining_choice`, `germany_auto_market` (three of the platform's seven
validation studies), reporting each one's provenance / dispersion / liveness diagnostics; all read
healthy in rule mode (0 fallback, no collapse).

## 4. Reproducibility statement

- **Seeds.** Every run takes an explicit integer seed; stages 3–4 sweep fixed seed lists printed in
  the output. Rule mode is fully deterministic (no sampling).
- **Models.** The free core uses no model (rule/scripted decisions). Live-LLM runs used
  `openai/gpt-4o-mini` and `openai/gpt-6-luna` via the RodiumAI OpenAI-compatible gateway
  (`SV_LLM_BASE_URL=https://api.rodiumai.io/v1`), one decision call per agent per step (not batched).
- **Call counts & cost.** Free core: 0 API calls, $0. Live-LLM evidence (saved, not rerun here):
  ~2,330 `gpt-4o-mini` calls, 0 fallbacks, a few USD total; plus a gpt-6-luna / gemini sensitivity
  sweep. Every saved run records model, call volume, and the fallback audit in its JSON.
- **Dry-run vs real-LLM.** `reproduce.py` is the dry-run (no key, deterministic). The real-LLM path
  is `studies/persona_override/run.py --llm --seeds ...` (reads `SV_LLM_*` from `.env`); its saved
  outputs are under `studies/persona_override/results/`.
- **One command.** `python reproduce.py`, exit code 0 = all stages passed.

## 5. Data-compliance statement

- **No personal data in the free core.** The persona-override ring is a stylized k-option ring with
  synthetic agents; it declares no real-world anchor (see its `grounding/grounding.json`). The
  detector evaluation uses the shipped `abm_*` reference models in rule mode.
- **Code & platform.** `sii-research/SocioVerse2` (Apache-2.0) and `Lishi905/SocioVerse-ABM`.
- **ISSP / SocioBench** (only if the extended analyses are included): International Social Survey
  Programme via GESIS, downloadable free after a simple registration under GESIS data-usage terms;
  cited, used for academic research. Aggregate ISSP is long-public → memorization leakage is treated
  explicitly as a limitation, not ignored.
- **Persona pools.** The optional real-persona experiment pulls demographic personas from the
  SocioVerse user-pool MCP (`matraix_1m`); that pool's licence is **not declared upstream**, so those
  personas are used for research/demo only and are **not redistributed** in this package. The free
  core does not depend on them.
- **Leakage isolation.** The free core makes no external data calls and no point-in-time claims, so
  there is no ground-truth-window leakage to isolate.

## 6. Known cosmetic test failures (upstream, not our code)

Two upstream tests assert POSIX `/` path suffixes and fail **cosmetically on Windows** (they pass on
POSIX); the paths they check are correct. These pre-date our work:
`tests/test_branch_vs_version.py::test_resolve_warm_start_serves_a_valid_branch` and
`tests/test_dashboard_hook.py::test_only_this_checkouts_server_matches`. Stage 1 deselects the first
so the Windows dry-run verdict is clean; the full suite otherwise reports **282 passed, 2 cosmetic
failed, 12 skipped**.
