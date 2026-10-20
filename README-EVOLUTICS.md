# EVOLUTICS — Track C Full Repository (platform changes + reproduction)

SocioVerse Challenge 2026, Track C (Methodological and Systems Innovation). Team **EVOLUTICS**
(Shadrak BESSANH). This repository is a fork of `sii-research/SocioVerse2` (v0.2.0, Apache-2.0)
with a **platform-level fix to the branching subsystem** plus the audit experiments. This file is
the run-instructions README required for a Track C entry that modifies code outside `studies/`.

> Contribution in one line: a reliability/validity auditing method for LLM social simulation, whose
> highest-impact instance is a reproducible defect in SocioVerse2's flagship counterfactual
> **branching** feature — a branch could silently degrade to an unpaired cold run — now fixed so the
> degradation is caught, downgraded, and audited like every other broken branch contract.

## 1. Environment setup

- Python 3.11+ (developed and verified on 3.14.8, Windows 11; Linux/macOS also fine).
- From this repository root:
  ```bash
  python -m pip install -e ".[dev,viz,abm,llm]"
  export PYTHONUTF8=1          # bash   /   $env:PYTHONUTF8="1"  (PowerShell)
  ```
  `PYTHONUTF8=1` is required on non-UTF-8 Windows locales (see REPRODUCE.md §1 — an upstream
  `fork_study` reads JSON without an explicit encoding; the flag prevents silent cp1252 corruption).
- The 11 `abm_*` studies and Chicago need the companion repo `SocioVerse-ABM` cloned next to this one
  (already present in this package). No LLM key is needed for anything in §3.

## 2. The platform changes (what we modified outside studies/)

Three contributions to the platform (code outside `studies/`): **(2a)** a fix to the
branch-integrity guarantee, **(2b)** a fix to a fork encoding bug, and **(2c)** a new reusable
validity-audit collector. Each is minimal, tested, and independent.

### 2a. Branch-replay silent-degradation fix

**File:** `skills/sv_workspace.py`, function `check_branch_invariant` (plus one small read-only
helper `_parent_step_coverage`).

**Defect (as shipped).** A counterfactual branch replays its parent run's steps `1..resume_from`
(no LLM) so treatment and control are paired. `sv-run` sets the engine's `warm_start` from
`resolve_warm_start`, which reads the parent's archived trajectory store
(`versions/<source>/trajectory/study.duckdb`). `check_branch_invariant` guarded the branch
contract (P / f / seed / pre-fork E unchanged) — but it never validated its own **replay
precondition**: that the parent's realized history actually covers the inherited steps. When it does
not, the degradation is silent and the run is still labelled a `branch` with no
`downgraded_from_branch` audit record.

**The defect is a class, with two facets** (both verified empirically, deterministically, free):
- *Facet 1 — replay source absent.* The parent store is missing or empty, so `resolve_warm_start`
  returns `None` and the branch cold-runs from step 0.
- *Facet 2 — replay source too short.* The parent store exists but records fewer steps than the
  branch inherits (`resume_from`). The engine (`socioverse/engine/loop.py:_replay_inherited`) draws
  each replayed step's actions from the parent panel and, for a step the parent never recorded,
  silently replays an **empty** step — so a branch forked beyond the parent's realized horizon is
  unpaired on the uncovered steps, again with no error. (Confirmed: parent run to step 3, branch
  forked at step 6 → steps 4–5 replayed with zero actions, no exception.)

Both share one root cause: `check_branch_invariant` did not validate its own replay precondition —
that the parent's realized history actually covers steps `1..resume_from`.

**Fix.** The invariant now validates that precondition: the parent trajectory store must exist, be
non-empty, **and cover at least `resume_from` steps**. Any failure is a branch-contract violation,
so it takes the framework's **existing** downgrade path: `kind` → `"version"`, warm-start dropped,
`downgraded_from_branch` recorded with the reason, reported to the user. Silent degradation → loud,
audited downgrade — consistent with how every other broken branch contract is already handled. The
coverage read is decoupled (a local `duckdb` import, read-only) and **fails open**: if coverage
cannot be determined (non-duckdb store, unusual schema) the branch is NOT downgraded on that basis,
so a healthy branch is never falsely rejected. No API cost, no new hard dependency.

**Tests.** `tests/test_branch_vs_version.py` gains three regression tests:
`test_missing_parent_trajectory_store_downgrades` (facet 1) and
`test_parent_trajectory_too_short_for_fork_step_downgrades` (facet 2) each fail on the unpatched
code and pass after the fix; `test_valid_branch_still_resolves_when_parent_trajectory_present`
guards against over-correction. Full suite (all three contributions): 282 passed, 2 pre-existing cosmetic Windows path-format
failures (see REPRODUCE.md §6), 12 skipped.

**Scope / honest reachability.** This is a robustness gap in a degraded / off-happy-path state, not
a routine-workflow failure: in the normal /sv-iterate flow a branch is offered only after the parent
has run, so the store is present and long enough. It is reachable because `create_branch` has no
guard requiring a completed parent run of sufficient length, and because an archived store can be
absent, empty or truncated (an interrupted run, a cleared/partial run, restoring an older base whose
outputs were dropped, or external file operations on the large `*.duckdb`). It is **not** produced
by a plain `git clone` (`studies/*/versions/` is gitignored in full → the whole snapshot directory
is dropped, which the pre-existing directory check already catches). The scientific harm is specific
to **real-LLM** branches: in rule/scripted mode a cold or empty-step replay is deterministic, but
with a stochastic LLM the affected steps diverge from the parent's realized history, breaking the
treatment/control pairing while the manifest still presents the run as a branch.

### 2b. Fork encoding fix (silent non-ASCII corruption)

**File:** `skills/sv_workspace.py`, function `fork_study`. The id-rewrite path read each copied
artifact with a bare `read_text()`, whose codec is the process locale (cp1252 on a default Windows
install) while the write side already forced UTF-8 — so every fork silently corrupted non-ASCII
artifact content (the shipped reference studies carry Chinese text and typographic dashes, e.g. an
em dash U+2014 became three mojibake characters). The read now specifies `encoding="utf-8"`, matching
the write. Regression test: `tests/test_fork_encoding.py` (a fork preserves non-ASCII byte-for-byte
while still rewriting the id). Independently, `PYTHONUTF8=1` papers over it; the fix removes the
dependence on the ambient locale.

### 2c. Reusable validity-audit collector (new tool)

**File:** `socioverse/diagnostics.py` (new module), class `AuditCollector`, registered as
`@register("collector", "audit.diagnostics")`. This turns the paper's diagnostics into a drop-in any
study enables with one config line — `collector_ref="audit.diagnostics"`, optional
`collector_args={"inner_ref": "<study.collector>"}` to keep the study's own metrics — with no change
to the study's code. From the per-step actions (each carries a `source`) it reports provenance
(`dx_llm_share`, `dx_nonreal_share` — catches a run the LLM never really drove), dispersion
(`dx_entropy`, `dx_top_share` — catches homogenization / collapse) and liveness (`dx_churn` —
catches a frozen population). Tests: `tests/test_audit_collector.py`. Reusability is shown on a study
it was not written for (`../probe/audit_collector_demo.py`) and across three shipped validation
studies (`../probe/audit_breadth.py`), unchanged.

## 3. Reproduction (one command, no key, deterministic, free)

```bash
cd SocioVerse2
python reproduce.py
```

Four stages, each printing `PASS`/`FAIL`, exit non-zero if any fails (verified 2026-10-20 →
**4/4 PASS**):

1. **branch-integrity regression tests** — the fix is correct (the defect test fails before, passes
   after), via the repo's own pytest.
2. **branch-replay defect demonstrator** (`../probe/reproduce_branch_bug.py`) — the silent
   degradation and the fix, offline, using the repo's own functions, with self-checks.
3. **detector evaluation** (`../probe/detection_eval.py`) — the method's quantitative core: the
   field-standard agreement metric misses **78%** of injected faults; the combined suite misses
   **1%**; 0 false positives.
4. **persona-override ring, rule mode** (`studies/persona_override/run.py`) — the mechanistic
   companion, deterministic.
5. **resource cost of the silent degradation** (`../probe/cost_of_silent_degradation.py`) — counts
   the engine's actual decision calls: a correct branch forking at step 7 of 12 (100 agents) pays
   600 calls vs 1200 cold — a **50% replay saving**; the degraded branch silently **re-pays** those
   600 (`saving == leak == N × resume_from`). The fix makes that re-charge an audited downgrade, not
   a silent one. This is the branch defect's **resource-cost** face (Track C asks for effectiveness,
   cost, and scope).

Full reproducibility statement, data-compliance statement, expected numbers, cost/provenance for the
live-LLM figures, and the known cosmetic failures are in **`REPRODUCE.md`**.

## 4. Package layout

- `skills/sv_workspace.py` — the two platform fixes (§2a branch-integrity, §2b fork encoding).
- `socioverse/diagnostics.py` — the new reusable audit collector (§2c).
- `tests/test_branch_vs_version.py`, `tests/test_audit_collector.py`, `tests/test_fork_encoding.py`
  — the regression/unit tests for the three contributions.
- `reproduce.py`, `REPRODUCE.md` — one-command repro (7 stages) + statements (§3).
- `studies/persona_override/` — the from-scratch study (model/run/README + materialized artifacts +
  saved results, including the live-LLM runs).
- `../probe/` — the audit experiments: `detection_eval.py` (detector evaluation),
  `reproduce_branch_bug.py` (branch-bug demonstrator), `cost_of_silent_degradation.py` (the cost
  face), `audit_collector_demo.py` + `audit_breadth.py` (the reusable collector, on one study and
  across three), plus the provenance/silent-fallback probes, user-pool MCP client, germany
  robustness audit, and real-persona experiment.

Upstream `README.md` / `CLAUDE.md` document the unmodified platform; this file documents only what
EVOLUTICS changed and how to reproduce it.
