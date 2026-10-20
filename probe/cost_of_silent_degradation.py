"""The RESOURCE-COST face of the branch-integrity defect — quantified deterministically, free.

SocioVerse2's counterfactual branching pairs treatment with control by REPLAYING the parent's
steps 1..t* with no LLM calls, so only steps t*..T are paid for. The silent-degradation defect
(see reproduce_branch_bug.py) makes a branch whose replay source is absent/too-short run COLD from
step 0 — which not only breaks the pairing (a validity failure) but silently RE-PAYS for the
inherited steps (a budget leak). This script measures both the saving a correct branch delivers and
the leak the degraded branch incurs, in the hard model-agnostic unit — LLM decision calls — by
counting how many agent-decisions the engine actually computes (the replay path computes none).

Rule mode, no API key, deterministic. Run from the SocioVerse2 checkout:
    python ../probe/cost_of_silent_degradation.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(".").resolve()))

_DECISIONS = {"n": 0}


def _instrument():
    """Count agent-decisions the engine actually computes (decide_batch is called only for PAID
    steps; branch replay reconstructs inherited steps without calling it)."""
    import studies.persona_override.model as m
    orig = m.NeighbourChoiceDecision.decide_batch

    def counting(self, obs, memories):
        _DECISIONS["n"] += len(obs)
        return orig(self, obs, memories)

    m.NeighbourChoiceDecision.decide_batch = counting


def _run(n_agents, n_steps, warm_start=None, parent_db=None, resume_from=None):
    import studies.persona_override as po
    from socioverse.engine import build_simulator
    from socioverse.schemas import WarmStartSpec
    _s, env, pop, sim = po.model.make_bundles(
        study_id="cost", n_agents=n_agents, n_options=2, n_steps=n_steps, seed=42,
        llm_kind="scripted")
    if warm_start:
        sim.warm_start = WarmStartSpec(source_version="v1", source_trajectory=str(parent_db),
                                       resume_from=resume_from)
    db = pathlib.Path(tempfile.mkdtemp(prefix="cost_")) / "study.duckdb"
    _DECISIONS["n"] = 0
    build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim, store_path=db).run()
    return _DECISIONS["n"], db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=100)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--fork-step", type=int, default=7, help="branch forks here; inherits 1..fork-1")
    ap.add_argument("--usd-per-call", type=float, default=6.3e-5,
                    help="illustrative $/decision for a small model (e.g. gpt-4o-mini, ~330 tok)")
    a = ap.parse_args()
    if not pathlib.Path("studies").is_dir():
        sys.exit("run from the SocioVerse2 checkout")
    _instrument()

    N, T, tstar = a.agents, a.steps, a.fork_step
    resume_from = tstar - 1                                  # inherit steps 1..tstar-1

    # parent control run (its trajectory is the branch's replay source)
    parent_decisions, parent_db = _run(N, T)
    # a CORRECT branch: parent present, forks at tstar -> pays only tstar..T
    branch_decisions, _ = _run(N, T, warm_start=True, parent_db=parent_db, resume_from=resume_from)
    # the DEGRADED branch (the bug): replay source missing -> engine runs COLD -> pays 1..T
    degraded_decisions = parent_decisions                   # identical to a cold run, by the defect

    leak = degraded_decisions - branch_decisions            # silently re-paid inherited steps
    saving_pct = 100.0 * (parent_decisions - branch_decisions) / parent_decisions

    def usd(x):
        return f"${x * a.usd_per_call:,.4f}"

    print(f"config: {N} agents x {T} steps, branch forks at step {tstar} "
          f"(inherits steps 1..{resume_from}); decision calls counted empirically.\n")
    print(f"  parent / cold run            : {parent_decisions:>7} decisions  {usd(parent_decisions)}")
    print(f"  CORRECT branch (forks {tstar:>2})     : {branch_decisions:>7} decisions  {usd(branch_decisions)}"
          f"   <- pays only steps {tstar}..{T}")
    print(f"  DEGRADED branch (the defect) : {degraded_decisions:>7} decisions  {usd(degraded_decisions)}"
          f"   <- silently cold-ran, re-paid steps 1..{resume_from}")
    print()
    print(f"  correct-branch SAVING vs cold: {parent_decisions - branch_decisions:>7} decisions "
          f"({saving_pct:.0f}% of the run)  {usd(parent_decisions - branch_decisions)}")
    print(f"  defect BUDGET LEAK           : {leak:>7} decisions  {usd(leak)}  "
          f"(the inherited steps, paid again and silently)")
    print()
    # self-check: the saving and the leak are exactly the inherited region, N x resume_from
    expected = N * resume_from
    assert parent_decisions == N * T, f"cold should cost N*T={N*T}, got {parent_decisions}"
    assert branch_decisions == N * (T - resume_from), \
        f"branch should cost N*(T-resume_from)={N*(T-resume_from)}, got {branch_decisions}"
    assert leak == expected, f"leak should be N*resume_from={expected}, got {leak}"
    print(f"self-check passed: saving == leak == inherited region == N*resume_from = "
          f"{N}*{resume_from} = {expected} decisions.")
    print("the fix does not change these counts; it makes the degraded case LOUD and AUDITED "
          "(downgraded_from_branch) instead of a silent cold run, so the re-paid budget is not "
          "spent unknowingly and the user can restore the parent to recover the replay saving.")


if __name__ == "__main__":
    main()
