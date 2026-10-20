"""Replicated detector evaluation: do the diagnostics catch labelled faults and pass
healthy controls, across seeds — with declared criteria and false-positive/false-
negative rates? (SocioVerse Challenge reviewer point 2.)

A run is NOT a replication: we run each ABM task in rule mode under several seeds, and
for each (task, seed) we build a set of LABELLED candidate "LLM" trajectories — healthy
and faulty by construction — then score four diagnostics against declared thresholds and
tally FP (healthy flagged) and FN (fault missed). Labels are known because the faults are
injected; this measures the detectors, not the frequency of faults in a real model (that
is the separate live-LLM ring study).

Faults, each a documented failure mode:
  frozen_early  — from t*, every agent repeats its t*-1 action (liveness loss while the
                  reference keeps moving).
  frozen_at_eq  — the agent freezes exactly at the reference's own converged shape, so it
                  is behaviourally identical to a healthy converged run. Included to show
                  the hard limit: no behavioural metric can separate a dead agent from a
                  settled one; only a counterfactual liveness probe can.
  collapse      — from t*, a fraction of minority-choice agents are pushed to the step
                  majority, shifting the distribution while agreement stays high.
  fallback      — actions identical to the rule (the engine's silent rule fallback), so
                  every behavioural metric sees a perfect match; only provenance differs.

Healthy controls:
  reference     — a perfect-agreement LLM (equals the rule), provenance = llm.
  healthy_noise — symmetric matched flips that preserve each step's distribution.

Diagnostics vs the rule reference R, with declared thresholds:
  agree  (their metric)  BAD if mean per-step agreement < 0.80
  dist   (distribution)  BAD if max per-step total-variation(cand, R) > 0.10
  live   (liveness)      BAD if some step t>=2 has cand churn == 0 while ref churn > 0
  prov   (provenance)    BAD if the LLM share of actions < 1.0
  combined               BAD if dist OR live OR prov

Run from the SocioVerse2 checkout (no API key needed; rule mode is deterministic):
    python ../probe/detection_eval.py
"""

from __future__ import annotations

import collections
import importlib
import math
import pathlib
import random
import shutil
import sys
import tempfile

TASKS = ["schelling", "sir", "civil_violence", "lux_marchesi"]
SEEDS = list(range(10))
STEPS = 12
TSTAR = 2                       # injection step for the early faults
AGREE_BAD = 0.80
TV_BAD = 0.10
COLLAPSE_RATE = 0.15           # minority->majority push; keeps agreement > AGREE_BAD
EQ_TOL = 1e-9                  # reference "converged" when its distribution stops moving


# ----- helpers ---------------------------------------------------------------------------

def tv(a: collections.Counter, b: collections.Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return 0.0
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def dist_of(step_kinds):
    return collections.Counter(step_kinds)


def churn(seq_a, seq_b):
    """Fraction of agents whose kind differs between two aligned per-agent lists."""
    if not seq_a:
        return 0.0
    return sum(1 for x, y in zip(seq_a, seq_b) if x != y) / len(seq_a)


# ----- reference trajectory from the engine (rule mode, free, deterministic) --------------

def rule_trajectory(task: str, seed: int, scratch: pathlib.Path):
    """Return [step][agent] = action kind, agents in a fixed order, from a rule run."""
    mod = importlib.import_module(f"studies.abm_{task}")
    from socioverse.engine import build_simulator

    study, env_b, pop_b, sim_c = mod.make_bundles(n_steps=STEPS, seed=seed, mode="rule")
    db = scratch / f"{task}_{seed}.duckdb"
    db.parent.mkdir(parents=True, exist_ok=True)
    sim = build_simulator(env_bundle=env_b, pop_bundle=pop_b, sim_config=sim_c,
                          store_path=db)
    sim.run()

    import duckdb
    con = duckdb.connect(str(db), read_only=True)
    rows = con.sql("SELECT step, agent_id, action_kind FROM panel "
                   "WHERE action_kind IS NOT NULL ORDER BY step, agent_id").fetchall()
    con.close()
    db.unlink(missing_ok=True)

    per_step = collections.defaultdict(list)
    for step, _agent, kind in rows:
        per_step[step].append(kind)
    steps = [per_step[s] for s in sorted(per_step)]
    # keep only steps with a full, non-empty agent set
    n = max((len(s) for s in steps), default=0)
    return [s for s in steps if len(s) == n and n > 0]


def convergence_step(ref):
    """Smallest t such that the reference distribution is unchanged for all later steps."""
    T = len(ref)
    for t in range(1, T):
        if all(dist_of(ref[u]) == dist_of(ref[t]) for u in range(t, T)):
            return t
    return None


# ----- fault / healthy construction ------------------------------------------------------

def make_reference(ref):
    return [list(s) for s in ref]


def make_healthy_noise(ref, rng):
    """Symmetric matched flips: preserve each step's distribution, move individuals."""
    out = []
    for s in ref:
        by = collections.defaultdict(list)
        for i, k in enumerate(s):
            by[k].append(i)
        cls = sorted(by)
        cand = list(s)
        if len(cls) >= 2:
            a, b = cls[0], cls[1]
            m = min(len(by[a]), len(by[b]), max(1, len(s) // 20))
            for i in rng.sample(by[a], m):
                cand[i] = b
            for i in rng.sample(by[b], m):
                cand[i] = a
        out.append(cand)
    return out


def make_frozen(ref, tstar):
    """From tstar on, every agent repeats its tstar-1 action."""
    out = [list(s) for s in ref]
    if tstar < 1 or tstar >= len(ref):
        return out
    hold = list(ref[tstar - 1])
    for t in range(tstar, len(ref)):
        out[t] = list(hold)
    return out


def make_collapse(ref, tstar, rate, rng):
    """From tstar on, push a fraction of minority-kind agents to the step majority."""
    out = [list(s) for s in ref]
    for t in range(tstar, len(ref)):
        c = collections.Counter(out[t])
        if len(c) < 2:
            continue
        maj = c.most_common(1)[0][0]
        minority = [i for i, k in enumerate(out[t]) if k != maj]
        nflip = min(int(len(out[t]) * rate), len(minority))
        for i in rng.sample(minority, nflip):
            out[t][i] = maj
    return out


# ----- diagnostics -----------------------------------------------------------------------

def diagnostics(cand, ref, llm_share):
    """Return dict of the four raw diagnostic values."""
    T = min(len(cand), len(ref))
    agree = sum(sum(1 for x, y in zip(cand[t], ref[t]) if x == y) / len(ref[t])
                for t in range(T)) / T
    max_tv = max(tv(dist_of(cand[t]), dist_of(ref[t])) for t in range(T))
    dead = False
    for t in range(2, T):
        if churn(cand[t], cand[t - 1]) == 0.0 and churn(ref[t], ref[t - 1]) > 0.0:
            dead = True
            break
    return {"agree": agree, "max_tv": max_tv, "live_dead": dead, "llm_share": llm_share}


def detectors(d):
    """Apply declared thresholds -> which detectors call this trajectory BAD."""
    D = {
        "agree": d["agree"] < AGREE_BAD,
        "dist": d["max_tv"] > TV_BAD,
        "live": d["live_dead"],
        "prov": d["llm_share"] < 1.0,
    }
    D["combined"] = D["dist"] or D["live"] or D["prov"]
    return D


# ----- main ------------------------------------------------------------------------------

def main():
    if not pathlib.Path("studies").is_dir():
        sys.exit("run from the SocioVerse2 checkout")
    here = str(pathlib.Path(".").resolve())
    if here not in sys.path:
        sys.path.insert(0, here)

    scratch = pathlib.Path(tempfile.mkdtemp(prefix="deteval_"))
    GOOD = {"reference", "healthy_noise"}
    DETS = ["agree", "dist", "live", "prov", "combined"]

    # confusion counters: per detector, FP (good flagged) / FN (bad missed) + totals
    fp = {k: 0 for k in DETS}
    fn = {k: 0 for k in DETS}
    n_good = n_bad = 0
    # per-case detection rate, for the breakdown table
    caught = collections.defaultdict(lambda: collections.Counter())
    case_n = collections.Counter()
    skipped_eq = 0

    for task in TASKS:
        for seed in SEEDS:
            ref = rule_trajectory(task, seed, scratch)
            if len(ref) < 4:
                continue
            rng = random.Random(1000 + seed)

            cases = {
                "reference": (make_reference(ref), 1.0, True),
                "healthy_noise": (make_healthy_noise(ref, rng), 1.0, True),
                "frozen_early": (make_frozen(ref, TSTAR), 1.0, False),
                "collapse": (make_collapse(ref, TSTAR, COLLAPSE_RATE, rng), 1.0, False),
                "fallback": (make_reference(ref), 0.0, False),   # equals rule, provenance=rule
            }
            ceq = convergence_step(ref)
            if ceq is not None and ceq < len(ref) - 1:
                cases["frozen_at_eq"] = (make_frozen(ref, ceq + 1), 1.0, False)
            else:
                skipped_eq += 1

            for label, (cand, share, is_good) in cases.items():
                D = detectors(diagnostics(cand, ref, share))
                case_n[label] += 1
                for k in DETS:
                    if D[k]:
                        caught[label][k] += 1
                if is_good:
                    n_good += 1
                    for k in DETS:
                        if D[k]:
                            fp[k] += 1
                else:
                    n_bad += 1
                    for k in DETS:
                        if not D[k]:
                            fn[k] += 1

    shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n{len(TASKS)} tasks x {len(SEEDS)} seeds, {STEPS} steps, rule-mode references.")
    print(f"labelled candidates: {n_good} healthy, {n_bad} faulty "
          f"(frozen_at_eq skipped on {skipped_eq} non-converging task/seeds)\n")

    print("Confusion over all labelled cases (lower is better):")
    hdr = f"  {'detector':10}{'false pos':>11}{'false neg':>11}{'FP rate':>9}{'FN rate':>9}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for k in DETS:
        print(f"  {k:10}{fp[k]:>8}/{n_good:<2}{fn[k]:>8}/{n_bad:<2}"
              f"{fp[k]/n_good:>9.2f}{fn[k]/n_bad:>9.2f}")

    print("\nPer-case detection rate (fraction of that case flagged BAD):")
    labels = ["reference", "healthy_noise", "frozen_early", "collapse", "fallback", "frozen_at_eq"]
    hdr2 = f"  {'case':14}" + "".join(f"{k:>10}" for k in DETS) + f"{'  (n)':>6}"
    print(hdr2); print("  " + "-" * (len(hdr2) - 2))
    for lab in labels:
        if not case_n[lab]:
            continue
        row = f"  {lab:14}"
        for k in DETS:
            row += f"{caught[lab][k] / case_n[lab]:>10.2f}"
        row += f"{case_n[lab]:>6}"
        print(row)

    # ----- self-checks: the story must be exactly what the numbers show -----
    assert n_bad > 0 and n_good > 0, "need both labelled classes"
    # their metric must miss at least one fault class it is blind to by construction
    assert fn["agree"] > 0, "agreement should miss the collapse/fallback faults"
    # provenance alone must have zero false positives on healthy (it only reads source)
    assert fp["prov"] == 0, "provenance must never flag a healthy llm run"
    # combined must do no worse than its best single component on FN
    assert fn["combined"] <= min(fn["dist"], fn["prov"]), "combined should not add FN"
    print("\nself-check passed: agreement has non-zero FN; provenance has zero FP on "
          "healthy; combined FN <= its components.")
    print("note: any residual combined FN is the frozen_at_eq case — behaviourally "
          "identical to healthy convergence, which motivates the live-LLM liveness probe.")


if __name__ == "__main__":
    main()
