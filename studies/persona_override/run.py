#!/usr/bin/env python3
"""persona_override — one-command runner, replicated over seeds.

Default (no API key, deterministic, free):
    python studies/persona_override/run.py

Live-LLM path (reads SV_LLM_* from .env; model defaults to the SV_LLM_MODEL there):
    python studies/persona_override/run.py --llm --seeds 0 1 2 3 4

Key switches reproduce the study's conditions:
    --persona / --no-persona   put the agent's trait in the prompt (default: persona on)
    --options N                number of choice options (default 2)
    --steps N                  horizon (default 10; freeze, when it happens, is by step ~2)
    --async                    asynchronous (agent-id order) update instead of synchronous
    --flip-step T --flip-count K   from step T, show the first K agents neighbours that
                                   disagree with them — the liveness probe
    --emit-artifacts           also materialise study.yaml + the P/E/sim JSON into this dir

Results print as a per-seed summary and are written to results/<tag>.json. Every run records
model, call volume and the fallback audit so cost and provenance are auditable.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def one_run(seed, *, llm, model, n_agents, n_options, n_steps, persona, persona_mode,
            async_update, flip_step, flip_count):
    import studies.persona_override as po
    from socioverse.engine import build_simulator
    import duckdb

    study, env, pop, sim = po.model.make_bundles(
        n_agents=n_agents, n_options=n_options, n_steps=n_steps, seed=seed,
        llm_kind="openai" if llm else "scripted", use_persona=persona, persona_mode=persona_mode,
        async_update=async_update, flip_step=flip_step, flip_count=flip_count, model=model)
    db = pathlib.Path(tempfile.mkdtemp(prefix="po_")) / "study.duckdb"
    t0 = time.time()
    build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim, store_path=db).run()
    con = duckdb.connect(str(db), read_only=True)
    cols = ("step, choice_entropy, churn, trait_alignment, llm_decisions, rule_fallbacks, "
            "parse_failure_rate")
    rows = con.sql(f"SELECT {cols} FROM metrics ORDER BY step").fetchall()
    con.close()
    keys = cols.replace(" ", "").split(",")
    traj = [dict(zip(keys, r)) for r in rows]
    return {"seed": seed, "elapsed_s": round(time.time() - t0, 1), "trajectory": traj}


def settle(traj):
    """Summaries that matter: final churn, final alignment, final entropy, total calls/fallbacks."""
    last = traj[-1]
    post = [s for s in traj if s["step"] >= 2]
    return {
        "final_churn": last["churn"],
        "mean_churn_post2": round(st.mean(s["churn"] for s in post), 4) if post else 0.0,
        "final_alignment": last["trait_alignment"],
        "final_entropy": last["choice_entropy"],
        "llm_calls": sum(s["llm_decisions"] for s in traj),
        "fallbacks": sum(s["rule_fallbacks"] for s in traj),
        "max_parse_fail": max((s["parse_failure_rate"] for s in traj), default=0.0),
    }


def emit_artifacts(out_dir, **kw):
    """Materialise the studies/<study>/ artifacts so the package matches the required format."""
    import studies.persona_override as po
    from socioverse.validation import write_artifact
    from skills.sv_workspace import study_paths, write_roster
    study, env, pop, sim = po.model.make_bundles(**kw)
    p = study_paths(out_dir)
    for key, art in (("study", study), ("environment", env),
                     ("population", pop), ("simulation", sim)):
        p[key].parent.mkdir(parents=True, exist_ok=True)
        write_artifact(p[key], art)
    # roster at t=0 under the declared seed
    prov = po.model.RingPopulationProvider(pop)
    write_roster(p["roster"], prov.build(sim.seed))
    (out_dir / "grounding").mkdir(exist_ok=True)
    (out_dir / "grounding" / "grounding.json").write_text(json.dumps({
        "study_id": study.study_id,
        "facts": [], "assumptions": [
            {"id": "a1", "text": "Stylised k-option ring; no real-world anchor. The claim is "
             "mechanistic (persona vs. neighbour information), not an effect size."}],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"artifacts written under {out_dir}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="use the live LLM (SV_LLM_* in .env)")
    ap.add_argument("--model", default=None, help="override SV_LLM_MODEL")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--agents", type=int, default=20)
    ap.add_argument("--options", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    persona = ap.add_mutually_exclusive_group()
    persona.add_argument("--persona", dest="persona", action="store_true")
    persona.add_argument("--no-persona", dest="persona", action="store_false")
    ap.set_defaults(persona=True)
    ap.add_argument("--persona-mode", default=None,
                    choices=["none", "weak", "moderate", "strong", "identity"],
                    help="persona phrasing strength; overrides --persona/--no-persona")
    ap.add_argument("--async", dest="async_update", action="store_true")
    ap.add_argument("--flip-step", type=int, default=None)
    ap.add_argument("--flip-count", type=int, default=0)
    ap.add_argument("--emit-artifacts", action="store_true")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)

    cfg = dict(n_agents=a.agents, n_options=a.options, n_steps=a.steps,
               persona=a.persona, persona_mode=a.persona_mode, async_update=a.async_update,
               flip_step=a.flip_step, flip_count=a.flip_count)

    if a.emit_artifacts:
        emit_artifacts(HERE, n_agents=a.agents, n_options=a.options, n_steps=a.steps,
                       seed=a.seeds[0], llm_kind="openai" if a.llm else "scripted",
                       use_persona=a.persona, async_update=a.async_update,
                       flip_step=a.flip_step, flip_count=a.flip_count, model=a.model)

    mode = "LLM" if a.llm else "rule"
    pm = a.persona_mode or ("persona" if a.persona else "noPersona")
    tag = a.tag or (f"{mode}_{pm}_{a.options}opt"
                    f"{'_async' if a.async_update else ''}"
                    f"{'_flip'+str(a.flip_step) if a.flip_step is not None else ''}")
    print(f"\n{tag}: {mode} mode, {a.agents} agents, {a.options} options, {a.steps} steps, "
          f"seeds {a.seeds}, persona={a.persona}\n")

    runs = []
    for s in a.seeds:
        r = one_run(s, llm=a.llm, model=a.model, **cfg)
        r["summary"] = settle(r["trajectory"])
        runs.append(r)
        su = r["summary"]
        print(f"  seed {s:>3}: final_churn={su['final_churn']:.3f}  "
              f"align={su['final_alignment']:.3f}  H={su['final_entropy']:.4f}  "
              f"calls={su['llm_calls']}  fb={su['fallbacks']}  ({r['elapsed_s']}s)")

    def agg(k):
        xs = [r["summary"][k] for r in runs]
        return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0)

    print("\n  across seeds (mean ± sd):")
    for k in ("final_churn", "final_alignment", "final_entropy"):
        m, sd = agg(k)
        print(f"    {k:18} {m:.4f} ± {sd:.4f}")
    tot_calls = sum(r["summary"]["llm_calls"] for r in runs)
    tot_fb = sum(r["summary"]["fallbacks"] for r in runs)
    print(f"    LLM calls total    {tot_calls}   fallbacks {tot_fb}   "
          f"parse-fail "
          f"{max((r['summary']['max_parse_fail'] for r in runs), default=0.0):.3f}")

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    path = out / f"{tag}.json"
    path.write_text(json.dumps({"tag": tag, "mode": mode, "config": cfg,
                                "model": a.model, "runs": runs}, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    print(f"\n  written {path.relative_to(REPO)}")
    return runs


if __name__ == "__main__":
    main()
