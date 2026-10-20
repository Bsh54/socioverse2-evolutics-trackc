"""Breadth: run the reusable validity-audit collector across the platform's OWN shipped validation
studies (rule mode, no key, free) — EVOLUTICS, Track C, lever 3.

Shows the collector is genuinely study-agnostic (one line of config per study, no study code change)
and reports each study's final-step validity diagnostics, so the audit can confirm health or surface
a problem on SocioVerse2's own showcase cases. Run from the SocioVerse2 checkout:
    python ../probe/audit_breadth.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(".").resolve()))


def _build(name):
    """Return (study, env, pop, sim, own_collector_ref) for a shipped study, rule/scripted mode."""
    if name == "opinion_diffusion":
        import studies.opinion_diffusion as m
        s, e, p, sim = m.model.make_opinion_bundles(n_agents=20, n_steps=6, seed=42)
    elif name == "campus_dining_choice":
        import studies.campus_dining_choice as m
        s, e, p, sim = m.model.make_campus_dining_choice_bundles(n_agents=60, n_steps=5, seed=42)
    elif name == "germany_auto_market":
        import studies.germany_auto_market as m
        s, e, p, sim = m.model.make_germany_auto_market_bundles(
            seed=42, llm_kind="scripted", model=None, n_agents=300, n_steps=6)
    else:
        raise ValueError(name)
    return s, e, p, sim, sim.collector_ref


def main():
    import duckdb
    import socioverse.diagnostics  # noqa: F401  (registers "audit.diagnostics")
    from socioverse.engine import build_simulator

    studies = ["opinion_diffusion", "campus_dining_choice", "germany_auto_market"]
    print("validity audit across the platform's own validation studies (rule mode, free)\n")
    print(f"{'study':24} {'n_act':>5} {'llm_sh':>6} {'nonreal':>7} {'entropy':>7} "
          f"{'top_sh':>6} {'churn':>6}  verdict")
    print("-" * 86)
    for name in studies:
        s, e, p, sim, own = _build(name)
        sim.collector_ref = "audit.diagnostics"
        sim.collector_args = {"inner_ref": own, "inner_args": {}}
        db = pathlib.Path(tempfile.mkdtemp(prefix=f"br_{name}_")) / "study.duckdb"
        build_simulator(env_bundle=e, pop_bundle=p, sim_config=sim, store_path=db).run()
        con = duckdb.connect(str(db), read_only=True)
        r = con.sql("SELECT dx_n_actions, dx_llm_share, dx_nonreal_share, dx_entropy, dx_top_share,"
                    " dx_churn FROM metrics ORDER BY step DESC LIMIT 1").fetchone()
        con.close()
        nonreal, top_share = r[2], r[4]
        verdict = "healthy" if (nonreal == 0.0 and top_share < 0.99) else \
                  ("SILENT FALLBACK" if nonreal > 0 else "COLLAPSED")
        print(f"{name:24} {r[0]:>5} {r[1]:>6} {r[2]:>7} {r[3]:>7} {r[4]:>6} {r[5]:>6}  {verdict}")
    print("\nthe same collector, enabled with one config line, ran unchanged across three of the "
          "platform's seven validation studies and reported each one's provenance / dispersion / "
          "liveness diagnostics.")


if __name__ == "__main__":
    main()
