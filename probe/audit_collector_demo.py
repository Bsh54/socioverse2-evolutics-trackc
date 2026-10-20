"""Drop-in reusability demo for the validity-audit collector (socioverse/diagnostics.py).

Enables "audit.diagnostics" on a SHIPPED study it was never written for (opinion_diffusion),
wrapping that study's own collector, and shows the audit columns (dx_*) land in the metrics table
alongside the study's own metrics — no change to the study's code. Scripted, no key, free.

Run from the SocioVerse2 checkout:
    python ../probe/audit_collector_demo.py
"""
from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(".").resolve()))


def main():
    import duckdb
    import socioverse.diagnostics  # noqa: F401  (registers "audit.diagnostics")
    import studies.opinion_diffusion as od  # noqa: F401  (registers opinion.* incl. its collector)
    from socioverse.engine import build_simulator

    study, env, pop, sim = od.model.make_opinion_bundles(n_agents=12, n_steps=5, seed=42)
    own_ref = sim.collector_ref
    # turn the audit on WITHOUT touching the study: wrap its own collector
    sim.collector_ref = "audit.diagnostics"
    sim.collector_args = {"inner_ref": own_ref, "inner_args": {}}

    db = pathlib.Path(tempfile.mkdtemp(prefix="auditdemo_")) / "study.duckdb"
    build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim, store_path=db).run()

    con = duckdb.connect(str(db), read_only=True)
    cols = [c[1] for c in con.sql("PRAGMA table_info('metrics')").fetchall()]
    own = [c for c in cols if not c.startswith("dx_") and c != "step"]
    dx = [c for c in cols if c.startswith("dx_")]
    print(f"study under audit : opinion_diffusion (collector '{own_ref}', unmodified)")
    print(f"study's own metrics kept : {own}")
    print(f"audit columns added      : {dx}\n")
    rows = con.sql("SELECT step, dx_n_actions, dx_llm_share, dx_nonreal_share, dx_entropy, "
                   "dx_top_share, dx_churn FROM metrics ORDER BY step").fetchall()
    con.close()
    print("step | n_act | llm_sh | nonreal | entropy | top_sh | churn")
    for r in rows:
        print(f"  {r[0]:>2} | {r[1]:>5} | {r[2]:>6} | {r[3]:>7} | {r[4]:>7} | {r[5]:>6} | {r[6]:>5}")

    assert dx, "audit columns must appear"
    assert own, "the study's own metrics must be preserved"
    print("\nself-check passed: the audit collector ran on a study it was not written for, kept that "
          "study's metrics, and added the dx_* validity diagnostics — one line of config, no code change.")


if __name__ == "__main__":
    main()
