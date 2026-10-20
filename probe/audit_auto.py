"""Robustness audit of the shipped germany_auto_market conclusion.

Vary only knobs the field assumes are harmless (seed; later model / llm_kind) and measure
how much the study's CONCLUSION moves: overall fidelity to KBA truth (MAE across 13 brands
x 29 months) and the paper's qualitative claims (VW dominant, Tesla declines, Skoda rises,
Opel weakens, BYD small). If the verdict flips across seeds/models, a single shipped run is
not a stable basis for the claim.

Run from the SocioVerse2 checkout:
    python ../probe/audit_auto.py --seeds 42 0 1 2 3 --llm-kind scripted
    python ../probe/audit_auto.py --seeds 0 1 2 --llm-kind openai --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys
import tempfile
import time

BRANDS = ["vw", "mercedes", "bmw", "audi", "skoda", "seat_cupra", "opel",
          "toyota", "hyundai", "renault", "tesla", "byd", "other"]


def truth():
    p = pathlib.Path("studies/germany_auto_market/brand_truth_series.json")
    return json.loads(p.read_text(encoding="utf-8"))["series"]


def run_once(seed, llm_kind, model, n_agents=1500, n_steps=29):
    import studies.germany_auto_market as g
    from socioverse.engine import build_simulator
    import duckdb
    study, env, pop, sim = g.model.make_germany_auto_market_bundles(
        seed=seed, llm_kind=llm_kind, model=model, n_agents=n_agents, n_steps=n_steps)
    db = pathlib.Path(tempfile.mkdtemp(prefix="auto_")) / "s.duckdb"
    t0 = time.time()
    build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim, store_path=db).run()
    con = duckdb.connect(str(db), read_only=True)
    cols = ", ".join(f"share_{b}" for b in BRANDS)
    rows = con.sql(f"SELECT step, {cols} FROM metrics ORDER BY step").fetchall()
    con.close()
    # keep the last 29 rows to align with the 29 truth months
    data = [dict(zip(BRANDS, r[1:])) for r in rows][-29:]
    return data, time.time() - t0


def mae_to_truth(sim_series, tru):
    n = min(len(sim_series), len(tru))
    errs = [abs(sim_series[i][b] - tru[i][b]) for i in range(n) for b in BRANDS]
    return sum(errs) / len(errs)


def verdicts(sim_series):
    first, last = sim_series[0], sim_series[-1]
    nonother = [b for b in BRANDS if b != "other"]
    dominant = max(nonother, key=lambda b: last[b])
    return {
        "mae": None,  # filled by caller
        "VW_dominant": dominant == "vw",
        "dominant_brand": dominant,
        "tesla_first": round(first["tesla"], 4),
        "tesla_last": round(last["tesla"], 4),
        "tesla_declines": last["tesla"] < first["tesla"],
        "skoda_rises": last["skoda"] > first["skoda"],
        "opel_weakens": last["opel"] < first["opel"],
        "byd_last": round(last["byd"], 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap.add_argument("--llm-kind", default="scripted")
    ap.add_argument("--model", default=None)
    ap.add_argument("--agents", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=29)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    if not pathlib.Path("studies").is_dir():
        sys.exit("run from the SocioVerse2 checkout")
    here = str(pathlib.Path(".").resolve())
    if here not in sys.path:
        sys.path.insert(0, here)

    tru = truth()
    print(f"germany_auto_market audit — llm_kind={a.llm_kind} model={a.model} "
          f"seeds={a.seeds}\n")
    rows = []
    for s in a.seeds:
        series, secs = run_once(s, a.llm_kind, a.model, a.agents, a.steps)
        v = verdicts(series)
        v["mae"] = mae_to_truth(series, tru)
        rows.append((s, v, secs))
        print(f"seed {s:>3} ({secs:.0f}s): MAE {v['mae']:.4f} | dominant={v['dominant_brand']} "
              f"| tesla {v['tesla_first']}->{v['tesla_last']} ({'down' if v['tesla_declines'] else 'UP'}) "
              f"| skoda {'up' if v['skoda_rises'] else 'DOWN'} "
              f"| opel {'down' if v['opel_weakens'] else 'UP'} | byd_last {v['byd_last']}")

    print("\nstability across seeds:")
    maes = [v["mae"] for _, v, _ in rows]
    print(f"  MAE: mean {st.mean(maes):.4f}  sd {st.pstdev(maes):.4f}  "
          f"min {min(maes):.4f}  max {max(maes):.4f}")
    for claim in ("VW_dominant", "tesla_declines", "skoda_rises", "opel_weakens"):
        vals = [v[claim] for _, v, _ in rows]
        agree = sum(vals)
        flag = "" if agree in (0, len(vals)) else "  <-- FLIPS across seeds"
        print(f"  {claim:16} holds in {agree}/{len(vals)} seeds{flag}")
    tl = [v["tesla_last"] for _, v, _ in rows]
    bl = [v["byd_last"] for _, v, _ in rows]
    print(f"  tesla_last range [{min(tl):.4f}, {max(tl):.4f}]   "
          f"byd_last range [{min(bl):.4f}, {max(bl):.4f}]")

    out = pathlib.Path(f"audit_auto_{a.tag or a.llm_kind}.json")
    out.write_text(json.dumps([{"seed": s, **v} for s, v, _ in rows], indent=2), encoding="utf-8")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
