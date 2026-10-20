"""Does a REAL demographic persona (no named option) suppress neighbour response?

Closes the bound left by the ring study: our persona named an option; real silicon-sampling
personas carry demographics. Here 20 real personas pulled from the SocioVerse user-pool MCP sit
on a ring and answer a real binary attitude question, updating each step under neighbour pressure.

Three arms on the SAME agents / question:
  bare         - prompt has the question + neighbours only (no persona)      -> expect responsive
  demographic  - real profile + question + neighbours (NO option named)      -> the test
  named        - "you have a slight preference for A/B" (control)            -> known to lock

Metric: churn (fraction of agents who changed answer since the previous step). If demographic
churn collapses to ~0 like named, a real identity overrides neighbours. If it stays high like
bare, demographics alone do not — and we say so.

Agent decisions run on the RodiumAI gateway (SV_LLM_* in .env). Reads are from a saved persona
file (free MCP pull). Run from the SocioVerse2 checkout:
    python ../probe/real_persona_exp.py --model openai/gpt-6-luna --steps 6 --arms bare demographic named
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import json
import os
import pathlib
import random
import re
import sys
import time

ENVF = pathlib.Path("SocioVerse2/.env")
if not ENVF.exists():
    ENVF = pathlib.Path(".env")
PERS = pathlib.Path("SocioVerse2/studies/persona_override/real_personas.json")
if not PERS.exists():
    PERS = pathlib.Path("studies/persona_override/real_personas.json")


def env(name, default=None):
    if ENVF.exists():
        for line in ENVF.read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip()
    return os.environ.get(name, default)


QUESTION = ("A government proposal would raise taxes to expand public healthcare coverage. "
            "Choose your position for the next round.\n"
            'Options: A = support, B = oppose.')
CHOICE_RE = re.compile(r'"choice"\s*:\s*"?([AB])"?', re.I)


def client():
    from openai import OpenAI
    return OpenAI(api_key=env("SV_LLM_API_KEY"), base_url=env("SV_LLM_BASE_URL"))


def call(cli, model, prompt, retries=4):
    m = model.lower()
    newer = any(x in m for x in ("gpt-5", "gpt-6"))
    kw = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if newer:
        kw["max_completion_tokens"] = 400
    else:
        kw["max_tokens"] = 60
        kw["temperature"] = 0.5
    for a in range(retries):
        try:
            r = cli.chat.completions.create(**kw)
            return r.choices[0].message.content or "", False
        except Exception:
            if a == retries - 1:
                return None, True
            time.sleep(0.6 * (a + 1))
    return None, True


def prompt_for(arm, profile, own, neigh):
    L = [QUESTION,
         f"- Your current position: {own}",
         f"- Your two neighbours currently chose: {neigh}"]
    if arm == "demographic":
        L.insert(0, f"You are this person:\n{profile}\n")
    elif arm == "named":
        L.append("- You have a slight preference for option A.")
    L.append('Output one line of JSON only: {"choice": "A" or "B", "reason": "<=8 words"}')
    return "\n".join(L)


def parse(txt):
    if not txt:
        return None
    m = CHOICE_RE.search(txt) or re.search(r'\b([AB])\b', txt)
    return m.group(1).upper() if m else None


def run_arm(arm, personas, model, steps, seed, workers=10):
    rng = random.Random(seed)
    n = len(personas)
    choice = {i: rng.choice(["A", "B"]) for i in range(n)}
    profiles = {i: personas[i]["summary"] for i in range(n)}
    cli = client()
    per_step = []
    err_tot = 0
    for t in range(1, steps + 1):
        neigh = {i: [choice[(i - 1) % n], choice[(i + 1) % n]] for i in range(n)}
        prompts = {i: prompt_for(arm, profiles[i], choice[i], neigh[i]) for i in range(n)}
        new = dict(choice)
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(call, cli, model, prompts[i]): i for i in range(n)}
            for f in cf.as_completed(futs):
                i = futs[f]
                txt, errored = f.result()
                c = parse(txt)
                if errored or c is None:
                    err_tot += 1
                else:
                    new[i] = c
        churn = sum(1 for i in range(n) if new[i] != choice[i]) / n
        shareA = sum(1 for i in range(n) if new[i] == "A") / n
        per_step.append((t, churn, shareA))
        choice = new
    return per_step, err_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=env("SV_LLM_MODEL", "openai/gpt-4o-mini"))
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", nargs="+", default=["bare", "demographic", "named"])
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()

    personas = json.loads(PERS.read_text(encoding="utf-8"))[:a.n]
    print(f"model={a.model}  personas={len(personas)}  steps={a.steps}  seed={a.seed}\n")
    results = {}
    for arm in a.arms:
        t0 = time.time()
        per_step, errs = run_arm(arm, personas, a.model, a.steps, a.seed)
        results[arm] = per_step
        churns = [c for _, c, _ in per_step]
        print(f"=== {arm:12} ({time.time()-t0:.0f}s, errors {errs}) ===")
        for t, c, sA in per_step:
            print(f"  step {t}: churn {c:.2f}  shareA {sA:.2f}")
        tail = churns[len(churns)//2:]
        print(f"  mean churn (2nd half): {sum(tail)/len(tail):.3f}\n")

    print("SUMMARY — mean churn over 2nd half (low = agents stopped responding to neighbours):")
    for arm in a.arms:
        ch = [c for _, c, _ in results[arm]]
        tail = ch[len(ch)//2:]
        print(f"  {arm:12} {sum(tail)/len(tail):.3f}")
    out = pathlib.Path("real_persona_exp_result.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
