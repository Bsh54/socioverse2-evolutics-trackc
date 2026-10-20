"""Deterministic, offline demonstration of a silent branch-replay degradation in SocioVerse2
— and of the fix.

THE DEFECT (as shipped). A counterfactual BRANCH replays its parent run's steps 0..t* (no LLM)
so treatment and control are paired agent-by-agent. The engine replays only when
``sim_config.warm_start`` is set; ``sv-run`` sets it from ``skills.sv_workspace.resolve_warm_start``.
When the parent version's archived trajectory store (``versions/<source>/trajectory/study.duckdb``)
is ABSENT — reachable in normal use: ``versions/`` and ``*.duckdb`` are gitignored, so a branch
snapshotted before its parent ran, or a moved / re-cloned / pruned workspace, keeps the committed
artifacts but loses the parent's realized history — the as-shipped code did this:

    check_branch_invariant(...)  -> ok=True     (it only tested the snapshot DIRECTORY existed)
    resolve_warm_start(...)      -> None         SILENTLY
    sv-run warm_start            -> None         => the BRANCH cold-runs from step 0, UNPAIRED,
                                                    with no error, no warning, no audit record.

The metrics then look like a valid paired branch but the treatment arm never replayed the
control's shared history. (This is what corrupted our hosted ev-subsidy run.)

THE FIX (skills/sv_workspace.py, check_branch_invariant). The snapshot directory existing is not
enough: the invariant now also requires the parent's trajectory STORE (the thing actually
replayed) to exist and be non-empty. A missing / empty store is a branch-contract violation, so it
takes the framework's EXISTING downgrade-with-audit path: ``kind`` -> "version", warm-start
dropped, and ``downgraded_from_branch`` recorded with the reason. The degradation becomes loud and
audited instead of silent, exactly like every other broken branch contract.

Uses the repo's own functions (skills.sv_workspace + the engine), scripted mode, no API key.
Run from the SocioVerse2 checkout:
    python ../probe/reproduce_branch_bug.py
"""
from __future__ import annotations

import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(".").resolve()))


def build_and_run(study_dir, seed=42, n_steps=6):
    """Run the persona_override scripted model into study_dir/trajectory/study.duckdb."""
    import studies.persona_override as po
    from socioverse.engine import build_simulator
    from skills.sv_workspace import study_paths
    _study, env, pop, sim = po.model.make_bundles(
        study_id=study_dir.name, n_agents=20, n_options=2, n_steps=n_steps, seed=seed,
        llm_kind="scripted")
    p = study_paths(study_dir)
    p["duckdb"].parent.mkdir(parents=True, exist_ok=True)
    build_simulator(env_bundle=env, pop_bundle=pop, sim_config=sim,
                    store_path=p["duckdb"]).run()


def main():
    from skills.sv_workspace import (init_manifest, create_branch, resolve_warm_start,
                                     check_branch_invariant, version_trajectory_db, study_paths,
                                     version_entry)

    root = pathlib.Path("studies")
    # start from a clean copy of the scripted reference so we have a valid study layout
    src = root / "opinion_diffusion"
    dst = root / "probe_branchbug"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        "trajectory", "reports", "versions", "versions.json", "__pycache__"))

    # --- v1: a normal control run ---
    init_manifest(dst, note="control")
    build_and_run(dst, seed=42, n_steps=6)
    print("v1 (control) run written:", (study_paths(dst)["duckdb"]).exists())

    # --- create a branch at step 3 (the counterfactual treatment) ---
    create_branch(dst, note="treatment branch", fork_step=3)
    inv = check_branch_invariant(dst)
    ws = resolve_warm_start(dst)
    print("\nHAPPY PATH (parent snapshot + trajectory present):")
    print("  check_branch_invariant ok =", inv.get("ok", "?"))
    print("  resolve_warm_start       =", ws)
    src_db = version_trajectory_db(dst, (ws or {}).get("source_version", "v1"))
    print("  parent snapshot db exists =", src_db.exists())

    # --- the failure condition: the parent snapshot DIRECTORY survives, its TRAJECTORY is gone ---
    if src_db.exists():
        src_db.unlink()
    assert (dst / "versions" / "v1").is_dir(), "the snapshot dir must survive (only the db is gone)"
    inv2 = check_branch_invariant(dst)
    ws2 = resolve_warm_start(dst)
    entry = version_entry(dst)
    print("\nFAILURE CONDITION (parent snapshot dir present, trajectory store missing):")
    print("  AS SHIPPED, this cold-ran the branch SILENTLY (ok=True, warm_start=None, no audit).")
    print("  WITH THE FIX:")
    print("    check_branch_invariant ok =", inv2.get("ok", "?"), " <- now CAUGHT")
    print("    downgraded                =", inv2.get("downgraded"))
    print("    violation                 =", next((v for v in inv2.get("violations", [])
                                                    if "trajectory store" in v), None))
    print("    manifest kind now         =", entry.get("kind"),
          "(was 'branch')  audit recorded =", bool(entry.get("downgraded_from_branch")))
    print("    resolve_warm_start        =", ws2,
          " <- None for the RECORDED reason, not silently")

    # --- self-checks (assert the FIXED behaviour) ---
    assert ws is not None and ws.get("resume_from") == 2, \
        f"happy path should resolve a warm start at resume_from=2, got {ws}"
    assert not inv2.get("ok", True), \
        "FIX REGRESSED: a missing parent trajectory should now fail the invariant"
    assert inv2.get("downgraded") and entry.get("kind") == "version", \
        "FIX REGRESSED: the branch should be downgraded to a version"
    assert entry.get("downgraded_from_branch"), \
        "FIX REGRESSED: the downgrade reason must be recorded for the audit trail"
    assert ws2 is None, "a downgraded branch has no warm start to resolve"
    print("\nself-check passed: the branch resolves correctly WITH the parent trajectory; WITHOUT it "
          "the invariant now downgrades it to a version and records the reason, instead of cold-running "
          "an unpaired arm silently.")
    shutil.rmtree(dst, ignore_errors=True)


if __name__ == "__main__":
    main()
