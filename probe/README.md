# `probe/` — the audit experiments

The experiments behind the EVOLUTICS Track C contribution (a validity + cost audit of SocioVerse2).
Run each from the repository root (`SocioVerse2/`) with `PYTHONUTF8=1` set. The free/deterministic
ones are wired into `../reproduce.py`; the two that need an LLM key are marked.

| script | what it shows | cost |
|---|---|---|
| `reproduce_branch_bug.py` | the branch-replay silent-degradation defect and the fix, offline, via the repo's own functions | free |
| `detection_eval.py` | detector evaluation: the field-standard agreement metric misses ~78% of injected faults; the combined suite misses ~1%; 0 false positives | free |
| `cost_of_silent_degradation.py` | the resource-cost face: a correct branch's replay saving vs the budget the degraded branch silently re-pays (`saving == leak == N × resume_from`) | free |
| `audit_collector_demo.py` | the reusable `audit.diagnostics` collector enabled on a study it was not written for, keeping that study's metrics | free |
| `audit_breadth.py` | the same collector run unchanged across three shipped validation studies | free |
| `audit_auto.py` | robustness audit of the shipped `germany_auto_market` conclusion across seeds (positive control) | free (rule mode) |
| `userpool_probe.py` | a working client for the SocioVerse user-pool MCP (handshake + `tools/call`) | reads free |
| `real_persona_exp.py` | does a real demographic persona (no named option) suppress neighbour response? | **needs `SV_LLM_*`** |

Reproduce everything free in one command from the repository root: `python reproduce.py`.
