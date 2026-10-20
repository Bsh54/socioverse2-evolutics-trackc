"""fork_study must preserve non-ASCII artifact content — EVOLUTICS, Track C (second platform fix).

`fork_study` rewrites each copied artifact's embedded `study_id`; that path reads the JSON and
writes it back. The read used a bare `read_text()`, whose codec is the process locale (cp1252 on a
default Windows install), so every fork silently corrupted non-ASCII content — and the shipped
reference studies carry Chinese text and typographic dashes. The write side already forced UTF-8;
the fix aligns the read. This pins the round-trip contract: a fork preserves non-ASCII exactly while
still rewriting the id.
"""
from __future__ import annotations

import json

from skills.sv_workspace import fork_study, save_study_yaml, scaffold, study_paths
from socioverse.schemas import StudySpec

NON_ASCII = "政策广播 — le garçon au café"      # Chinese + em dash (U+2014) + accented Latin


def test_fork_preserves_non_ascii_and_rewrites_study_id(tmp_path):
    scaffold(tmp_path, "src")
    save_study_yaml(study_paths(tmp_path / "src")["study"],
                    StudySpec(study_id="src", domain="opinion-dynamics"))
    env = study_paths(tmp_path / "src")["environment"]
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text(json.dumps({"study_id": "src", "note": NON_ASCII}, ensure_ascii=False),
                   encoding="utf-8")

    dst = fork_study(tmp_path, "src", "dst")

    obj = json.loads(study_paths(dst)["environment"].read_text(encoding="utf-8"))
    assert obj["study_id"] == "dst"                 # the read->rewrite path ran
    assert obj["note"] == NON_ASCII                 # and non-ASCII survived byte-for-byte
    # the em dash must still be one codepoint, not the cp1252 mojibake expansion
    assert "—" in obj["note"] and "—".encode("cp1252", "ignore") != b""
