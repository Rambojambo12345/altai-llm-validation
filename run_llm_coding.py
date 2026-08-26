#!/usr/bin/env python3
"""
LLM-assisted validation coder for the ALTAI/CDA component dataset.

WHAT THIS DOES
---------------
For every coded component in master_components_final.csv, this script asks
an LLM (via a fresh, isolated API call -- NOT a chat conversation) to assign
a sub-principle code using ONLY the codebook, blind to your original code.
It runs each component through the model three independent times, takes the
modal (most common) label, and writes out everything needed to compute
human-LLM agreement statistics: Krippendorff's alpha, Cohen's kappa, per-
class precision/recall/F1, a confusion matrix, and self-consistency rates.

WHY IT MUST RUN OUTSIDE THIS CHAT
----------------------------------
Any conversation that has seen the manuscript, its results, or its
hypotheses is a contaminated coder -- it already knows what the "right"
answer is supposed to look like, which defeats the purpose of an
independent validation. This script makes fresh, stateless API calls: each
call contains nothing but the codebook and one component's text. Nothing
about your paper, your findings, or this conversation is ever sent to the
model being validated.

HOW TO RUN IT
-------------
1. Install dependencies:
     pip install anthropic pandas

2. Get an API key from https://console.anthropic.com/ (Settings > API Keys).
   Do NOT paste it into a chat with any AI assistant. Set it as an
   environment variable in your own terminal:

     export ANTHROPIC_API_KEY="sk-ant-..."         (macOS/Linux)
     setx ANTHROPIC_API_KEY "sk-ant-..."            (Windows, new terminal after)

3. Put this script, codebook.json, and master_components_final.csv in the
   same folder, then run:

     python run_llm_coding.py --dev-run          # cheap sanity check, ~20 items
     python run_llm_coding.py                    # full production run

The full run makes roughly 3 x (number of components) API calls. At current
Claude Haiku/Sonnet pricing this is a few dollars for ~2,200 components;
using a larger model costs more but is more defensible for a validation
exercise (see the protocol notes sent earlier -- prefer the strongest model
you can afford for this).

OUTPUT FILES (all written to ./output/)
----------------------------------------
  raw_runs.jsonl        - every single API call: full prompt params, raw
                           response, timestamp. This is your reproducibility
                           record -- keep it, deposit it with the paper.
  coded_results.csv      - one row per component: original code, all 3 LLM
                           run codes, modal LLM code, agreement flag,
                           self-consistency flag, confidence, verbatim span.
  dev_set.csv / held_out_set.csv - the frozen 20/80 split (see protocol).
  run_manifest.json      - exact model id, temperature, seed, timestamps,
                           row counts -- paste this into your methods section.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Missing dependency. Run: pip install anthropic pandas", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# CONFIG -- pin the exact model snapshot. Do not use a "latest" alias for the
# production run: your methods section needs to name an exact, re-runnable
# model. Check https://docs.claude.com/en/docs/about-claude/models for the
# current list of dated snapshot IDs and swap this if it has been retired.
# ---------------------------------------------------------------------------
MODEL_ID = "claude-opus-4-5-20251101"   # <-- verify this is still a valid snapshot id before running
TEMPERATURE = 0.0
MAX_TOKENS = 500
N_RUNS = 3
DEV_FRACTION = 0.20
RANDOM_SEED = 20260825  # fixed and reported, not re-rolled between runs

HERE = Path(__file__).parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)


def load_codebook():
    with open(HERE / "codebook.json", "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_codebook_for_prompt(codebook):
    """Render the codebook as the text block the model sees. This text is
    the ENTIRE substantive content of every prompt -- it never changes
    between components, so it's easy to audit for leakage."""
    lines = [
        "You are applying a fixed coding scheme to short excerpts from AI",
        "governance / auditing documents. Read the CODEBOOK below, then code",
        "ONE component according to the DECISION RULE. Do not use any",
        "knowledge about specific companies, papers, or research findings --",
        "code strictly from the text given and the codebook definitions.",
        "",
        "DECISION RULE:",
        codebook["decision_rule"],
        "",
        "CODEBOOK:",
    ]
    for principle in codebook["principles"]:
        lines.append(f"\n[{principle['key_principle']}]")
        for sp in principle["sub_principles"]:
            lines.append(f"  {sp['id']}. {sp['name']}: {sp['definition']}")
    u = codebook["unmapped"]
    lines.append(f"\n{u['id']}. {u['name']}: {u['definition']}")
    return "\n".join(lines)


TOOL_SCHEMA = {
    "name": "code_component",
    "description": "Assign a single codebook sub-principle id to the given component.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning_principle_level": {
                "type": "string",
                "description": "One sentence: which of the 8 key principles (or none) this component's primary requirement falls under, and why."
            },
            "code_id": {
                "type": "integer",
                "description": "The single best-fitting sub-principle id from the codebook (0-27). Use 0 if nothing fits."
            },
            "verbatim_span": {
                "type": "string",
                "description": "The exact substring of the component text that most directly supports this code. Must be copied verbatim, not paraphrased."
            },
            "justification": {
                "type": "string",
                "description": "One sentence explaining why this code, not an adjacent/competing one."
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Your confidence that this is the single best code, considering plausible alternative codes."
            }
        },
        "required": ["reasoning_principle_level", "code_id", "verbatim_span", "justification", "confidence"]
    }
}


def code_one_component(client, system_prompt, component_text, run_index):
    """Make one fresh, isolated API call. No conversation history, no prior
    component's context -- this is critical for independence between rows."""
    user_msg = (
        "Component to code (assign exactly one code_id from the codebook):\n\n"
        f'"{component_text}"'
    )
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        tools=[TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "code_component"},
        messages=[{"role": "user", "content": user_msg}],
        extra_body={"temperature": TEMPERATURE},
    )
    tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
    if tool_use is None:
        return None, resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
    return tool_use.input, resp.model_dump() if hasattr(resp, "model_dump") else str(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", default=str(HERE / "master_components_final.csv"))
    ap.add_argument("--dev-run", action="store_true",
                     help="Only process ~20 randomly sampled components, for a cheap sanity check.")
    ap.add_argument("--split-only", action="store_true",
                     help="Only produce the frozen dev/held-out split, make no API calls.")
    args = ap.parse_args()

    codebook = load_codebook()
    system_prompt = flatten_codebook_for_prompt(codebook)

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Blind: strip anything that could leak the original code or reveal
    # sector/provenance in a way the human coder didn't see at coding time.
    # We KEEP original_code/sector internally for later scoring, but they are
    # NEVER placed into a prompt sent to the model.
    for i, r in enumerate(rows):
        r["_row_id"] = i

    rng = random.Random(RANDOM_SEED)
    shuffled = rows[:]
    rng.shuffle(shuffled)  # randomised order -- avoids same-framework context effects

    n_dev = int(len(shuffled) * DEV_FRACTION)
    dev_set = shuffled[:n_dev]
    held_out_set = shuffled[n_dev:]

    with open(OUT_DIR / "dev_set.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(dev_set)
    with open(OUT_DIR / "held_out_set.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(held_out_set)

    print(f"Dev set: {len(dev_set)} components. Held-out set: {len(held_out_set)} components.")
    print("Wrote output/dev_set.csv and output/held_out_set.csv")

    if args.split_only:
        return

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("\nERROR: ANTHROPIC_API_KEY is not set in this environment.", file=sys.stderr)
        print("Set it in your own terminal before running -- see the header of this file.", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic()

    targets = shuffled
    if args.dev_run:
        targets = shuffled[:20]
        print(f"--dev-run: processing only {len(targets)} components as a sanity check.")

    run_manifest = {
        "model_id": MODEL_ID,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "n_runs": N_RUNS,
        "random_seed": RANDOM_SEED,
        "dev_fraction": DEV_FRACTION,
        "n_components_total": len(rows),
        "n_dev": len(dev_set),
        "n_held_out": len(held_out_set),
        "n_processed_this_execution": len(targets),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "input_csv": args.input_csv,
    }

    raw_log_path = OUT_DIR / "raw_runs.jsonl"
    results = []

    with open(raw_log_path, "a", encoding="utf-8") as raw_log:
        for idx, row in enumerate(targets):
            component_runs = []
            for run_i in range(N_RUNS):
                try:
                    parsed, raw = code_one_component(client, system_prompt, row["component_text"], run_i)
                except Exception as e:
                    print(f"  ERROR on row {row['_row_id']} run {run_i}: {e}", file=sys.stderr)
                    time.sleep(2)
                    continue
                log_entry = {
                    "row_id": row["_row_id"],
                    "framework": row["framework"],
                    "run_index": run_i,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": MODEL_ID,
                    "temperature": TEMPERATURE,
                    "parsed_output": parsed,
                    "raw_response": raw,
                }
                raw_log.write(json.dumps(log_entry, default=str) + "\n")
                raw_log.flush()
                if parsed is not None:
                    component_runs.append(parsed)

            codes = [r["code_id"] for r in component_runs if "code_id" in r]
            if codes:
                modal_code, modal_count = Counter(codes).most_common(1)[0]
                self_consistent = modal_count == len(codes)
            else:
                modal_code, modal_count, self_consistent = None, 0, False

            best_run = component_runs[0] if component_runs else {}
            results.append({
                "row_id": row["_row_id"],
                "framework": row["framework"],
                "sector": row["sector"],
                "identifier": row["identifier"],
                "component_text": row["component_text"],
                "original_code_id": row["code_id"],
                "original_code_name": row["code_name"],
                "in_dev_set": row in dev_set,
                "llm_run_codes": json.dumps(codes),
                "llm_modal_code_id": modal_code,
                "llm_self_consistent": self_consistent,
                "llm_confidence": best_run.get("confidence"),
                "llm_verbatim_span": best_run.get("verbatim_span"),
                "llm_justification": best_run.get("justification"),
                "human_llm_agree": (str(modal_code) == str(row["code_id"])) if modal_code is not None else None,
            })

            if (idx + 1) % 25 == 0:
                print(f"  processed {idx + 1}/{len(targets)}")

    with open(OUT_DIR / "coded_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    run_manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(OUT_DIR / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    n_agree = sum(1 for r in results if r["human_llm_agree"])
    n_scored = sum(1 for r in results if r["human_llm_agree"] is not None)
    n_self_consistent = sum(1 for r in results if r["llm_self_consistent"])
    print(f"\nDone. {len(results)} components processed.")
    print(f"Raw agreement: {n_agree}/{n_scored} = {n_agree/max(n_scored,1):.1%}")
    print(f"Self-consistency across 3 runs: {n_self_consistent}/{len(results)} = {n_self_consistent/max(len(results),1):.1%}")
    print(f"\nWrote: output/coded_results.csv, output/raw_runs.jsonl, output/run_manifest.json")
    print("Next: run compute_reliability.py on output/coded_results.csv")


if __name__ == "__main__":
    main()
