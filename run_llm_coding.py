#!/usr/bin/env python3
"""
LLM-assisted validation coder for the ALTAI/CDA component dataset.
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

MODEL_ID = "claude-sonnet-4-5-20250929"   
TEMPERATURE = 0.0
MAX_TOKENS = 500
N_RUNS_DEFAULT = 1   
CONSISTENCY_CHECK_N_DEFAULT = 100  
DEV_FRACTION = 0.20
RANDOM_SEED = 20260825  

HERE = Path(__file__).parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)


def load_codebook():
    with open(HERE / "codebook.json", "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_codebook_for_prompt(codebook):
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
        system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
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
                     help="Only process the first N of the frozen dev set (see --dev-run-size), "
                          "for a cheap sanity check before spending on the full corpus.")
    ap.add_argument("--dev-run-size", type=int, default=20,
                     help="How many dev-set components --dev-run processes. Pass the full dev "
                          "set size (see the 'Dev set: N components' line from --split-only) to "
                          "run the whole dev set after revising the codebook, before freezing it "
                          "and moving to --held-out.")
    ap.add_argument("--held-out", action="store_true",
                     help="Process the frozen held-out set (everything NOT in the dev set). Use "
                          "this only after the codebook is finalised from dev-set results -- "
                          "editing the codebook after seeing held-out results invalidates the "
                          "held-out/dev split.")
    ap.add_argument("--split-only", action="store_true",
                     help="Only produce the frozen dev/held-out split, make no API calls.")
    ap.add_argument("--n-runs", type=int, default=None,
                     help="Independent runs per component. Default: 3 for --dev-run (cheap, "
                          "establishes whether self-consistency holds), 1 for a full run "
                          "(cheaper; use --sample-n with --n-runs 3 separately to report a "
                          "self-consistency statistic on a subsample instead of tripling the "
                          "whole corpus).")
    ap.add_argument("--sample-n", type=int, default=None,
                     help="Process only a random N-component sample instead of everything. "
                          "Combine with --n-runs 3 and --out-suffix to produce a standalone "
                          "self-consistency check without re-running/re-paying for the full corpus.")
    ap.add_argument("--out-suffix", default="",
                     help="Suffix appended to output filenames, so a --sample-n side run "
                          "doesn't overwrite your main coded_results.csv.")
    args = ap.parse_args()

    n_runs = args.n_runs if args.n_runs is not None else (3 if args.dev_run else N_RUNS_DEFAULT)

    print(f"anthropic SDK version: {getattr(anthropic, '__version__', 'unknown')} "
          f"(this gets recorded in run_manifest.json -- keep it for your methods section)")

    codebook = load_codebook()
    system_prompt = flatten_codebook_for_prompt(codebook)

    with open(args.input_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for i, r in enumerate(rows):
        r["_row_id"] = i

    rng = random.Random(RANDOM_SEED)
    shuffled = rows[:]
    rng.shuffle(shuffled)  

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
        targets = dev_set[:args.dev_run_size]
        print(f"--dev-run: processing {len(targets)}/{len(dev_set)} dev-set components.")
    elif args.held_out:
        targets = held_out_set
        print(f"--held-out: processing all {len(targets)} held-out components. "
              f"Make sure the codebook is FINAL before this -- editing it after seeing "
              f"these results invalidates the dev/held-out split.")
    elif args.sample_n:
        targets = shuffled[:args.sample_n]
        print(f"--sample-n {args.sample_n}: processing a random subsample.")

    n_calls = len(targets) * n_runs
    approx_system_tokens = len(system_prompt) // 4
    approx_output_tokens = 250
    print(f"\nAbout to make ~{n_calls} API calls ({len(targets)} components x {n_runs} run(s)) "
          f"against model '{MODEL_ID}'.")
    print(f"Each call sends ~{approx_system_tokens} codebook tokens (cached after the first "
          f"call) plus a short component + ~{approx_output_tokens} output tokens.")
    print("Check https://www.anthropic.com/pricing for this model's current per-token rate "
          "before a large run, and consider --sample-n for a smaller cost-check first.\n")

    run_manifest = {
        "model_id": MODEL_ID,
        "anthropic_sdk_version": getattr(anthropic, "__version__", "unknown"),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "n_runs": n_runs,
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

    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    raw_log_path = OUT_DIR / f"raw_runs{suffix}.jsonl"
    results = []

    with open(raw_log_path, "a", encoding="utf-8") as raw_log:
        for idx, row in enumerate(targets):
            component_runs = []
            for run_i in range(n_runs):
                try:
                    parsed, raw = code_one_component(client, system_prompt, row["component_text"], run_i)
                except TypeError as e:
                    if "temperature" in str(e):
                        print("\nFATAL: your installed anthropic SDK's messages.create() does not "
                              "accept this call shape. Run `pip show anthropic` to check the version, "
                              "then `pip install --upgrade anthropic` and try again.", file=sys.stderr)
                        sys.exit(1)
                    print(f"  ERROR on row {row['_row_id']} run {run_i}: {e}", file=sys.stderr)
                    time.sleep(2)
                    continue
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

    results_path = OUT_DIR / f"coded_results{suffix}.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    run_manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path = OUT_DIR / f"run_manifest{suffix}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    n_agree = sum(1 for r in results if r["human_llm_agree"])
    n_scored = sum(1 for r in results if r["human_llm_agree"] is not None)
    n_self_consistent = sum(1 for r in results if r["llm_self_consistent"])
    print(f"\nDone. {len(results)} components processed.")
    print(f"Raw agreement: {n_agree}/{n_scored} = {n_agree/max(n_scored,1):.1%}")
    if n_runs > 1:
        print(f"Self-consistency across {n_runs} runs: {n_self_consistent}/{len(results)} = {n_self_consistent/max(len(results),1):.1%}")
    print(f"\nWrote: {results_path.name}, {raw_log_path.name}, {manifest_path.name} (in output/)")
    print(f"Next: python compute_reliability.py --input output/{results_path.name}")


if __name__ == "__main__":
    main()
