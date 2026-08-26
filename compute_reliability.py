#!/usr/bin/env python3
"""
Compute human-LLM agreement statistics from output/coded_results.csv.

Run this AFTER run_llm_coding.py has produced output/coded_results.csv.

    pip install scikit-learn krippendorff pandas
    python compute_reliability.py

Writes output/reliability_report.json and output/confusion_matrix.csv, and
prints a summary you can paste into the methods/results section, split by
dev vs. held-out (report the held-out numbers as your primary result; the
dev numbers only show what the prompt was tuned on).
"""
import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score, precision_recall_fscore_support, confusion_matrix
)

try:
    import krippendorff
    HAVE_KRIPPENDORFF = True
except ImportError:
    HAVE_KRIPPENDORFF = False
    print("NOTE: `pip install krippendorff` to also get Krippendorff's alpha. "
          "Continuing with Cohen's kappa and percent agreement only.\n")

HERE = Path(__file__).parent
OUT_DIR = HERE / "output"


def compute_for_subset(df, label):
    df = df.dropna(subset=["llm_modal_code_id", "original_code_id"]).copy()
    df["original_code_id"] = df["original_code_id"].astype(str)
    df["llm_modal_code_id"] = df["llm_modal_code_id"].astype(str)

    n = len(df)
    if n == 0:
        return {"label": label, "n": 0}

    raw_agree = (df["original_code_id"] == df["llm_modal_code_id"]).mean()
    kappa = cohen_kappa_score(df["original_code_id"], df["llm_modal_code_id"])

    labels = sorted(set(df["original_code_id"]) | set(df["llm_modal_code_id"]), key=lambda x: (len(x), x))
    precision, recall, f1, support = precision_recall_fscore_support(
        df["original_code_id"], df["llm_modal_code_id"], labels=labels, zero_division=0
    )
    per_class = {
        lab: {"precision": round(float(p), 3), "recall": round(float(r), 3),
              "f1": round(float(f), 3), "support": int(s)}
        for lab, p, r, f, s in zip(labels, precision, recall, f1, support)
    }

    cm = confusion_matrix(df["original_code_id"], df["llm_modal_code_id"], labels=labels)
    cm_path = OUT_DIR / f"confusion_matrix_{label}.csv"
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["human_code\\llm_code"] + labels)
        for lab, row in zip(labels, cm):
            w.writerow([lab] + list(row))

    result = {
        "label": label,
        "n": n,
        "raw_percent_agreement": round(float(raw_agree), 4),
        "cohens_kappa": round(float(kappa), 4),
        "per_class_precision_recall_f1": per_class,
        "confusion_matrix_csv": str(cm_path.name),
    }

    if HAVE_KRIPPENDORFF:
        # reliability_data: 2 raters x n units, categorical/nominal
        rel_data = [
            df["original_code_id"].tolist(),
            df["llm_modal_code_id"].tolist(),
        ]
        # krippendorff needs numeric-coercible or consistent categorical codes
        value_domain = sorted(set(rel_data[0]) | set(rel_data[1]))
        vmap = {v: i for i, v in enumerate(value_domain)}
        numeric_data = [[vmap[x] for x in rater] for rater in rel_data]
        alpha = krippendorff.alpha(reliability_data=numeric_data, level_of_measurement="nominal")
        result["krippendorff_alpha"] = round(float(alpha), 4)

    # self-consistency
    if "llm_self_consistent" in df.columns:
        sc = df["llm_self_consistent"].astype(str).str.lower().eq("true")
        result["llm_self_consistency_rate"] = round(float(sc.mean()), 4)
        # accuracy conditional on self-consistency (the Pangakis et al. check)
        if sc.sum() > 0:
            result["raw_agreement_when_self_consistent"] = round(
                float((df.loc[sc, "original_code_id"] == df.loc[sc, "llm_modal_code_id"]).mean()), 4
            )
        if (~sc).sum() > 0:
            result["raw_agreement_when_NOT_self_consistent"] = round(
                float((df.loc[~sc, "original_code_id"] == df.loc[~sc, "llm_modal_code_id"]).mean()), 4
            )

    return result


def main():
    csv_path = OUT_DIR / "coded_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run run_llm_coding.py first.")
        return
    df = pd.read_csv(csv_path)

    report = {}
    report["overall"] = compute_for_subset(df, "overall")
    if "in_dev_set" in df.columns:
        dev_mask = df["in_dev_set"].astype(str).str.lower() == "true"
        report["dev_set"] = compute_for_subset(df[dev_mask], "dev")
        report["held_out_set"] = compute_for_subset(df[~dev_mask], "held_out")

    with open(OUT_DIR / "reliability_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote output/reliability_report.json")
    print("\n>>> Report the HELD-OUT numbers as your primary reliability statistic.")
    print(">>> The dev-set numbers only describe what the prompt was tuned on.")

    # disagreement sample for blind human adjudication
    df["original_code_id"] = df["original_code_id"].astype(str)
    df["llm_modal_code_id"] = df["llm_modal_code_id"].astype(str)
    disagreements = df[df["original_code_id"] != df["llm_modal_code_id"]]
    sample_n = min(150, len(disagreements))
    sample = disagreements.sample(n=sample_n, random_state=20260825) if sample_n > 0 else disagreements

    adjudication_rows = []
    import random as _random
    r = _random.Random(20260825)
    for _, row in sample.iterrows():
        candidates = [
            ("A", row["original_code_id"]),
            ("B", row["llm_modal_code_id"]),
        ]
        r.shuffle(candidates)  # blind labelling: adjudicator doesn't see which is which
        adjudication_rows.append({
            "row_id": row["row_id"],
            "framework": row["framework"],
            "component_text": row["component_text"],
            "candidate_A_label": candidates[0][1],
            "candidate_B_label": candidates[1][1],
            "which_was_human": "A" if candidates[0][1] == row["original_code_id"] else "B",
            "adjudicator_verdict": "",  # fill in: A / B / neither
        })

    adj_path = OUT_DIR / "adjudication_sample.csv"
    with open(adj_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(adjudication_rows[0].keys()) if adjudication_rows else
                            ["row_id","framework","component_text","candidate_A_label","candidate_B_label","which_was_human","adjudicator_verdict"])
        w.writeheader()
        w.writerows(adjudication_rows)

    print(f"\nWrote output/adjudication_sample.csv ({len(adjudication_rows)} disagreements).")
    print("IMPORTANT: before handing this to an adjudicator, delete the 'which_was_human'")
    print("column from their copy -- it exists here only so you can score their verdicts")
    print("afterward. The adjudicator must not see which candidate came from which source.")


if __name__ == "__main__":
    main()
