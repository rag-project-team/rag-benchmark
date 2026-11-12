"""
Summary tables for the Voyager RAG vs no-RAG experiment.

Joins, by question id:
- the final human labels   results/rag_vs_no_rag/labels/labels_final.csv
- the RAGAS faithfulness   results/rag_vs_no_rag/ragas/ragas_scores.csv

ragas_scores.csv is a raw run log: it also holds old metrics from the first
test (factual_correctness, context_precision, context_recall) and rows that
failed on the daily quota. Only metric = faithfulness, system = rag rows that
have a score (or were skipped as abstention) are used; if a question has
several such rows, the latest one counts.

Output: results/rag_vs_no_rag/summary_v1.csv (one row per question).
Label counts and mean faithfulness per system are only printed, not saved.

Usage (from the repo root):
    python src/summarize_voyager.py
"""
import csv
import math
from pathlib import Path

LABELS_FILE = Path("results/rag_vs_no_rag/labels/labels_final.csv")
RAGAS_FILE = Path("results/rag_vs_no_rag/ragas/ragas_scores.csv")
SUMMARY_FILE = Path("results/rag_vs_no_rag/summary_v1.csv")

SYSTEMS = ["rag", "no_rag"]
CORRECTNESS = ["correct", "partial", "incorrect", "abstain"]
SKIP_ABSTENTION = "skipped: abstention (no claims to verify)"


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_faithfulness():
    """id -> score string ("1.0", "0.4", "nan"); latest valid row wins."""
    scores = {}
    for r in read_csv(RAGAS_FILE):
        if r["metric"] != "faithfulness" or r["system"] != "rag":
            continue
        if r["score"] not in ("", "nan"):
            scores[r["id"]] = r["score"]
        elif r["error"] == SKIP_ABSTENTION:
            scores[r["id"]] = "nan"
    return scores


def main():
    labels = {(r["id"], r["system"]): r for r in read_csv(LABELS_FILE)}
    faith = load_faithfulness()

    ids = sorted({qid for qid, _ in labels}, key=int)
    for qid in ids:
        for system in SYSTEMS:
            if (qid, system) not in labels:
                raise SystemExit(f"Q{qid}: no {system} row in {LABELS_FILE}")
        if qid not in faith:
            raise SystemExit(f"Q{qid}: no faithfulness score in {RAGAS_FILE}")

    # 1) one row per question
    summary = []
    for qid in ids:
        rag, no_rag = labels[(qid, "rag")], labels[(qid, "no_rag")]
        summary.append({
            "id": qid,
            "type": rag["type"],
            "question": rag["question"],
            "faithfulness_rag": faith[qid],
            "correctness_rag": rag["correctness_final"],
            "hallucination_rag": rag["hallucination_final"],
            "correctness_no_rag": no_rag["correctness_final"],
            "hallucination_no_rag": no_rag["hallucination_final"],
        })
    write_csv(SUMMARY_FILE, summary, list(summary[0].keys()))

    # 2) counts per system (printed only)
    by_system = []
    for system in SYSTEMS:
        rows = [labels[(qid, system)] for qid in ids]
        row = {"system": system, "n": len(rows)}
        for c in CORRECTNESS:
            row[c] = sum(r["correctness_final"] == c for r in rows)
        row["hallucination_yes"] = sum(r["hallucination_final"] == "yes" for r in rows)
        if system == "rag":
            scored = [float(faith[q]) for q in ids if faith[q] != "nan"]
            row["faithfulness_scored"] = len(scored)
            row["faithfulness_mean"] = round(sum(scored) / len(scored), 3)
        else:  # no retrieved contexts, so faithfulness does not apply
            row["faithfulness_scored"] = 0
            row["faithfulness_mean"] = ""
        by_system.append(row)

    # print both tables
    print(f"{'id':>3} {'type':<4} {'faith':>5}  {'rag':<18} {'no_rag':<18}")
    for s in summary:
        rag = f"{s['correctness_rag']}/{s['hallucination_rag']}"
        no_rag = f"{s['correctness_no_rag']}/{s['hallucination_no_rag']}"
        print(f"{s['id']:>3} {s['type']:<4} {s['faithfulness_rag']:>5}  {rag:<18} {no_rag:<18}")
    print("\n(correctness/hallucination)\n")
    for row in by_system:
        print(row)
    print(f"\nWritten: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()