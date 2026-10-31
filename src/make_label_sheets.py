"""
Create blank labeling sheets for the independent human evaluation
of the Voyager RAG vs no-RAG experiment.

Each team member gets their own file and labels it without seeing
the other member's labels.
"""
import csv
from pathlib import Path

COMPARISON_FILE = Path("results/rag_vs_no_rag/comparison_v1.csv")
OUTPUT_DIR = Path("results/rag_vs_no_rag/labels")
EVALUATORS = ["elif", "ekin"]

OUTPUT_COLUMNS = [
    "id", "type", "question", "reference_answer",
    "system", "answer",
    "correctness",    # correct / partial / incorrect / abstain
    "hallucination",  # yes / no
    "comment",
]


def build_rows():
    """One row per answer: each question gets a RAG row and a no-RAG row."""
    rows = []
    with open(COMPARISON_FILE, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            for system, answer_col in [("rag", "answer_rag"), ("no_rag", "answer_no_rag")]:
                rows.append({
                    "id": r["id"],
                    "type": r["type"],
                    "question": r["question"],
                    "reference_answer": r["reference_answer"],
                    "system": system,
                    "answer": r[answer_col],
                    "correctness": "",
                    "hallucination": "",
                    "comment": "",
                })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    for name in EVALUATORS:
        out_path = OUTPUT_DIR / f"labels_{name}.csv"
        if out_path.exists():
            print(f"Skipped {out_path} (already exists, not overwritten)")
            continue
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()