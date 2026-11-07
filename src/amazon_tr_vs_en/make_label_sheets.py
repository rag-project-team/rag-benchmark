"""
Create blank labeling sheets for the independent human evaluation
of the Amazon TR vs EN RAG experiment.

Each question pair (pair_id) gets two rows: the English answer and the
Turkish answer. The "system" column holds the language (en / tr), so
src/resolve_labels.py can later compare the two members' files with
only its paths changed.

Each team member gets their own file and labels it without seeing
the other member's labels.

Usage (from the repo root):
    python src/amazon_tr_vs_en/make_label_sheets.py
"""
import csv
from pathlib import Path

RESULTS_DIR = Path("results/amazon_tr_vs_en")
ANSWER_FILES = {
    "en": RESULTS_DIR / "rag_answers_en.csv",
    "tr": RESULTS_DIR / "rag_answers_tr.csv",
}
OUTPUT_DIR = RESULTS_DIR / "labels"
EVALUATORS = ["elif", "ekin"]

OUTPUT_COLUMNS = [
    "id", "type", "evidence_section", "question", "reference_answer",
    "system", "answer",
    "correctness",    # correct / partial / incorrect / abstain
    "hallucination",  # yes / no
    "comment",
]


def read_answers(path):
    """Read one language's answer file into a dict keyed by pair_id."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["pair_id"]: r for r in csv.DictReader(f)}


def build_rows():
    """Two rows per pair: English first, then Turkish."""
    answers = {lang: read_answers(p) for lang, p in ANSWER_FILES.items()}

    en_ids, tr_ids = set(answers["en"]), set(answers["tr"])
    if en_ids != tr_ids:
        raise SystemExit(f"pair_id mismatch between EN and TR: {sorted(en_ids ^ tr_ids)}")

    rows = []
    for pair_id in answers["en"]:  # keep the order of the English file
        for lang in ["en", "tr"]:
            r = answers[lang][pair_id]
            rows.append({
                "id": pair_id,
                "type": r["type"],
                "evidence_section": r["evidence_section"],
                "question": r["question"],
                "reference_answer": r["reference_answer"],
                "system": lang,
                "answer": r["answer_rag"],
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