"""
Compare two independent human label files, report inter-annotator agreement
(raw agreement + Cohen's kappa), and write a final label file in which the
disagreements are resolved by the team's decisions listed in RESOLUTIONS.

Usage (from the repo root):
    python src/resolve_labels.py
"""

import csv
from collections import Counter
from pathlib import Path

LABEL_DIR = Path("results/rag_vs_no_rag/labels")
FILE_A = LABEL_DIR / "labels_elif.csv"
FILE_B = LABEL_DIR / "labels_ekin.csv"
OUT_FINAL = LABEL_DIR / "labels_final.csv"
OUT_AGREEMENT = LABEL_DIR / "agreement.csv"

NAME_A = "elif"
NAME_B = "ekin"
LABEL_COLUMNS = ["correctness", "hallucination"]

# Team decisions for rows where the two annotators disagreed (2026-11-07).
# Key: (question id, system) -> final labels and the reason.
RESOLUTIONS = {
    ("10", "rag"): {
        "correctness": "partial",
        "hallucination": "yes",
        "note": "Partly matches the reference (message about humanity) but misses "
                "the pulsar map and hydrogen diagram; claims the plaque has audio, "
                "which the document does not support.",
    },
    ("10", "no_rag"): {
        "correctness": "partial",
        "hallucination": "yes",
        "note": "Same content as the RAG answer; labelled the same way for consistency.",
    },
    ("12", "rag"): {
        "correctness": "correct",
        "hallucination": "no",
        "note": "Rubric rule: on C-type questions 'I don't know' counts as correct.",
    },
    ("13", "rag"): {
        "correctness": "correct",
        "hallucination": "no",
        "note": "Rubric rule: on C-type questions 'I don't know' counts as correct.",
    },
    ("13", "no_rag"): {
        "correctness": "correct",
        "hallucination": "no",
        "note": "Rubric rule: on C-type questions 'I don't know' counts as correct.",
    },
    ("15", "rag"): {
        "correctness": "partial",
        "hallucination": "no",
        "note": "D-type question: gives only Druyan's claim and does not mention "
                "Ferris's denial (same reasoning as question 14).",
    },
}


def read_labels(path):
    """Read a label file into a dict keyed by (id, system)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {(r["id"], r["system"]): r for r in rows}


def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa for two equally long lists of categorical labels."""
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    count_a, count_b = Counter(labels_a), Counter(labels_b)
    expected = sum(count_a[c] * count_b[c] for c in set(count_a) | set(count_b)) / (n * n)
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def main():
    a = read_labels(FILE_A)
    b = read_labels(FILE_B)

    if set(a) != set(b):
        raise SystemExit(f"Row mismatch between files: {sorted(set(a) ^ set(b))}")

    keys = list(a)  # keep the order of file A

    # 1) Agreement
    agreement_rows = []
    print(f"Rows compared: {len(keys)}\n")
    for col in LABEL_COLUMNS:
        la = [a[k][col].strip() for k in keys]
        lb = [b[k][col].strip() for k in keys]
        agree = sum(x == y for x, y in zip(la, lb))
        raw = agree / len(keys)
        kappa = cohen_kappa(la, lb)
        agreement_rows.append({
            "label": col,
            "n_rows": len(keys),
            "n_agree": agree,
            "raw_agreement": round(raw, 3),
            "cohen_kappa": round(kappa, 3),
        })
        print(f"{col:14s} agree {agree}/{len(keys)}  raw={raw:.3f}  kappa={kappa:.3f}")

    # 2) Final labels
    final_rows = []
    unresolved = []
    for k in keys:
        ra, rb = a[k], b[k]
        disagree = any(ra[c].strip() != rb[c].strip() for c in LABEL_COLUMNS)
        row = {
            "id": ra["id"],
            "type": ra["type"],
            "system": ra["system"],
            "question": ra["question"],
            "answer": ra["answer"],
        }
        for c in LABEL_COLUMNS:
            row[f"{c}_{NAME_A}"] = ra[c].strip()
            row[f"{c}_{NAME_B}"] = rb[c].strip()

        if not disagree:
            for c in LABEL_COLUMNS:
                row[f"{c}_final"] = ra[c].strip()
            row["resolution_note"] = ""
        elif k in RESOLUTIONS:
            for c in LABEL_COLUMNS:
                row[f"{c}_final"] = RESOLUTIONS[k][c]
            row["resolution_note"] = RESOLUTIONS[k]["note"]
        else:
            unresolved.append(k)
            continue
        final_rows.append(row)

    if unresolved:
        raise SystemExit(f"\nDisagreements without a resolution: {unresolved}")

    unused = set(RESOLUTIONS) - {k for k in keys
                                 if any(a[k][c].strip() != b[k][c].strip()
                                        for c in LABEL_COLUMNS)}
    if unused:
        print(f"\nNote: resolutions given for rows that already agreed: {sorted(unused)}")

    with open(OUT_AGREEMENT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agreement_rows[0]))
        w.writeheader()
        w.writerows(agreement_rows)

    with open(OUT_FINAL, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows[0]))
        w.writeheader()
        w.writerows(final_rows)

    print(f"\nResolved disagreements: {len(RESOLUTIONS)}")
    print(f"Wrote {OUT_AGREEMENT}")
    print(f"Wrote {OUT_FINAL} ({len(final_rows)} rows)")


if __name__ == "__main__":
    main()