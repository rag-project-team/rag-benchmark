"""
Amazon TR vs EN: compare the two independent human label files, report
inter-annotator agreement (raw agreement + Cohen's kappa) overall and per
language, and write a final label file in which the disagreements are
resolved by the team's decisions listed in RESOLUTIONS.

Same method as the Voyager script (src/resolve_labels.py); the kappa and
file-reading functions are imported from there so both experiments use
identical measurement code.

While RESOLUTIONS does not yet cover every disagreement, the script prints
the disagreeing rows (both labels + comments) and writes only agreement.csv,
not labels_final.csv.

Usage (from the repo root):
    python src/amazon_tr_vs_en/resolve_labels.py
"""

import csv
import sys
from pathlib import Path

# Reuse the Voyager functions (src/resolve_labels.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from resolve_labels import cohen_kappa, read_labels  # noqa: E402

LABEL_DIR = Path("results/amazon_tr_vs_en/labels")
FILE_A = LABEL_DIR / "labels_elif.csv"
FILE_B = LABEL_DIR / "labels_ekin.csv"
OUT_FINAL = LABEL_DIR / "labels_final.csv"
OUT_AGREEMENT = LABEL_DIR / "agreement.csv"

NAME_A = "elif"
NAME_B = "ekin"
LABEL_COLUMNS = ["correctness", "hallucination"]
SCOPES = ["all", "en", "tr"]

# Team decisions for rows where the two annotators disagreed.
# Key: (question id, system) -> final labels and the reason. Example:
#     ("A14", "tr"): {
#         "correctness": "partial",
#         "hallucination": "no",
#         "note": "Reason for the decision.",
#     },
RESOLUTIONS = {
}


def disagrees(ra, rb):
    return any(ra[c].strip() != rb[c].strip() for c in LABEL_COLUMNS)


def main():
    a = read_labels(FILE_A)
    b = read_labels(FILE_B)

    if set(a) != set(b):
        raise SystemExit(f"Row mismatch between files: {sorted(set(a) ^ set(b))}")

    keys = list(a)  # keep the order of file A

    # 1) Agreement: overall and per language
    agreement_rows = []
    print(f"Rows compared: {len(keys)}\n")
    for scope in SCOPES:
        scope_keys = keys if scope == "all" else [k for k in keys if k[1] == scope]
        for col in LABEL_COLUMNS:
            la = [a[k][col].strip() for k in scope_keys]
            lb = [b[k][col].strip() for k in scope_keys]
            agree = sum(x == y for x, y in zip(la, lb))
            raw = agree / len(scope_keys)
            kappa = cohen_kappa(la, lb)
            agreement_rows.append({
                "scope": scope,
                "label": col,
                "n_rows": len(scope_keys),
                "n_agree": agree,
                "raw_agreement": round(raw, 3),
                "cohen_kappa": round(kappa, 3),
            })
            print(f"{scope:4s} {col:14s} agree {agree}/{len(scope_keys)}  "
                  f"raw={raw:.3f}  kappa={kappa:.3f}")

    with open(OUT_AGREEMENT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agreement_rows[0]))
        w.writeheader()
        w.writerows(agreement_rows)
    print(f"\nWrote {OUT_AGREEMENT}")

    # 2) Disagreements without a team decision yet
    disagree_keys = [k for k in keys if disagrees(a[k], b[k])]
    unresolved = [k for k in disagree_keys if k not in RESOLUTIONS]
    print(f"\nDisagreements: {len(disagree_keys)} "
          f"(resolved: {len(disagree_keys) - len(unresolved)}, "
          f"unresolved: {len(unresolved)})")

    if unresolved:
        for k in unresolved:
            ra, rb = a[k], b[k]
            print("\n" + "-" * 70)
            print(f"{k[0]} ({ra['type']}, {k[1]})  Q: {ra['question']}")
            print(f"  Reference: {ra['reference_answer']}")
            print(f"  Answer:    {ra['answer']}")
            for name, r in ((NAME_A, ra), (NAME_B, rb)):
                labels = ", ".join(r[c].strip() for c in LABEL_COLUMNS)
                comment = r.get("comment", "").strip()
                print(f"  {name:5s}: {labels}" + (f"  | {comment}" if comment else ""))
        print("\nAdd decisions to RESOLUTIONS, then run again to write "
              f"{OUT_FINAL.name}.")
        return

    unused = set(RESOLUTIONS) - set(disagree_keys)
    if unused:
        print(f"\nNote: resolutions given for rows that already agreed: {sorted(unused)}")

    # 3) Final labels
    final_rows = []
    for k in keys:
        ra, rb = a[k], b[k]
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
        row[f"comment_{NAME_A}"] = ra.get("comment", "").strip()
        row[f"comment_{NAME_B}"] = rb.get("comment", "").strip()

        if k in RESOLUTIONS:
            for c in LABEL_COLUMNS:
                row[f"{c}_final"] = RESOLUTIONS[k][c]
            row["resolution_note"] = RESOLUTIONS[k]["note"]
        else:
            for c in LABEL_COLUMNS:
                row[f"{c}_final"] = ra[c].strip()
            row["resolution_note"] = ""
        final_rows.append(row)

    with open(OUT_FINAL, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(final_rows[0]))
        w.writeheader()
        w.writerows(final_rows)
    print(f"Wrote {OUT_FINAL} ({len(final_rows)} rows)")


if __name__ == "__main__":
    main()