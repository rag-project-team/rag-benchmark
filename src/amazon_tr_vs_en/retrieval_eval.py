"""
Amazon TR vs EN — retrieval ölçümü.

İndeks (her dil için ayrı): Amazon'un 5 paragrafı (P1–P5) + XQuAD'ın diğer
47 makalesinden 235 dikkat dağıtıcı paragraf (D1–D235) = 240 parça.
Sorular yalnızca Amazon'dan sorulur.

Bir kanıt yalnızca Amazon parçasında bulunursa sayılır (aynı ifade bir
dikkat dağıtıcı paragrafta da geçse bile o parça doğru kabul edilmez).

Ölçütler (Voyager'daki src/retrieval_eval.py ile aynı mantık):
  - hit@k : kanıtlardan EN AZ BİRİ ilk k parçada var mı
  - full@k: kanıtların HEPSİ ilk k parçada var mı
  - first_rank: kanıt içeren ilk Amazon parçasının sırası

C tipi soruların kanıtı olmadığı için ölçüme katılmaz.

Kullanım:
    python src/amazon_tr_vs_en/retrieval_eval.py
Çıktı:
    results/amazon_tr_vs_en/retrieval_k_en.csv
    results/amazon_tr_vs_en/retrieval_k_tr.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
from chunking import chunk_by_paragraph  # noqa: E402
from retrieval_eval import normalize, load_questions, embed  # noqa: E402

LANGS = ["en", "tr"]
K_VALUES = [1, 3, 5, 10]
OUT_DIR = Path("results/amazon_tr_vs_en")


def build_index(lang):
    amazon = chunk_by_paragraph(f"data/amazon/amazon_{lang}.txt")
    distractors = chunk_by_paragraph(f"data/xquad/distractors_{lang}.txt")
    for c in distractors:
        c.section_title = "D" + c.section_title[1:]   # P12 -> D12
    return amazon + distractors


def evaluate(lang):
    chunks = build_index(lang)
    texts_norm = [normalize(c.text) for c in chunks]
    is_amazon = [c.document_id.startswith("amazon") for c in chunks]
    questions = load_questions(f"data/amazon/sorular_{lang}.csv")

    # Ön kontrol: kanıtlar Amazon'da var mı, başka makalede de geçiyor mu?
    for q in questions:
        for s in q["snippets"]:
            in_amazon = any(s in t for t, a in zip(texts_norm, is_amazon) if a)
            in_distr = any(s in t for t, a in zip(texts_norm, is_amazon) if not a)
            if not in_amazon:
                print(f"UYARI [{lang}] soru {q['id']}: kanıt Amazon parçalarında yok: {s[:60]}")
            if in_distr:
                print(f"Not [{lang}] soru {q['id']}: kanıt bir dikkat dağıtıcı paragrafta da geçiyor: {s[:60]}")

    evaluated = [q for q in questions if q["snippets"]]
    chunk_vecs = embed([c.text for c in chunks], "passage: ")
    q_vecs = embed([q["question"] for q in evaluated], "query: ")
    ranking = np.argsort(-(q_vecs @ chunk_vecs.T), axis=1)

    def found_in(i, s):
        return is_amazon[i] and s in texts_norm[i]

    rows = []
    for qi, q in enumerate(evaluated):
        row = {"id": q["id"], "pair_id": q.get("pair_id", ""), "type": q.get("type", ""),
               "evidence_section": q.get("evidence_section", "")}
        for k in K_VALUES:
            top = ranking[qi, :k]
            found = [any(found_in(i, s) for i in top) for s in q["snippets"]]
            row[f"hit@{k}"] = int(any(found))
            row[f"full@{k}"] = int(all(found))
        first = next((r + 1 for r, i in enumerate(ranking[qi])
                      if any(found_in(i, s) for s in q["snippets"])), None)
        row["first_rank"] = first
        row["top3_sections"] = " | ".join(chunks[i].section_title for i in ranking[qi, :3])
        rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"retrieval_k_{lang}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n[{lang}] Parça: {len(chunks)} | Ölçülen soru: {len(rows)} "
          f"(kanıtsız {len(questions) - len(rows)} soru atlandı)")
    return rows


def main():
    results = {lang: evaluate(lang) for lang in LANGS}
    print(f"\n{'k':>3} | {'EN hit':>8} | {'TR hit':>8} | {'EN full':>8} | {'TR full':>8}")
    for k in K_VALUES:
        cells = []
        for metric in ("hit", "full"):
            for lang in LANGS:
                rows = results[lang]
                v = sum(r[f"{metric}@{k}"] for r in rows)
                cells.append(f"{v:>2}/{len(rows)}")
        print(f"{k:>3} | " + " | ".join(f"{c:>8}" for c in cells))
    print(f"\nSoru bazında sonuçlar: {OUT_DIR}/retrieval_k_en.csv, retrieval_k_tr.csv")


if __name__ == "__main__":
    main()