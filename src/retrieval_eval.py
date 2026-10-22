"""
Retrieval ölçümü: farklı k değerlerinde doğru kanıt getiriliyor mu?

LLM kullanılmaz. Her soru için:
  1. Soru embedding'e çevrilir, tüm parçalarla benzerliği hesaplanır.
  2. En benzer k parça alınır.
  3. sorular.csv'deki evidence_snippet metinleri bu parçaların içinde aranır.

İki ölçüt raporlanır:
  - hit@k : kanıt metinlerinden EN AZ BİRİ ilk k parçada var mı
  - full@k: kanıt metinlerinin HEPSİ ilk k parçada var mı
            (çok parçalı B tipi sorular için önemli)

C tipi (belgede olmayan) soruların kanıtı olmadığı için ölçüme katılmaz.

Kullanım:
    python src/retrieval_eval.py
Çıktı:
    results/retrieval_k.csv  (soru bazında sonuçlar)
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from chunking import chunk_document  # noqa: E402

DOC_PATH = "data/voyager_clean.txt"
QUESTIONS_PATH = "data/sorular.csv"
OUT_PATH = "results/retrieval_k.csv"
EMBED_MODEL = "intfloat/multilingual-e5-small"
K_VALUES = [1, 3, 5]


def normalize(s: str) -> str:
    """Büyük/küçük harf ve boşluk farklarını yok sayar."""
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        snippets = (r.get("evidence_snippet") or "").split("|")
        r["snippets"] = [normalize(s) for s in snippets if s.strip()]
    return rows


def embed(texts: list[str], prefix: str):
    """e5 modelleri 'query: ' ve 'passage: ' ön eklerini bekler."""
    from sentence_transformers import SentenceTransformer
    global _model
    if "_model" not in globals():
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode([prefix + t for t in texts], normalize_embeddings=True)


def main():
    chunks = chunk_document(DOC_PATH)
    chunk_texts_norm = [normalize(c.text) for c in chunks]
    questions = load_questions(QUESTIONS_PATH)

    # Ön kontrol: her kanıt metni en az bir parçada geçiyor mu?
    for q in questions:
        for s in q["snippets"]:
            if not any(s in t for t in chunk_texts_norm):
                print(f"UYARI: soru {q['id']} kanıtı hiçbir parçada bulunamadı "
                      f"(yazım farkı ya da parça sınırı olabilir): {s[:60]}...")

    chunk_vecs = embed([c.text for c in chunks], "passage: ")
    evaluated = [q for q in questions if q["snippets"]]
    skipped = len(questions) - len(evaluated)
    q_vecs = embed([q["question"] for q in evaluated], "query: ")

    scores = q_vecs @ chunk_vecs.T          # kosinüs benzerliği
    ranking = np.argsort(-scores, axis=1)   # en benzerden en aza

    rows = []
    totals = {k: {"hit": 0, "full": 0} for k in K_VALUES}
    for qi, q in enumerate(evaluated):
        row = {"id": q["id"], "type": q.get("type", ""),
               "evidence_section": q.get("evidence_section", "")}
        for k in K_VALUES:
            top = [chunk_texts_norm[i] for i in ranking[qi, :k]]
            found = [any(s in t for t in top) for s in q["snippets"]]
            row[f"hit@{k}"] = int(any(found))
            row[f"full@{k}"] = int(all(found))
            totals[k]["hit"] += row[f"hit@{k}"]
            totals[k]["full"] += row[f"full@{k}"]
        row["top3_sections"] = " | ".join(chunks[i].section_title for i in ranking[qi, :3])
        rows.append(row)

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(evaluated)
    print(f"\nParça sayısı: {len(chunks)} | Ölçülen soru: {n} "
          f"(kanıtı olmayan {skipped} soru atlandı)\n")
    print(f"{'k':>3} | {'hit@k':>12} | {'full@k':>12}")
    for k in K_VALUES:
        h, fu = totals[k]["hit"], totals[k]["full"]
        print(f"{k:>3} | {h:>2}/{n} (%{100*h/n:>3.0f}) | {fu:>2}/{n} (%{100*fu/n:>3.0f})")
    print(f"\nSoru bazında sonuçlar: {OUT_PATH}")


if __name__ == "__main__":
    main()