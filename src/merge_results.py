"""
RAG ve no-RAG sonuçlarını soru bazında tek tabloda birleştirir.

Girdi:
    results/rag_answers.csv                  (rag_v1)
    results/rag_vs_no_rag/no_rag_results.csv (no_rag_v1)
Çıktı:
    results/rag_vs_no_rag/comparison_v1.csv

Not: Elle değerlendirme (etiket) sütunları şimdilik eklenmedi;
ekip metne yeterince hakim olunca ayrıca eklenecek.
"""

from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

RAG_PATH = RESULTS_DIR / "rag_answers.csv"
NO_RAG_PATH = RESULTS_DIR / "rag_vs_no_rag" / "no_rag_results.csv"
OUT_PATH = RESULTS_DIR / "rag_vs_no_rag" / "comparison_v1.csv"

LABEL_COLUMNS = [
    "notes",
]


def fail(message):
    print("HATA:", message)
    sys.exit(1)


# utf-8-sig: BOM'lu (no-RAG) ve BOM'suz (RAG) dosyaları aynı şekilde okur
rag = pd.read_csv(RAG_PATH, encoding="utf-8-sig")
no_rag = pd.read_csv(NO_RAG_PATH, encoding="utf-8-sig")

# --------------------------------------------------
# Tutarlılık kontrolleri: karşılaştırma ancak bunlar geçerse anlamlı
# --------------------------------------------------

if sorted(rag["id"]) != sorted(no_rag["id"]):
    fail("İki dosyadaki soru id'leri farklı.")

merged = rag.merge(no_rag, on="id", suffixes=("_rag", "_no_rag"))

for col in ["question", "reference_answer"]:
    diff = merged[merged[f"{col}_rag"] != merged[f"{col}_no_rag"]]
    if not diff.empty:
        fail(f"'{col}' şu id'lerde farklı: {list(diff['id'])}")

# RAG model sütunu "qwen3:4b-instruct (0edcdef34593)" biçiminde
model_ok = merged.apply(
    lambda r: str(r["model_rag"]).startswith(str(r["model_no_rag"])), axis=1
)
if not model_ok.all():
    fail("Model adları eşleşmiyor.")

for col in ["temperature", "num_predict"]:
    if (merged[f"{col}_rag"] != merged[f"{col}_no_rag"]).any():
        fail(f"'{col}' ayarı iki koşuda farklı.")

# --------------------------------------------------
# Tek tablo
# --------------------------------------------------

out = pd.DataFrame({
    "id": merged["id"],
    "type": merged["type"],
    "question": merged["question_rag"],
    "reference_answer": merged["reference_answer_rag"],

    "answer_rag": merged["answer_rag"],
    "answer_no_rag": merged["answer_no_rag"],

    "retrieved_sections": merged["retrieved_sections"],
    "retrieved_chunks": merged["retrieved_chunks"],

    # Süreler ms cinsinden; no-RAG için model yükleme hariç işlem süresi
    "rag_retrieval_ms": merged["retrieval_ms"],
    "rag_generation_ms": merged["generation_ms"],
    "no_rag_generation_ms": (merged["processing_time_seconds"] * 1000).round(1),

    "rag_prompt_tokens": merged["prompt_tokens_rag"],
    "no_rag_prompt_tokens": merged["prompt_tokens_no_rag"],
    "rag_answer_tokens": merged["answer_tokens_rag"],
    "no_rag_answer_tokens": merged["answer_tokens_no_rag"],

    "rag_done_reason": merged["done_reason_rag"],
    "no_rag_done_reason": merged["done_reason_no_rag"],

    "model": merged["model_rag"],
    "embedding": merged["embedding"],
    "top_k": merged["top_k"],
    "rag_prompt_version": merged["prompt_version_rag"],
    "no_rag_prompt_version": merged["prompt_version_no_rag"],
    "temperature": merged["temperature_rag"],
    "num_predict": merged["num_predict_rag"],
})

for col in LABEL_COLUMNS:
    out[col] = ""

out = out.sort_values("id")
out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

# --------------------------------------------------
# Özet
# --------------------------------------------------

print("Kaydedildi:", OUT_PATH)
print("Soru sayısı:", len(out))
print()
print("Tip bazında soru sayısı:")
print(out["type"].value_counts().sort_index().to_string())
print()
print("Ortalama üretim süresi (ms):")
print("  RAG   :", round(out["rag_generation_ms"].mean(), 1),
      "+ retrieval", round(out["rag_retrieval_ms"].mean(), 1))
print("  No-RAG:", round(out["no_rag_generation_ms"].mean(), 1))
print()
print("'I don't know' sayısı:")
for side in ["rag", "no_rag"]:
    n = out[f"answer_{side}"].str.strip().str.lower().str.startswith("i don't know").sum()
    print(f"  {side:7s}: {n}/{len(out)}")