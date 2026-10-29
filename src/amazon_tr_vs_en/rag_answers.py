"""
Amazon TR vs EN — RAG cevap üretimi.

Her dil için ayrı indeks (retrieval_eval.py ile aynı: Amazon P1–P5 +
235 dikkat dağıtıcı paragraf). Her soru için en benzer TOP_K parça LLM'e
verilir.

Model, LLM ayarları, retrieval ve bağlam biçimi Voyager deneyindeki
src/rag_pipeline.py'den alınır. Tek fark sistem istemindeki bir satırdır:
"Answer in the same language as the question." (prompt_version rag_tren_v1).
İstem iki dilde birebir aynıdır.

Kullanım (Ollama açıkken, proje klasöründen):
    python src/amazon_tr_vs_en/rag_answers.py
Çıktı:
    results/amazon_tr_vs_en/rag_answers_en.csv
    results/amazon_tr_vs_en/rag_answers_tr.csv
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import ollama
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rag_pipeline as base  # noqa: E402  (Voyager deneyinin kodu: model, ayarlar, retriever)

# Aynı klasördeki retrieval_eval.py'den build_index (240 parçalık indeks)
spec = importlib.util.spec_from_file_location("amazon_eval", HERE / "retrieval_eval.py")
amazon_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(amazon_eval)

LANGS = ["en", "tr"]
TOP_K = 3
PROMPT_VERSION = "rag_tren_v1"
OUT_DIR = ROOT / "results" / "amazon_tr_vs_en"

# Voyager istemi (rag_v1) + "Answer in the same language as the question."
SYSTEM_PROMPT = """
You are answering questions for an evaluation experiment.

Rules:
- Answer only using the context provided with the question.
- Answer in the same language as the question.
- If the answer is not in the context, reply exactly: I don't know.
- Do not guess.
- Do not invent information.
- Keep the answer concise.
"""


class AmazonRetriever(base.Retriever):
    """Voyager'daki Retriever ile aynı; sadece indeks farklı (240 parça)."""

    def __init__(self, lang: str):
        self.chunks = amazon_eval.build_index(lang)
        self.vecs = base.embed([c.text for c in self.chunks], "passage: ")


def ask(retriever: AmazonRetriever, question: str) -> dict:
    t0 = time.perf_counter()
    retrieved = retriever.retrieve(question, k=TOP_K)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    r = ollama.chat(
        model=base.MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": base.build_user_message(question, retrieved)},
        ],
        options=base.LLM_OPTIONS,
    )
    load_ns = r["load_duration"] or 0
    prompt_tokens = r["prompt_eval_count"]
    if prompt_tokens and prompt_tokens >= base.LLM_OPTIONS["num_ctx"]:
        print(f"UYARI: istem {prompt_tokens} token, bağlam penceresi dolmuş olabilir.")

    return {
        "answer_rag": r["message"]["content"],
        "retrieved_chunks": " | ".join(c.id for c in retrieved),
        "retrieved_sections": " | ".join(c.section_title for c in retrieved),
        "retrieval_ms": round(retrieval_ms, 1),
        "generation_ms": round((r["total_duration"] - load_ns) / 1e6, 1),
        "prompt_tokens": prompt_tokens,
        "answer_tokens": r["eval_count"],
        "done_reason": getattr(r, "done_reason", None),  # "length" = sınıra takıldı
    }


def run(lang: str):
    retriever = AmazonRetriever(lang)
    df = pd.read_csv(ROOT / "data" / "amazon" / f"sorular_{lang}.csv")
    print(f"\n[{lang}] {len(retriever.chunks)} parça, {len(df)} soru")

    results = []
    for n, (_, row) in enumerate(df.iterrows(), 1):
        out = ask(retriever, row["question"])
        results.append({
            "id": row["id"],
            "pair_id": row["pair_id"],
            "language": lang,
            "type": row["type"],
            "source": row["source"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "evidence_section": row["evidence_section"],
            **out,
            "model": f"{base.MODEL} ({base.MODEL_ID})",
            "embedding": base.EMBED_MODEL,
            "top_k": TOP_K,
            "prompt_version": PROMPT_VERSION,
            "temperature": base.LLM_OPTIONS["temperature"],
            "num_predict": base.LLM_OPTIONS["num_predict"],
            "num_ctx": base.LLM_OPTIONS["num_ctx"],
        })
        print(f"[{lang}] {n}/{len(df)} {row['id']} tamamlandı")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"rag_answers_{lang}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Kaydedildi: {out_path.relative_to(ROOT)}")


def main():
    for lang in LANGS:
        run(lang)


if __name__ == "__main__":
    main()