"""
RAG'li soru-cevap deneyi.

Her soru için:
  1. Belgeden en ilgili TOP_K parça bulunur (multilingual-e5-small).
  2. Parçalar ve soru, sistem istemiyle birlikte LLM'e verilir.
  3. Cevap, getirilen parçalar, süreler ve token sayıları kaydedilir.

İstem, RAG'siz deneyle aynı iskelettedir; tek fark ilk kural
("verilen metne göre cevapla") ve soruya eklenen bağlamdır.
LLM ayarları (temperature, num_predict, num_ctx) RAG'siz deneyle aynı olmalıdır.

Kullanım (proje klasöründen):
    python src/rag_pipeline.py
Çıktı:
    results/rag_answers.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import ollama
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from chunking import chunk_document  # noqa: E402
from retrieval_eval import embed, EMBED_MODEL  # noqa: E402

# --- Deney ayarları ---
DOC_PATH = ROOT / "data" / "voyager_clean.txt"
QUESTIONS_PATH = ROOT / "data" / "sorular.csv"
OUT_PATH = ROOT / "results" / "rag_answers.csv"

MODEL = "qwen3:4b-instruct"
MODEL_ID = "0edcdef34593"          # `ollama list` çıktısındaki ID
TOP_K = 3
PROMPT_VERSION = "rag_v1"
LLM_OPTIONS = {
    "temperature": 0,
    "num_predict": 256,             # cevap için en fazla token (RAG'sizde de aynı)
    "num_ctx": 4096,                # bağlam penceresi; istem bunu aşmamalı
}

SYSTEM_PROMPT = """
You are answering questions for an evaluation experiment.

Rules:
- Answer only using the context provided with the question.
- If the answer is not in the context, reply exactly: I don't know.
- Do not guess.
- Do not invent information.
- Keep the answer concise.
"""


class Retriever:
    def __init__(self, doc_path: Path):
        self.chunks = chunk_document(doc_path)
        self.vecs = embed([c.text for c in self.chunks], "passage: ")

    def retrieve(self, question: str, k: int = TOP_K):
        q_vec = embed([question], "query: ")[0]
        scores = self.vecs @ q_vec
        top = np.argsort(-scores)[:k]
        return [self.chunks[i] for i in top]


def build_user_message(question: str, retrieved) -> str:
    context = "\n\n".join(
        f"[{n}] (Section: {c.section_title})\n{c.text}"
        for n, c in enumerate(retrieved, 1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


def ask(retriever: Retriever, question: str) -> dict:
    t0 = time.perf_counter()
    retrieved = retriever.retrieve(question)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    r = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(question, retrieved)},
        ],
        options=LLM_OPTIONS,
    )
    load_ns = r["load_duration"] or 0
    prompt_tokens = r["prompt_eval_count"]
    if prompt_tokens and prompt_tokens >= LLM_OPTIONS["num_ctx"]:
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


def main():
    retriever = Retriever(DOC_PATH)
    df = pd.read_csv(QUESTIONS_PATH)
    print(f"{len(retriever.chunks)} parça, {len(df)} soru")

    results = []
    for _, row in df.iterrows():
        out = ask(retriever, row["question"])
        results.append({
            "id": row["id"],
            "type": row["type"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            **out,
            "model": f"{MODEL} ({MODEL_ID})",
            "embedding": EMBED_MODEL,
            "top_k": TOP_K,
            "prompt_version": PROMPT_VERSION,
            "temperature": LLM_OPTIONS["temperature"],
            "num_predict": LLM_OPTIONS["num_predict"],
        })
        print(f"{row['id']}/{len(df)} tamamlandı")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUT_PATH, index=False)
    print(f"Kaydedildi: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()