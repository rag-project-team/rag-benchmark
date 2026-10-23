import pandas as pd
import ollama
import os


# --------------------------------------------------
# VERİ SETİNİ OKU
# --------------------------------------------------

df = pd.read_csv("data/sorular.csv")

print("Toplam soru sayısı:", len(df))
print()


# --------------------------------------------------
# MODEL AYARLARI
# --------------------------------------------------

MODEL_NAME = "qwen3:4b-instruct"

SYSTEM_PROMPT = """
Rules:
- Answer the question concisely.
- If you do not know the answer, say: I don't know.
- Do not guess.
- Do not invent information.
"""

options = {
    "temperature": 0,
    "num_predict": 256,
    "num_ctx": 4096
}


# --------------------------------------------------
# NO-RAG DENEYİ
# --------------------------------------------------

results = []

for index, row in df.iterrows():

    question = row["question"]
    reference_answer = row["reference_answer"]

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        options=options
    )

    answer_no_rag = response["message"]["content"]

    total_duration = response["total_duration"] / 1_000_000_000
    load_duration = response["load_duration"] / 1_000_000_000
    prompt_eval_duration = response["prompt_eval_duration"] / 1_000_000_000
    eval_duration = response["eval_duration"] / 1_000_000_000

    processing_time = total_duration - load_duration

    prompt_tokens = response["prompt_eval_count"]
    answer_tokens = response["eval_count"]

    if eval_duration > 0:
        tokens_per_second = answer_tokens / eval_duration
    else:
        tokens_per_second = None

    results.append({
        "id": row["id"],
        "question": question,
        "reference_answer": reference_answer,
        "answer_no_rag": answer_no_rag,

        "model": MODEL_NAME,
        "prompt_version": "no_rag_v1",

        "temperature": options["temperature"],
        "num_predict": options["num_predict"],
        "num_ctx": options["num_ctx"],

        "done_reason": response.get("done_reason"),

        "total_duration_seconds": round(total_duration, 3),
        "load_duration_seconds": round(load_duration, 3),
        "processing_time_seconds": round(processing_time, 3),
        "prompt_eval_duration_seconds": round(prompt_eval_duration, 3),
        "eval_duration_seconds": round(eval_duration, 3),

        "prompt_tokens": prompt_tokens,
        "answer_tokens": answer_tokens,

        "tokens_per_second": (
            round(tokens_per_second, 2)
            if tokens_per_second is not None
            else None
        )
    })

    print(
        f"{index + 1}/{len(df)} tamamlandı | "
        f"Süre: {processing_time:.2f} sn | "
        f"Prompt: {prompt_tokens} token | "
        f"Cevap: {answer_tokens} token | "
        f"Hız: {(tokens_per_second or 0):.2f} token/sn"
    )


# --------------------------------------------------
# SONUÇLARI KAYDET
# --------------------------------------------------

results_df = pd.DataFrame(results)

os.makedirs("results/rag_vs_no_rag", exist_ok=True)

file_path = "results/rag_vs_no_rag/no_rag_results.csv"

results_df.to_csv(
    file_path,
    index=False,
    encoding="utf-8-sig"
)

print()
print("Sonuçlar kaydedildi:", file_path)
print("Kaydedilen soru sayısı:", len(results_df))
print()


# --------------------------------------------------
# DENEY ÖZETİ
# --------------------------------------------------

print("NO-RAG EXPERIMENT SUMMARY")
print("-------------------------")

print("Model:", MODEL_NAME)
print("Prompt Version: no_rag_v1")
print("Temperature:", options["temperature"])
print("Num Predict:", options["num_predict"])
print("Num Context:", options["num_ctx"])

print(
    "Ortalama işlem süresi:",
    round(results_df["processing_time_seconds"].mean(), 3),
    "saniye"
)

print(
    "Ortalama prompt token:",
    round(results_df["prompt_tokens"].mean(), 2)
)

print(
    "Ortalama cevap token:",
    round(results_df["answer_tokens"].mean(), 2)
)

print(
    "Ortalama token/s:",
    round(results_df["tokens_per_second"].mean(), 2)
)

results_df