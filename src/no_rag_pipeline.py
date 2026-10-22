from pathlib import Path
import pandas as pd
import ollama


# --------------------------------------------------
# dosya yolları belirleniyor.
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "sorular.csv"

RESULTS_DIR = BASE_DIR / "results" / "rag_vs_no_rag"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULT_PATH = RESULTS_DIR / "no_rag_results.csv"


# --------------------------------------------------
#llm model ayarları ve modelin uyması gereken kurallar belirleniyor.
# --------------------------------------------------

MODEL_NAME = "qwen3:4b-instruct"

options = {
    "temperature": 0,
    "num_predict": 512,
    "num_ctx": 4096
}


# --------------------------------------------------
# veri seti okunuyor.
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Veri seti yüklendi.")
print("Toplam soru sayısı:", len(df))
print()


# --------------------------------------------------
# no rag deneyi
# --------------------------------------------------

results = []

for index, row in df.iterrows():

    question = row["question"]
    reference_answer = row["reference_answer"]

    # Modele sadece soru gönderiliyor.
    # Voyager dokümanı veya referans cevap gönderilmiyor.
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        options=options
    )

    answer_no_rag = response["message"]["content"]

    # --------------------------------------------------
    
    # Ollama süreleri nanosaniye olarak döndürür.
    # --------------------------------------------------

    total_duration = (
        response["total_duration"]
        / 1_000_000_000
    )

    load_duration = (
        response["load_duration"]
        / 1_000_000_000
    )

    prompt_eval_duration = (
        response["prompt_eval_duration"]
        / 1_000_000_000
    )

    eval_duration = (
        response["eval_duration"]
        / 1_000_000_000
    )

    # Model yükleme süresi çıkarılmış süre
    processing_time = (
        total_duration
        - load_duration
    )


    # --------------------------------------------------
    # token sayıları
    # --------------------------------------------------

    prompt_tokens = response["prompt_eval_count"]

    answer_tokens = response["eval_count"]


    # --------------------------------------------------
    # token üretim hızı
    # --------------------------------------------------

    if eval_duration > 0:
        tokens_per_second = (
            answer_tokens
            / eval_duration
        )
    else:
        tokens_per_second = None


    # --------------------------------------------------
    # SONUCU KAYDET
    # --------------------------------------------------

    results.append({
        "id": row["id"],
        "question": question,
        "reference_answer": reference_answer,
        "answer_no_rag": answer_no_rag,

        "model": MODEL_NAME,
        "temperature": options["temperature"],
        "num_predict": options["num_predict"],
        "num_ctx": options["num_ctx"],

        "total_duration_seconds":
            round(total_duration, 3),

        "load_duration_seconds":
            round(load_duration, 3),

        "processing_time_seconds":
            round(processing_time, 3),

        "prompt_eval_duration_seconds":
            round(prompt_eval_duration, 3),

        "eval_duration_seconds":
            round(eval_duration, 3),

        "prompt_tokens":
            prompt_tokens,

        "answer_tokens":
            answer_tokens,

        "tokens_per_second":
            round(tokens_per_second, 2)
            if tokens_per_second is not None
            else None
    })


    # --------------------------------------------------
    # İLERLEME BİLGİSİ
    # --------------------------------------------------

    print(
        f"{index + 1}/{len(df)} tamamlandı | "
        f"Süre: {processing_time:.2f} sn | "
        f"Prompt: {prompt_tokens} token | "
        f"Cevap: {answer_tokens} token | "
        f"Hız: {tokens_per_second:.2f} token/sn"
    )



results_df = pd.DataFrame(results)


results_df.to_csv(
    RESULT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print()
print("Sonuçlar başarıyla kaydedildi.")
print("Dosya:", RESULT_PATH)
print("Kaydedilen soru sayısı:", len(results_df))


# --------------------------------------------------
# DENEY ÖZETİ
# --------------------------------------------------

print()
print("NO-RAG EXPERIMENT SUMMARY")
print("-------------------------")

print("Model:", MODEL_NAME)
print("Temperature:", options["temperature"])
print("Num Predict:", options["num_predict"])
print("Num Context:", options["num_ctx"])

print(
    "Ortalama işlem süresi:",
    round(
        results_df[
            "processing_time_seconds"
        ].mean(),
        3
    ),
    "saniye"
)

print(
    "Ortalama prompt token:",
    round(
        results_df[
            "prompt_tokens"
        ].mean(),
        2
    )
)

print(
    "Ortalama cevap token:",
    round(
        results_df[
            "answer_tokens"
        ].mean(),
        2
    )
)

print(
    "Ortalama token/s:",
    round(
        results_df[
            "tokens_per_second"
        ].mean(),
        2
    )
)