"""
Amazon TR vs EN — token ve süre özeti.

rag_answers_en.csv ve rag_answers_tr.csv dosyalarındaki token sayılarını
(Ollama'nın ölçtüğü prompt_eval_count / eval_count) ve üretim sürelerini
dil bazında karşılaştırır. LLM çalıştırılmaz; sadece kayıtlı sonuçlar okunur.

Kullanım:
    python src/amazon_tr_vs_en/token_summary.py
Çıktı:
    results/amazon_tr_vs_en/token_summary.csv
"""
from pathlib import Path

import pandas as pd

RESULTS = Path("results/amazon_tr_vs_en")

d = pd.concat([pd.read_csv(RESULTS / f"rag_answers_{l}.csv") for l in ["en", "tr"]])
summary = (d.groupby("language")[["prompt_tokens", "answer_tokens", "generation_ms"]]
             .agg(["mean", "sum"]).round(1))
summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]

out = RESULTS / "token_summary.csv"
summary.to_csv(out)
print(summary.to_string())
print(f"\nKaydedildi: {out}")