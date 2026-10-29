"""
XQuAD'dan dikkat dağıtıcı paragrafları düz metne çevirir.

Amazon makalesi (XQuAD'da 16. makale) hariç 47 makalenin paragraflarını
data/xquad/distractors_en.txt ve distractors_tr.txt dosyalarına yazar.
Paragraflar boş satırla ayrılır; iki dosyada aynı sıradadır.
Önce 16. makalenin data/amazon/ altındaki metinlerle birebir aynı
olduğunu kontrol eder.

Kullanım (bir kez çalıştırılır):
    python src/amazon_tr_vs_en/make_distractors.py
"""
import json
import re
from pathlib import Path

XQUAD_DIR = Path("data/xquad")
AMAZON_DIR = Path("data/amazon")
AMAZON_IDX = 16


def clean(text):
    # baştaki/sondaki boşluk ve görünmez BOM karakteri silinir
    return text.replace("\ufeff", "").strip()


def load_xquad(lang):
    with open(XQUAD_DIR / f"xquad.{lang}.json", encoding="utf-8") as f:
        return json.load(f)["data"]


def load_paragraphs(path):
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return [clean(p) for p in text.split("\n\n") if clean(p)]


for lang in ("en", "tr"):
    articles = load_xquad(lang)

    # 1) Amazon kontrolü: 16. makale bizim txt dosyamızla aynı mı?
    amazon_json = [clean(p["context"]) for p in articles[AMAZON_IDX]["paragraphs"]]
    amazon_txt = load_paragraphs(AMAZON_DIR / f"amazon_{lang}.txt")
    if amazon_json != amazon_txt:
        raise SystemExit(f"[{lang}] HATA: 16. makale amazon_{lang}.txt ile aynı değil")

    # 2) Amazon dışındaki makalelerin paragrafları
    paragraphs = []
    for i, art in enumerate(articles):
        if i == AMAZON_IDX:
            continue
        for p in art["paragraphs"]:
            text = clean(p["context"])
            if "\n" in text:
                print(f"[{lang}] Not: makale {i} içindeki satır sonu boşlukla değiştirildi")
                text = re.sub(r"\s*\n\s*", " ", text)
            paragraphs.append(text)

    out = XQUAD_DIR / f"distractors_{lang}.txt"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(paragraphs) + "\n")

    n_words = sum(len(p.split()) for p in paragraphs)
    print(f"[{lang}] Amazon kontrolü OK ({articles[AMAZON_IDX]['title']})")
    print(f"[{lang}] {len(articles) - 1} makale, {len(paragraphs)} paragraf, "
          f"{n_words} kelime -> {out}")