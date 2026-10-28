"""
Amazon TR/EN veri kontrolü.

Kontrol edilenler:
  1. Her iki metinde de 5 paragraf var mı, metinlerde görünmez karakter (BOM) var mı?
  2. sorular_en.csv ve sorular_tr.csv aynı soruları aynı sırada mı içeriyor?
     (pair_id, type, evidence_section iki dilde aynı olmalı)
  3. Her kanıt alıntısı, belirtilen paragrafta birebir geçiyor mu?

Çalıştırma (repo kök klasöründen):
  python src/amazon_tr_vs_en/check_data.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "amazon"
LANGS = ["en", "tr"]
EXPECTED_PARAGRAPHS = 5


def read_paragraphs(lang):
    text = (DATA / f"amazon_{lang}.txt").read_text(encoding="utf-8")
    has_bom = "\ufeff" in text
    # Windows'ta Git satır sonlarını \r\n yapabilir; bölmeden önce eşitliyoruz
    text = text.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs, has_bom


def read_questions(lang):
    with open(DATA / f"sorular_{lang}.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_field(value):
    return [x.strip() for x in value.split("|") if x.strip()]


def main():
    errors = 0

    # 1. Metinler
    print("1) METİNLER")
    paragraphs = {}
    for lang in LANGS:
        paras, has_bom = read_paragraphs(lang)
        paragraphs[lang] = paras
        words = sum(len(p.split()) for p in paras)
        status = "OK" if len(paras) == EXPECTED_PARAGRAPHS else "HATA"
        print(f"   {lang}: {len(paras)} paragraf, {words} kelime -> {status}")
        if len(paras) != EXPECTED_PARAGRAPHS:
            errors += 1
        if has_bom:
            print(f"   {lang}: UYARI - metinde görünmez BOM karakteri var")
            errors += 1

    # 2. Soru dosyalarının eşleşmesi
    print("\n2) SORU DOSYALARI")
    questions = {lang: read_questions(lang) for lang in LANGS}
    for lang in LANGS:
        print(f"   {lang}: {len(questions[lang])} soru")
    if len(questions["en"]) != len(questions["tr"]):
        print("   HATA - soru sayıları farklı")
        errors += 1
    for q_en, q_tr in zip(questions["en"], questions["tr"]):
        for field in ["pair_id", "type", "evidence_section"]:
            if q_en[field] != q_tr[field]:
                print(f"   HATA - {q_en['pair_id']}: '{field}' iki dilde farklı")
                errors += 1
    print("   pair_id, type ve evidence_section eşleşmesi kontrol edildi")

    # 3. Kanıt alıntıları
    print("\n3) KANIT ALINTILARI")
    for lang in LANGS:
        checked = 0
        for q in questions[lang]:
            sections = split_field(q["evidence_section"])
            for snippet in split_field(q["evidence_snippet"]):
                checked += 1
                found_in = [f"P{i + 1}" for i, p in enumerate(paragraphs[lang]) if snippet in p]
                if not set(found_in) & set(sections):
                    print(f"   HATA - {q['id']}: alıntı {sections} içinde bulunamadı: {snippet[:60]}...")
                    errors += 1
        print(f"   {lang}: {checked} alıntı kontrol edildi")

    print("\nSONUÇ:", "Her şey doğru." if errors == 0 else f"{errors} sorun bulundu.")


if __name__ == "__main__":
    main()