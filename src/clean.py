import re
from pathlib import Path

RAW = Path("data/voyager.txt")
OUT = Path("data/voyager_clean.txt")

text = RAW.read_text(encoding="utf-8")
before = len(re.findall(r"\[\d+\]", text))

text = re.sub(r"\[\d+\]", "", text)             # [1], [23]
text = re.sub(r"\[(update|edit)\]", "", text)   # [update], [edit]
text = re.sub(r"[ \t]+", " ", text)             # fazla boşluklar
text = re.sub(r"\n{3,}", "\n\n", text).strip()  # fazla boş satırlar

OUT.write_text(text, encoding="utf-8")

print("Silinen atıf numarası:", before)
print("Kalan [rakam]:", len(re.findall(r"\[\d+\]", text)))
print("Kelime sayısı:", len(text.split()))