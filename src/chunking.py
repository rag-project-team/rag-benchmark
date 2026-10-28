"""
Belge parçalama (chunking) modülü.

voyager_clean.txt gibi düz metin belgelerini bölüm başlıklarına göre ayırır,
ardından her bölümü kelime bazlı, örtüşmeli parçalara böler.

Kurallar:
- Başlık satırı: önceki satır boş (ya da dosyanın ilk satırı), kısa (en fazla
  6 kelime), büyük harfle başlıyor, noktalama ile bitmiyor ve "—" ile başlamıyor.
- İlk başlıktan önceki metin "Introduction" bölümü olarak etiketlenir
  (Wikipedia'da bu bölümün başlığı yoktur; metin dosyası değiştirilmez).
- Parçalar bölüm sınırlarını aşmaz: her parça tek bir bölüme aittir.

Kullanım (kontrol için):
    python src/chunking.py data/voyager_clean.txt
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path

INTRO_TITLE = "Introduction"
CHUNK_SIZE = 200      # kelime
CHUNK_OVERLAP = 40    # kelime
MAX_HEADING_WORDS = 6
NON_HEADING_ENDINGS = tuple('.!?:;,"\'”’)')


@dataclass
class Chunk:
    id: str
    document_id: str
    section_title: str
    chunk_index: int       # bölüm içindeki sırası
    text: str


def is_heading(line: str, prev_line: str | None) -> bool:
    s = line.strip()
    if not s:
        return False
    if prev_line is not None and prev_line.strip():
        return False  # başlıktan önce boş satır olmalı
    if len(s.split()) > MAX_HEADING_WORDS:
        return False
    if s.endswith(NON_HEADING_ENDINGS):
        return False
    if s.startswith(("—", "-", "–")):
        return False  # alıntı atfı, ör. "— President Jimmy Carter"
    return s[0].isupper()


def split_sections(text: str) -> list[tuple[str, str]]:
    """Metni (başlık, bölüm metni) çiftlerine ayırır."""
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = [(INTRO_TITLE, [])]
    prev = None
    for line in lines:
        if is_heading(line, prev):
            sections.append((line.strip(), []))
        else:
            sections[-1][1].append(line)
        prev = line
    result = []
    for title, body in sections:
        body_text = "\n".join(body).strip()
        if body_text:
            result.append((title, body_text))
    return result


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def chunk_document(path: str | Path,
                   size: int = CHUNK_SIZE,
                   overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    path = Path(path)
    document_id = path.stem
    text = path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    for title, body in split_sections(text):
        for i, piece in enumerate(chunk_words(body, size, overlap)):
            chunks.append(Chunk(
                id=f"{document_id}-{len(chunks):03d}",
                document_id=document_id,
                section_title=title,
                chunk_index=i,
                text=piece,
            ))
    return chunks


def chunk_by_paragraph(path: str | Path) -> list[Chunk]:
    """Her paragrafı (boş satırla ayrılmış) bir parça yapar.

    Bölüm başlığı olmayan paralel metinler (ör. Amazon TR/EN) için:
    parça sınırları iki dilde aynı yere düşer. Bölüm adı P1, P2, ...
    olur; sorular_*.csv'deki evidence_section ile aynıdır.
    """
    path = Path(path)
    document_id = path.stem
    # Windows'ta Git satır sonlarını \r\n yapabilir; bölmeden önce eşitliyoruz
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        Chunk(
            id=f"{document_id}-{i:03d}",
            document_id=document_id,
            section_title=f"P{i + 1}",
            chunk_index=0,
            text=p,
        )
        for i, p in enumerate(paragraphs)
    ]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    file = args[0] if args else "data/voyager_clean.txt"
    if "--paragraphs" in sys.argv:
        result = chunk_by_paragraph(file)
    else:
        result = chunk_document(file)
    print(f"Toplam parça: {len(result)}\n")
    for c in result:
        preview = c.text[:70].replace("\n", " ")
        print(f"{c.id} | {c.section_title:<20} | {len(c.text.split()):>3} kelime | {preview}...")