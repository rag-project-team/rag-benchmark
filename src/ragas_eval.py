"""
RAGAS evaluation of the Voyager RAG vs no-RAG answers.

Judge LLM: Gemini (Google AI Studio, free tier) through Google's
OpenAI-compatible endpoint. The API key is read from the GEMINI_API_KEY
environment variable; it is never written to code or files.

Metric
- RAG rows only: faithfulness (see metric_calls() for why the other
  metrics were dropped). no-RAG rows have no retrieved contexts.
- "I don't know" (abstention) answers contain no claims, so faithfulness
  is undefined for them (RAGAS returns nan). They are not sent to the
  judge; a row with score "nan" and error "skipped: abstention" is
  written instead, and such rows count as done on the next run.

The retrieved chunk texts are rebuilt from data/voyager_clean.txt with the
same chunk_document() call rag_pipeline.py uses; their section titles are
checked against the retrieved_sections column of rag_answers.csv.

Results are appended row by row to results/rag_vs_no_rag/ragas/ragas_scores.csv.
Rows that already have a score are skipped, so the script can be stopped
(or hit the daily limit) and simply be run again to continue.

Usage (from the repo root):
    python src/ragas_eval.py --limit 2     # small test: first 2 questions
    python src/ragas_eval.py               # all questions
"""
import argparse
import asyncio
import csv
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
import ragas
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import chunk_document  # noqa: E402

RAG_FILE = Path("results/rag_answers.csv")
COMPARISON_FILE = Path("results/rag_vs_no_rag/comparison_v1.csv")  # has answer_no_rag
DOC_FILE = Path("data/voyager_clean.txt")
OUT_FILE = Path("results/rag_vs_no_rag/ragas/ragas_scores.csv")

JUDGE_MODEL = "gemini-3.5-flash"
GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

PAUSE_BETWEEN_CALLS = 4      # seconds between metric calls (free-tier rate limit)
RATE_LIMIT_WAIT = 65         # seconds to wait after a rate-limit error
MAX_RETRIES = 4

OUT_COLUMNS = ["id", "type", "system", "metric", "score", "error",
               "seconds", "judge_model", "ragas_version", "timestamp"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_list(value):
    return [v.strip() for v in value.split("|") if v.strip()]


def build_samples():
    """One sample per (question, system), with retrieved chunk texts for RAG."""
    chunks = {c.id: c for c in chunk_document(DOC_FILE)}
    no_rag = {r["id"]: r["answer_no_rag"] for r in read_csv(COMPARISON_FILE)}

    samples = []
    for r in read_csv(RAG_FILE):
        ids = split_list(r["retrieved_chunks"])
        sections = split_list(r["retrieved_sections"])
        for cid, sec in zip(ids, sections):
            if cid not in chunks:
                raise SystemExit(f"Q{r['id']}: chunk {cid} not found when re-chunking")
            if chunks[cid].section_title != sec:
                raise SystemExit(f"Q{r['id']}: chunk {cid} section is "
                                 f"'{chunks[cid].section_title}', expected '{sec}'")
        base = {"id": r["id"], "type": r["type"],
                "question": r["question"], "reference": r["reference_answer"]}
        samples.append({**base, "system": "rag", "answer": r["answer_rag"],
                        "contexts": [chunks[c].text for c in ids]})
        samples.append({**base, "system": "no_rag", "answer": no_rag[r["id"]],
                        "contexts": None})
    return samples


def metric_calls(llm, s):
    """(metric name, coroutine factory) pairs for one sample.

    Only faithfulness is computed, and only for RAG rows:
    - correctness is measured by the two-annotator human evaluation
      (RAGAS factual_correctness scored a correct Q1 answer 0.0 because
      the reference answers are short and have no subject);
    - retrieval quality is measured deterministically by retrieval_eval.py,
      so context_precision / context_recall are not needed;
    - no-RAG rows have no retrieved contexts, so faithfulness does not apply.
    """
    if s["system"] != "rag":
        return []
    return [
        ("faithfulness", lambda: Faithfulness(llm=llm).ascore(
            user_input=s["question"], response=s["answer"],
            retrieved_contexts=s["contexts"])),
    ]


ABSTENTION_ANSWERS = {"i don't know", "i do not know"}
SKIP_ABSTENTION = "skipped: abstention (no claims to verify)"


def is_abstention(answer):
    """True if the answer is just "I don't know" (case/punctuation ignored)."""
    text = answer.strip().lower().replace("’", "'").rstrip(".!")
    return text in ABSTENTION_ANSWERS


def is_rate_limit(err):
    text = str(err).lower()
    return "429" in text or "resource_exhausted" in text or "rate limit" in text or "quota" in text


def load_done():
    """(id, system, metric) triples that already have a numeric score,
    or were skipped on purpose because the answer was an abstention."""
    if not OUT_FILE.exists():
        return set()
    done = set()
    for r in read_csv(OUT_FILE):
        if r["score"] not in ("", "nan") or r["error"] == SKIP_ABSTENTION:
            done.add((r["id"], r["system"], r["metric"]))
    return done


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="only evaluate the first N questions (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY environment variable is not set")

    client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_OPENAI_URL, timeout=120)
    llm = llm_factory(JUDGE_MODEL, provider="openai", client=client,
                      temperature=0, max_tokens=8192)

    samples = build_samples()
    if args.limit:
        keep = []
        for s in samples:
            if s["id"] not in keep:
                keep.append(s["id"])
        keep = set(keep[:args.limit])
        samples = [s for s in samples if s["id"] in keep]

    done = load_done()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT_FILE.exists()

    with open(OUT_FILE, "a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        if new_file:
            writer.writeheader()

        for s in samples:
            for metric, make_call in metric_calls(llm, s):
                key = (s["id"], s["system"], metric)
                if key in done:
                    continue

                if is_abstention(s["answer"]):
                    writer.writerow({
                        "id": s["id"], "type": s["type"], "system": s["system"],
                        "metric": metric, "score": "nan",
                        "error": SKIP_ABSTENTION, "seconds": 0,
                        "judge_model": JUDGE_MODEL, "ragas_version": ragas.__version__,
                        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    })
                    f.flush()
                    print(f"Q{s['id']:>2} {s['system']:<6} {metric:<20} nan  "
                          f"(abstention, judge not called)")
                    continue

                score, error, start = math.nan, "", time.time()
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        result = await make_call()
                        score, error = float(result.value), ""
                        break
                    except Exception as e:  # keep going; record the error
                        error = f"{type(e).__name__}: {str(e)[:300]}"
                        if is_rate_limit(e) and attempt < MAX_RETRIES:
                            print(f"  rate limit, waiting {RATE_LIMIT_WAIT}s "
                                  f"(attempt {attempt}/{MAX_RETRIES})")
                            await asyncio.sleep(RATE_LIMIT_WAIT)
                        else:
                            break

                seconds = round(time.time() - start, 1)
                writer.writerow({
                    "id": s["id"], "type": s["type"], "system": s["system"],
                    "metric": metric,
                    "score": "" if math.isnan(score) else round(score, 4),
                    "error": error, "seconds": seconds,
                    "judge_model": JUDGE_MODEL, "ragas_version": ragas.__version__,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                f.flush()

                shown = "ERROR" if error else f"{score:.3f}"
                print(f"Q{s['id']:>2} {s['system']:<6} {metric:<20} {shown}  ({seconds}s)")
                if error:
                    print(f"    {error}")
                await asyncio.sleep(PAUSE_BETWEEN_CALLS)

    print(f"\nScores written to {OUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())