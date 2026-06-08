import asyncio
import json
import random
from pathlib import Path
import argparse

from googletrans import Translator
from tqdm import tqdm


INPUT_SENTENCES = Path("data/bbc_wikipedia_sentences.jsonl")
OUTPUT_SYNTHETIC = Path("sentence_pairs_data/synthetic_wikipedia_bbc_to_id_en_googletrans.jsonl")
FAILED_OUTPUT = Path("sentence_pairs_data/failed_wikipedia_bbc_googletrans.jsonl")

# TOTAL_SUCCESSFUL_PAIRS = 10_000

# ID_RATIO = 0.70
# N_ID = int(TOTAL_SUCCESSFUL_PAIRS * ID_RATIO)
# N_EN = TOTAL_SUCCESSFUL_PAIRS - N_ID

SRC_LANG = "bbc"
SEED = 42

MIN_CHARS = 20
MAX_CHARS = 300

MAX_RETRIES = 3
SLEEP_EVERY_N_REQUESTS = 100
SLEEP_SECONDS = 3


def is_valid_sentence(text):
    if text is None:
        return False

    text = text.strip()

    if len(text) < MIN_CHARS:
        return False

    if len(text) > MAX_CHARS:
        return False

    if len(text.split()) < 4:
        return False

    return True


def is_valid_translation(source, target):
    if source is None or target is None:
        return False

    source = source.strip()
    target = target.strip()

    if len(source) < MIN_CHARS:
        return False

    if len(target) < 3:
        return False

    if len(target) > MAX_CHARS * 2:
        return False

    if source == target:
        return False

    # Avoid extreme length mismatch.
    if len(target) > len(source) * 5:
        return False

    if len(source) > len(target) * 5:
        return False

    return True


def load_bbc_sentences(path):
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            row = json.loads(line)

            text = row.get("text")

            if not is_valid_sentence(text):
                continue

            examples.append(
                {
                    "bbc": text.strip(),
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "sentence_index": row.get("sentence_index"),
                    "source_dataset": row.get("source_dataset", "wikipedia_bbc"),
                    "original_row": line_no,
                }
            )

    return examples


def build_jobs(examples, n_id, n_en):
    random.seed(SEED)
    random.shuffle(examples)

    id_candidates = examples[: min(len(examples), n_id * 2)]
    en_candidates = examples[n_id * 2 : n_id * 2 + n_en * 2]

    jobs = {
        "id": [],
        "en": [],
    }

    for i, row in enumerate(id_candidates):
        jobs["id"].append(
            {
                "job_id": f"bbc-id-{i}",
                "src_lang": SRC_LANG,
                "tgt_lang": "id",
                "source": row["bbc"],
                "wiki_id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "sentence_index": row["sentence_index"],
                "source_dataset": row["source_dataset"],
                "original_row": row["original_row"],
            }
        )

    for i, row in enumerate(en_candidates):
        jobs["en"].append(
            {
                "job_id": f"bbc-en-{i}",
                "src_lang": SRC_LANG,
                "tgt_lang": "en",
                "source": row["bbc"],
                "wiki_id": row["id"],
                "title": row["title"],
                "url": row["url"],
                "sentence_index": row["sentence_index"],
                "source_dataset": row["source_dataset"],
                "original_row": row["original_row"],
            }
        )

    return jobs


async def translate_with_retry(translator, text, src_lang, tgt_lang):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await translator.translate(
                text,
                src=src_lang,
                dest=tgt_lang,
            )
            return result.text

        except Exception as e:
            print(f"Attempt {attempt}/{MAX_RETRIES} failed for {src_lang}->{tgt_lang}: {e}")
            await asyncio.sleep(2 * attempt)

    return None


async def process_jobs_for_language(
    translator,
    jobs,
    needed,
    out,
    failed_out,
    description,
):
    written = 0
    failed = 0

    for idx, job in enumerate(tqdm(jobs, desc=description), start=1):
        if written >= needed:
            break

        source = job["source"]
        src_lang = job["src_lang"]
        tgt_lang = job["tgt_lang"]

        translated = await translate_with_retry(
            translator=translator,
            text=source,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )

        if not is_valid_translation(source, translated):
            failed_row = {
                **job,
                "translated": translated,
                "reason": "invalid_or_failed_translation",
            }
            failed_out.write(json.dumps(failed_row, ensure_ascii=False) + "\n")
            failed += 1
            continue

        item = {
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "source": source,
            "target": translated.strip(),
            "source_dataset": job["source_dataset"],
            "generation_method": "googletrans",
            "wiki_id": job["wiki_id"],
            "title": job["title"],
            "url": job["url"],
            "sentence_index": job["sentence_index"],
            "original_row": job["original_row"],
        }

        out.write(json.dumps(item, ensure_ascii=False) + "\n")
        written += 1

        if idx % SLEEP_EVERY_N_REQUESTS == 0:
            await asyncio.sleep(SLEEP_SECONDS)

    return written, failed


async def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic translations from Batak Wikipedia sentences using googletrans."
    )

    parser.add_argument(
        "--total",
        type=int,
        default=10_000,
        help="Total number of successful synthetic pairs to generate.",
    )

    parser.add_argument(
        "--id-ratio",
        type=float,
        default=0.70,
        help="Ratio of translations targeting Indonesian. Default: 0.70",
    )
    args = parser.parse_args()

    total_successful_pairs = args.total
    id_ratio = args.id_ratio

    n_id = int(total_successful_pairs * id_ratio)
    n_en = total_successful_pairs - n_id

    OUTPUT_SYNTHETIC.parent.mkdir(parents=True, exist_ok=True)
    FAILED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    examples = load_bbc_sentences(INPUT_SENTENCES)
    print(f"Loaded {len(examples)} valid Batak Wikipedia sentences.")

    jobs = build_jobs(examples, n_id=n_id, n_en=n_en)

    print(f"Target Batak Toba → Indonesian: {n_id}")
    print(f"Target Batak Toba → English: {n_en}")
    print(f"Prepared Indonesian jobs: {len(jobs['id'])}")
    print(f"Prepared English jobs: {len(jobs['en'])}")

    async with Translator() as translator:
        with OUTPUT_SYNTHETIC.open("w", encoding="utf-8") as out, \
             FAILED_OUTPUT.open("w", encoding="utf-8") as failed_out:

            written_id, failed_id = await process_jobs_for_language(
                translator=translator,
                jobs=jobs["id"],
                needed=n_id,
                out=out,
                failed_out=failed_out,
                description="Translating bbc→id",
            )

            written_en, failed_en = await process_jobs_for_language(
                translator=translator,
                jobs=jobs["en"],
                needed=n_en,
                out=out,
                failed_out=failed_out,
                description="Translating bbc→en",
            )

    total_written = written_id + written_en
    total_failed = failed_id + failed_en

    print()
    print(f"Saved {total_written} synthetic pairs to {OUTPUT_SYNTHETIC}")
    print(f"Batak Toba → Indonesian: {written_id}")
    print(f"Batak Toba → English: {written_en}")
    print(f"Failed rows: {total_failed}")
    print(f"Failed rows saved to {FAILED_OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
