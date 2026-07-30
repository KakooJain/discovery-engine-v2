import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_ALIASES = {
    "playstore": "play_store",
    "play_store": "play_store",
    "appstore": "app_store",
    "app_store": "app_store",
    "reddit": "reddit",
}


def is_englishish(text: str, threshold: float = 0.3) -> bool:
    if not text:
        return False

    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    punctuation = set(" .,!?:;\"'()[]{}-_/\\@#$%^&*+=<>`~")
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch in punctuation):
            pass

    total_chars = len(text)
    if total_chars == 0:
        return False

    non_english_ratio = non_ascii / total_chars
    return non_english_ratio <= threshold


def alphabetic_word_count(text: str) -> int:
    words = re.findall(r"[A-Za-z]{2,}", text)
    return len(words)


def normalize_record(record):
    text = record.get("text") or record.get("review") or record.get("content") or ""
    rating = record.get("rating", record.get("score"))
    date = record.get("date") or record.get("review_date") or ""
    app = record.get("app") or "blinkit"
    source = record.get("source") or ""

    return {
        "source": source,
        "app": app,
        "text": text,
        "rating": rating,
        "date": date,
    }


def infer_source(file_path: Path, record: dict) -> str:
    source = str(record.get("source") or "").strip()
    if source:
        return source
    return SOURCE_ALIASES.get(file_path.stem.lower(), file_path.stem.lower())


if __name__ == "__main__":
    all_records = []
    source_started = {}
    source_final = {}
    short_text_dropped = 0
    non_english_dropped = 0
    low_word_count_dropped = 0
    duplicate_dropped = 0

    raw_files = sorted(RAW_DIR.glob("*.json"))
    if not raw_files:
        print(f"No raw JSON files found in {RAW_DIR}")

    for file_path in raw_files:
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                raw_records = json.load(handle)
        except json.JSONDecodeError as exc:
            print(f"Skipping invalid JSON file {file_path.name}: {exc}")
            continue

        if not raw_records:
            continue

        for record in raw_records:
            source = infer_source(file_path, record)
            source_started[source] = source_started.get(source, 0) + 1
            normalized = normalize_record(record)
            normalized["source"] = source

            if len(normalized["text"]) < 25:
                short_text_dropped += 1
                continue

            if not is_englishish(normalized["text"]):
                non_english_dropped += 1
                continue

            if alphabetic_word_count(normalized["text"]) < 5:
                low_word_count_dropped += 1
                continue

            all_records.append(normalized)

    seen_texts = set()
    deduped_records = []
    for record in all_records:
        text = record["text"]
        if text in seen_texts:
            duplicate_dropped += 1
            continue
        seen_texts.add(text)
        deduped_records.append(record)

    for record in deduped_records:
        source = record.get("source") or "unknown"
        source_final[source] = source_final.get(source, 0) + 1

    output_path = OUTPUT_DIR / "reviews.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(deduped_records, handle, ensure_ascii=False, indent=2)

    print("Starting count per source:")
    for source, count in sorted(source_started.items()):
        print(f"- {source}: {count}")

    print(f"Dropped for short text: {short_text_dropped}")
    print(f"Dropped for non-English-like text: {non_english_dropped}")
    print(f"Dropped for low alphabetic word count: {low_word_count_dropped}")
    print(f"Dropped as duplicates: {duplicate_dropped}")
    print("Final count per source:")
    for source, count in sorted(source_final.items()):
        print(f"- {source}: {count}")
    print(f"Final count total: {len(deduped_records)}")
