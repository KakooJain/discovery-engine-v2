import json
import re
from pathlib import Path

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/clean")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

APP_FILES = [
    ("blinkit", RAW_DIR / "blinkit.json"),
    ("instamart", RAW_DIR / "instamart.json"),
    ("zepto", RAW_DIR / "zepto.json"),
]


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
    text = record.get("review") or record.get("content") or ""
    rating = record.get("score")
    date = record.get("date") or ""
    app = record.get("app") or ""

    return {
        "app": app,
        "text": text,
        "rating": rating,
        "date": date,
    }


if __name__ == "__main__":
    all_records = []
    started_count = 0
    short_text_dropped = 0
    non_english_dropped = 0
    low_word_count_dropped = 0
    duplicate_dropped = 0

    for app_name, file_path in APP_FILES:
        if not file_path.exists():
            print(f"Missing file for {app_name}: {file_path}")
            continue

        with file_path.open("r", encoding="utf-8") as handle:
            raw_records = json.load(handle)

        for record in raw_records:
            started_count += 1
            normalized = normalize_record(record)

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

    output_path = OUTPUT_DIR / "reviews.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(deduped_records, handle, ensure_ascii=False, indent=2)

    print(f"Started with {started_count} records")
    print(f"Dropped for short text: {short_text_dropped}")
    print(f"Dropped for non-English-like text: {non_english_dropped}")
    print(f"Dropped for low alphabetic word count: {low_word_count_dropped}")
    print(f"Dropped as duplicates: {duplicate_dropped}")
    print(f"Final count: {len(deduped_records)}")
