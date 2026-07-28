import json
from pathlib import Path

from google_play_scraper import Sort, reviews

APPS = {
    "blinkit": "com.grofers.customerapp",
    "instamart": "in.swiggy.android",
    "zepto": "com.zeptoconsumerapp",
}

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_review(review_data, app_name):
    review_text = review_data.get("content") or review_data.get("review") or ""
    score = review_data.get("score")
    date_value = review_data.get("at")

    if isinstance(date_value, str):
        date_str = date_value
    elif date_value is None:
        date_str = ""
    else:
        date_str = date_value.isoformat()

    return {
        "app": app_name,
        "review": review_text,
        "score": score,
        "date": date_str,
    }


def fetch_and_save_reviews(app_name, package_name):
    result = reviews(
        package_name,
        lang="en",
        country="in",
        sort=Sort.NEWEST,
        count=500,
    )

    if isinstance(result, tuple):
        review_items = result[0] if result else []
    elif isinstance(result, dict):
        review_items = result.get("data", [])
    else:
        review_items = []

    cleaned_reviews = [clean_review(review, app_name) for review in review_items]

    output_path = OUTPUT_DIR / f"{app_name}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(cleaned_reviews, handle, ensure_ascii=False, indent=2)

    return len(cleaned_reviews)


if __name__ == "__main__":
    summary = {}

    for app_name, package_name in APPS.items():
        saved_count = fetch_and_save_reviews(app_name, package_name)
        summary[app_name] = saved_count
        print(f"Saved {saved_count} reviews for {app_name}")

    print("Summary:")
    for app_name, count in summary.items():
        print(f"- {app_name}: {count} reviews")
