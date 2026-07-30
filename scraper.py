import json
import time
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from google_play_scraper import Sort, reviews

PLAYSTORE_APP_ID = "com.grofers.customerapp"
APPSTORE_APP_ID = "960335206"
REDDIT_HEADERS = {"User-Agent": "discovery-engine-research/1.0"}
APPSTORE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}
REDDIT_QUERIES = [
    "blinkit",
    "blinkit review",
    "blinkit vs zepto",
    "blinkit experience",
    "quick commerce india",
    "blinkit grocery",
]
REDDIT_REQUEST_DELAY_SECONDS = 2

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _to_iso_date(value):
    if value is None:
        return ""

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    return str(value)


def _utc_seconds_to_iso(value):
    try:
        return datetime.utcfromtimestamp(float(value)).isoformat() + "Z"
    except (TypeError, ValueError, OSError, OverflowError):
        return ""


def _valid_text(value):
    text = str(value or "").strip()
    if not text:
        return ""
    if text in {"[deleted]", "[removed]"}:
        return ""
    return text


def _extract_appstore_text(entry):
    content = entry.get("content", {})
    if isinstance(content, dict):
        return str(content.get("label", "") or "")
    return ""


def _extract_appstore_rating(entry):
    rating = entry.get("im:rating", {})
    if isinstance(rating, dict):
        raw = rating.get("label")
        try:
            return int(str(raw))
        except (TypeError, ValueError):
            return None
    return None


def _extract_appstore_date(entry):
    updated = entry.get("updated", {})
    if isinstance(updated, dict):
        return str(updated.get("label", "") or "")
    return ""


def fetch_playstore_reviews():
    result = reviews(
        PLAYSTORE_APP_ID,
        lang="en",
        country="in",
        sort=Sort.NEWEST,
        count=10000,
    )

    review_items, _continuation_token = result
    records = []
    for item in review_items:
        records.append(
            {
                "source": "play_store",
                "app": "blinkit",
                "text": str(item.get("content", "") or ""),
                "rating": item.get("score"),
                "date": _to_iso_date(item.get("at")),
            }
        )

    return records


def fetch_appstore_reviews():
    all_records = []

    for page in range(1, 51):
        url = (
            "https://itunes.apple.com/in/rss/customerreviews/"
            f"page={page}/id={APPSTORE_APP_ID}/sortby=mostrecent/json"
        )

        try:
            request = Request(url, headers=APPSTORE_HEADERS)
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code == 404:
                break
            raise
        except URLError:
            break
        finally:
            time.sleep(1)

        feed = payload.get("feed", {})
        entries = feed.get("entry", [])
        if not entries:
            break

        if isinstance(entries, dict):
            entries = [entries]

        if page == 1 and entries:
            entries = entries[1:]

        if not entries:
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = _extract_appstore_text(entry)
            if not text:
                continue
            all_records.append(
                {
                    "source": "app_store",
                    "app": "blinkit",
                    "text": text,
                    "rating": _extract_appstore_rating(entry),
                    "date": _extract_appstore_date(entry),
                }
            )

    return all_records


def _fetch_json_url(url, headers):
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Reddit request failed for {url}: {exc}")
        return None
    finally:
        time.sleep(REDDIT_REQUEST_DELAY_SECONDS)


def _build_reddit_post_record(post_data):
    title = _valid_text(post_data.get("title"))
    body = _valid_text(post_data.get("selftext"))
    combined_parts = [part for part in [title, body] if part]
    text = "\n\n".join(combined_parts).strip()
    text = _valid_text(text)
    if not text:
        return None

    return {
        "source": "reddit",
        "app": "blinkit",
        "text": text,
        "rating": None,
        "date": _utc_seconds_to_iso(post_data.get("created_utc")),
    }


def _build_reddit_comment_records(comment_listing):
    if not isinstance(comment_listing, dict):
        return []

    data = comment_listing.get("data", {})
    children = data.get("children", []) if isinstance(data, dict) else []
    records = []
    for child in children:
        if not isinstance(child, dict):
            continue
        if child.get("kind") != "t1":
            continue
        comment_data = child.get("data", {})
        if not isinstance(comment_data, dict):
            continue
        text = _valid_text(comment_data.get("body"))
        if not text:
            continue
        records.append(
            {
                "source": "reddit",
                "app": "blinkit",
                "text": text,
                "rating": None,
                "date": _utc_seconds_to_iso(comment_data.get("created_utc")),
            }
        )
    return records


def fetch_reddit_reviews():
    records = []
    posts_fetched = 0
    comments_fetched = 0

    for query in REDDIT_QUERIES:
        search_url = "https://www.reddit.com/search.json?" + urlencode(
            {
                "q": query,
                "limit": 100,
                "sort": "relevance",
                "t": "year",
            }
        )

        search_payload = _fetch_json_url(search_url, REDDIT_HEADERS)
        if not search_payload:
            continue

        search_data = search_payload.get("data", {}) if isinstance(search_payload, dict) else {}
        children = search_data.get("children", []) if isinstance(search_data, dict) else []
        posts = []

        for child in children:
            if not isinstance(child, dict):
                continue
            post_data = child.get("data", {})
            if not isinstance(post_data, dict):
                continue

            post_record = _build_reddit_post_record(post_data)
            if not post_record:
                continue

            records.append(post_record)
            posts_fetched += 1
            posts.append(post_data)

        top_posts = sorted(posts, key=lambda item: item.get("score", 0), reverse=True)[:10]

        for post in top_posts:
            post_id = str(post.get("id", "")).strip()
            if not post_id:
                continue

            comments_url = f"https://www.reddit.com/comments/{post_id}.json?limit=100"
            comments_payload = _fetch_json_url(comments_url, REDDIT_HEADERS)
            if not isinstance(comments_payload, list) or len(comments_payload) < 2:
                continue

            comment_records = _build_reddit_comment_records(comments_payload[1])
            comments_fetched += len(comment_records)
            records.extend(comment_records)

    return records, posts_fetched, comments_fetched


def write_json(path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    playstore_reviews = fetch_playstore_reviews()
    appstore_reviews = fetch_appstore_reviews()
    reddit_records, reddit_posts, reddit_comments = fetch_reddit_reviews()

    write_json(OUTPUT_DIR / "playstore.json", playstore_reviews)
    write_json(OUTPUT_DIR / "appstore.json", appstore_reviews)
    write_json(OUTPUT_DIR / "reddit.json", reddit_records)

    print(f"Fetched {len(playstore_reviews)} reviews from Google Play")
    print(f"Fetched {len(appstore_reviews)} reviews from Apple App Store")
    print(f"Fetched {reddit_posts} Reddit posts")
    print(f"Fetched {reddit_comments} Reddit comments")
    print(f"Saved {len(reddit_records)} Reddit records")
    print("Summary:")
    print(f"- play_store: {len(playstore_reviews)}")
    print(f"- app_store: {len(appstore_reviews)}")
    print(f"- reddit_posts: {reddit_posts}")
    print(f"- reddit_comments: {reddit_comments}")
    print(f"- reddit_records: {len(reddit_records)}")
