import json
import os
import re
import time
import argparse
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq
from supabase import create_client

from taxonomy import TAXONOMY_PROMPT

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PROVIDER = os.getenv("PROVIDER", "gemini").strip().lower()
_gemini_models_env = os.getenv("GEMINI_MODELS", "").strip()
_gemma_model_env = os.getenv("GEMMA_MODEL", "gemma-4-31b-it").strip()
if PROVIDER == "gemma":
    GEMINI_MODELS = [_gemma_model_env]
elif _gemini_models_env:
    GEMINI_MODELS = [item.strip() for item in _gemini_models_env.split(",") if item.strip()]
else:
    GEMINI_MODELS = [
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
    ]

if PROVIDER not in {"groq", "gemini", "gemma"}:
    raise RuntimeError("PROVIDER must be one of: 'groq', 'gemini', 'gemma'")

if PROVIDER == "groq" and not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set in .env when PROVIDER='groq'")

if PROVIDER in {"gemini", "gemma"} and not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY must be set in .env when PROVIDER is 'gemini' or 'gemma'")
if PROVIDER in {"gemini", "gemma"} and not GEMINI_MODELS:
    raise RuntimeError("GEMINI_MODELS must contain at least one model when PROVIDER is 'gemini' or 'gemma'")

print(f"GEMINI_MODELS env raw: {_gemini_models_env or '<unset>'}")
print(f"GEMMA_MODEL env raw: {_gemma_model_env or '<unset>'}")
print(f"GEMINI_MODELS configured: {', '.join(GEMINI_MODELS)}")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env")

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
active_gemini_model_index = 0
exhausted_gemini_models = set()
gemini_request_count = 0

print(f"Gemini first request model (initial): {GEMINI_MODELS[active_gemini_model_index]}")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
FAILED_IDS_PATH = Path(__file__).resolve().parent / "failed_ids.txt"
SYSTEM_PROMPT_SUFFIX = "\n\nReturn ONLY valid JSON with exactly these keys: behavioral_driver, discovery_barrier, discovery_channel, frustration_type, segment_marker, unmet_need, categories, sentiment_score, supporting_quote. Use plain integers for sentiment_score, such as -2, -1, 0, 1, 2. Do not use plus signs or strings for sentiment_score."


class RateLimitError(Exception):
    pass


def _fetch_all_rows(table_name, select_cols, page_size=1000):
    rows = []
    start = 0
    while True:
        end = start + page_size - 1
        response = (
            supabase.table(table_name)
            .select(select_cols)
            .range(start, end)
            .execute()
        )
        batch = getattr(response, "data", None) or []
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < page_size:
            break
        start += page_size
    return rows

ALLOWED_TAGS = {
    "behavioral_driver": {
        "habit-routine",
        "trust-familiarity",
        "convenience-speed",
        "price-sensitivity",
    },
    "discovery_barrier": {
        "discovery-friction",
        "trust-deficit-new-category",
        "info-gap",
        "risk-aversion",
        "no-need-perceived",
        "habit-lock-in",
    },
    "discovery_channel": {
        "algo-discovery",
        "social-discovery",
        "search-driven",
        "offline-to-online",
        "word-of-mouth",
    },
    "frustration_type": {
        "stockout-availability",
        "quality-freshness",
        "delivery-experience",
        "pricing-surge",
        "app-ux-bug",
        "cs-support",
        "return-refund",
    },
    "segment_marker": {
        "power-user",
        "occasional-user",
        "new-user",
        "price-conscious",
        "premium-experimenter",
        "parent-household",
        "pet-owner",
        "single-professional",
    },
    "unmet_need": {
        "unmet-need-category",
        "unmet-need-bundling",
        "unmet-need-trust-signal",
    },
    "categories": {
        "groceries",
        "fruits-vegetables",
        "dairy",
        "snacks-beverages",
        "household-essentials",
        "personal-care",
        "baby-care",
        "pet-supplies",
        "electronics",
        "home-decor",
        "stationery",
        "pharmacy",
        "meat-seafood",
        "bakery",
        "frozen-food",
    },
}


def _coerce_array(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _infer_sentiment_score(text, score):
    text_l = (text or "").lower()
    if not text_l:
        return score

    strong_neg = ["bakwas", "ghatiya", "worst", "scam", "fraud", "never again", "uninstall", "terrible", "awful", "hate"]
    neg_cues = ["bad", "bug", "issue", "problem", "difficult", "hard", "cancel", "fail", "late", "delay", "wrong", "expensive", "overpriced", "waste", "refund", "stale", "rotten", "damaged", "stock out", "sold out", "nahi", "difficult", "hard"]
    strong_pos = ["best", "love it", "amazing", "mast"]
    pos_cues = ["good", "works fine", "achha", "great", "excellent", "fantastic", "awesome", "smooth", "fast", "easy"]

    has_strong_neg = any(term in text_l for term in strong_neg)
    has_neg = has_strong_neg or any(term in text_l for term in neg_cues)
    has_strong_pos = any(term in text_l for term in strong_pos)
    has_pos = has_strong_pos or any(term in text_l for term in pos_cues)

    if has_pos and has_neg:
        return 0
    if has_strong_neg:
        return -2
    if has_neg:
        return -1
    if has_strong_pos:
        return 2
    if has_pos:
        return 1
    return score


def _sanitize_payload(raw_payload, text=""):
    if not isinstance(raw_payload, dict):
        return {
            "behavioral_driver": [],
            "discovery_barrier": [],
            "discovery_channel": [],
            "frustration_type": [],
            "segment_marker": [],
            "unmet_need": [],
            "categories": [],
            "sentiment_score": 0,
            "supporting_quote": "",
        }

    payload = {
        "behavioral_driver": [],
        "discovery_barrier": [],
        "discovery_channel": [],
        "frustration_type": [],
        "segment_marker": [],
        "unmet_need": [],
        "categories": [],
        "sentiment_score": 0,
        "supporting_quote": "",
    }

    for field, allowed in ALLOWED_TAGS.items():
        if field == "categories":
            values = _coerce_array(raw_payload.get(field))
            cleaned = []
            for value in values:
                if isinstance(value, str) and value.strip() in allowed:
                    cleaned.append(value.strip())
            payload[field] = cleaned
        else:
            values = _coerce_array(raw_payload.get(field))
            cleaned = []
            for value in values:
                if isinstance(value, str) and value.strip() in allowed:
                    cleaned.append(value.strip())
            payload[field] = cleaned

    try:
        value = int(str(raw_payload.get("sentiment_score", 0)).replace("+", ""))
    except (TypeError, ValueError):
        value = 0
    payload["sentiment_score"] = max(-2, min(2, _infer_sentiment_score(text, value)))

    quote = raw_payload.get("supporting_quote")
    if isinstance(quote, str):
        words = quote.strip().split()
        if len(words) > 25:
            quote = " ".join(words[:25])
        if len(words) > 15:
            quote = " ".join(words[:15])
        payload["supporting_quote"] = quote

    return payload


def _validate_json_payload(payload, text=""):
    if not isinstance(payload, dict):
        raise ValueError("Model response is not a JSON object")

    expected_keys = {
        "behavioral_driver",
        "discovery_barrier",
        "discovery_channel",
        "frustration_type",
        "segment_marker",
        "unmet_need",
        "categories",
        "sentiment_score",
        "supporting_quote",
    }
    missing_keys = expected_keys - set(payload.keys())
    if missing_keys:
        for key in missing_keys:
            payload[key] = [] if key != "sentiment_score" and key != "supporting_quote" else 0 if key == "sentiment_score" else ""
    return _sanitize_payload(payload, text=text)


def _parse_json_response(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(1))

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
        raise ValueError(f"No object found in parsed list: {parsed}")

    if isinstance(parsed, dict):
        return parsed

    raise ValueError(f"Unexpected JSON payload: {parsed}")


def _coerce_text_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                part_text = part.get("text") or part.get("content") or ""
            elif hasattr(part, "text"):
                part_text = getattr(part, "text")
            else:
                part_text = str(part)
            if part_text:
                parts.append(part_text)
        return "\n".join(parts)
    if hasattr(content, "text"):
        return getattr(content, "text")
    return str(content)


def _is_rate_limit_error(exc):
    error_text = str(exc).lower()
    return (
        "429" in error_text
        or "rate_limit" in error_text
        or "rate limit" in error_text
        or "daily_limit_all_gemini_models_exhausted" in error_text
    )


def _append_failed_id(raw_review_id):
    if raw_review_id is None:
        return
    with FAILED_IDS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{raw_review_id}\n")


def _active_gemini_model():
    return GEMINI_MODELS[active_gemini_model_index]


def _is_gemini_per_day_limit_error(exc):
    error_text = str(exc).lower()
    return (
        "429" in error_text
        and (
            "generate_content_free_tier_requests" in error_text
            or "generaterequestsperdayperprojectpermodel-freetier" in error_text
            or "quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests" in error_text
        )
    )


def _rotate_gemini_model_if_exhausted(exc):
    global active_gemini_model_index
    if not _is_gemini_per_day_limit_error(exc):
        return False

    current_model = _active_gemini_model()
    exhausted_gemini_models.add(current_model)

    for index, model_name in enumerate(GEMINI_MODELS):
        if model_name in exhausted_gemini_models:
            continue
        active_gemini_model_index = index
        print(f"Model {current_model} exhausted, switching to {model_name}")
        return True

    return False


def _cleanup_failed_ids_against_analyzed_reviews():
    if not FAILED_IDS_PATH.exists():
        return

    existing_rows = _fetch_all_rows("analyzed_reviews", "raw_review_id")
    existing_ids = {
        str(row.get("raw_review_id")).strip()
        for row in existing_rows
        if row.get("raw_review_id") is not None
    }

    cleaned = []
    seen = set()
    with FAILED_IDS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            candidate = line.strip()
            if not candidate:
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate in existing_ids:
                continue
            cleaned.append(candidate)

    with FAILED_IDS_PATH.open("w", encoding="utf-8") as handle:
        for candidate in cleaned:
            handle.write(f"{candidate}\n")


def _extract_usage_tokens(response):
    usage = getattr(response, "usage", None)
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
    else:
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            if isinstance(usage_metadata, dict):
                prompt_tokens = usage_metadata.get("prompt_token_count", 0)
                completion_tokens = usage_metadata.get("candidates_token_count", 0)
                total_tokens = usage_metadata.get("total_token_count", 0)
            else:
                prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
                completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
                total_tokens = getattr(usage_metadata, "total_token_count", 0)

            if not total_tokens:
                total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    return {
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def _call_groq(system_prompt, user_text):
    if groq_client is None:
        raise RuntimeError("Groq client is not configured")

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0,
        max_tokens=220,
        response_format={"type": "json_object"},
    )

    message = (response.choices[0].message.content if response.choices else "") or ""
    normalized_content = _coerce_text_content(message)
    if not normalized_content:
        raise ValueError("Empty response content")

    return normalized_content, response


def _call_gemini(system_prompt, user_text):
    global gemini_request_count

    if gemini_client is None:
        raise RuntimeError("Gemini client is not configured")

    while True:
        current_model = _active_gemini_model()
        gemini_request_count += 1
        print(f"Gemini request #{gemini_request_count} using model: {current_model}")
        try:
            config_kwargs = {
                "system_instruction": system_prompt,
                "temperature": 0,
                "max_output_tokens": 220,
            }
            if current_model.startswith("gemma-4-"):
                # Gemma-4 spends tokens on thoughts; keep a larger output budget and avoid JSON MIME.
                config_kwargs["max_output_tokens"] = 2000
            else:
                config_kwargs["response_mime_type"] = "application/json"

            response = gemini_client.models.generate_content(
                model=current_model,
                contents=user_text,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            normalized_content = ""
            candidates = getattr(response, "candidates", None) or []
            if candidates:
                candidate_content = getattr(candidates[0], "content", None)
                parts = getattr(candidate_content, "parts", None) if candidate_content else None
                if parts:
                    normalized_content = "\n".join(
                        getattr(part, "text", "") for part in parts if getattr(part, "text", "")
                    )

            if not normalized_content:
                content = getattr(response, "text", "")
                normalized_content = _coerce_text_content(content)

            if not normalized_content:
                raise ValueError("Empty response content")

            return normalized_content, response
        except Exception as exc:
            if _rotate_gemini_model_if_exhausted(exc):
                continue
            if _is_gemini_per_day_limit_error(exc) and len(exhausted_gemini_models) >= len(GEMINI_MODELS):
                raise RateLimitError("DAILY_LIMIT_ALL_GEMINI_MODELS_EXHAUSTED") from exc
            raise


def classify_review(text, retries=3, rate_limit_delay=60):
    for attempt in range(retries):
        try:
            system_prompt = TAXONOMY_PROMPT + SYSTEM_PROMPT_SUFFIX
            if PROVIDER == "groq":
                normalized_content, response = _call_groq(system_prompt, text)
            elif PROVIDER in {"gemini", "gemma"}:
                normalized_content, response = _call_gemini(system_prompt, text)
            else:
                raise RuntimeError(f"Unsupported PROVIDER: {PROVIDER}")

            try:
                parsed = _parse_json_response(normalized_content)
            except Exception:
                if attempt < retries - 1:
                    continue
                raise

            return _validate_json_payload(parsed, text=text), _extract_usage_tokens(response)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                if attempt < retries - 1:
                    delay = rate_limit_delay
                    print(f"Transient API issue ({exc}); retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                raise RateLimitError(str(exc))

            error_text = str(exc).lower()
            should_retry = attempt < retries - 1 and any(token in error_text for token in ["timed out", "connection reset", "temporarily unavailable", "json_validate_failed", "failed_generation", "failed to generate", "invalid json"])
            if should_retry:
                delay = 2 ** attempt + 1
                print(f"Transient API issue ({exc}); retrying in {delay}s...")
                time.sleep(delay)
                continue
            raise


def classify_and_store_all(
    max_classifications=None,
    progress_every=50,
    sleep_seconds=5,
    retries=3,
    rate_limit_delay=60,
    stop_on_daily_limit=True,
):
    raw_rows = _fetch_all_rows("raw_reviews", "id,text")
    if not raw_rows:
        print("No raw reviews found.")
        return

    _cleanup_failed_ids_against_analyzed_reviews()

    analyzed_rows = _fetch_all_rows("analyzed_reviews", "raw_review_id")
    existing_ids = {
        item.get("raw_review_id")
        for item in analyzed_rows
        if item.get("raw_review_id") is not None
    }

    total_rows = len(raw_rows)
    skipped_count = 0
    succeeded_count = 0
    failed_count = 0
    consecutive_rate_limit_failures = 0
    classified_attempts = 0
    usage_total_tokens_sum = 0
    usage_request_count = 0

    for index, row in enumerate(raw_rows, start=1):
        if max_classifications is not None and classified_attempts >= max_classifications:
            break

        raw_review_id = row.get("id")
        text = row.get("text") or ""

        if raw_review_id in existing_ids:
            skipped_count += 1
            if skipped_count % 50 == 0 or index == total_rows:
                print(f"Skipped {skipped_count} existing rows so far")
            continue

        classified_attempts += 1
        try:
            classification, usage = classify_review(
                text,
                retries=retries,
                rate_limit_delay=rate_limit_delay,
            )
            usage_total_tokens_sum += usage.get("total_tokens", 0)
            usage_request_count += 1
            if not isinstance(classification, dict):
                raise ValueError(f"Unexpected classification payload: {classification}")
            payload = {
                "raw_review_id": raw_review_id,
                "behavioral_driver": classification.get("behavioral_driver", []),
                "discovery_barrier": classification.get("discovery_barrier", []),
                "discovery_channel": classification.get("discovery_channel", []),
                "frustration_type": classification.get("frustration_type", []),
                "segment_marker": classification.get("segment_marker", []),
                "unmet_need": classification.get("unmet_need", []),
                "categories": classification.get("categories", []),
                "sentiment_score": classification.get("sentiment_score", 0),
                "supporting_quote": classification.get("supporting_quote", ""),
            }
            insert_response = supabase.table("analyzed_reviews").insert(payload).execute()
            if getattr(insert_response, "data", None):
                succeeded_count += 1
                consecutive_rate_limit_failures = 0
            else:
                failed_count += 1
                consecutive_rate_limit_failures = 0
                print(f"Failed to insert for raw_review_id {raw_review_id}: no data returned")
        except RateLimitError as exc:
            failed_count += 1
            consecutive_rate_limit_failures += 1
            _append_failed_id(raw_review_id)
            print(f"Failed for raw_review_id {raw_review_id}: {exc}")
            if stop_on_daily_limit and consecutive_rate_limit_failures >= 10:
                if "DAILY_LIMIT_ALL_GEMINI_MODELS_EXHAUSTED" in str(exc):
                    print("DAILY LIMIT LIKELY REACHED — stopping cleanly")
                else:
                    print(f"Rate limit encountered ({exc})")
                break
        except Exception as exc:
            failed_count += 1
            consecutive_rate_limit_failures = 0
            _append_failed_id(raw_review_id)
            print(f"Failed for raw_review_id {raw_review_id}: {exc}")

        avg_total_tokens = (
            usage_total_tokens_sum / usage_request_count
            if usage_request_count
            else 0
        )

        is_progress_checkpoint = (
            (classified_attempts > 0 and classified_attempts % progress_every == 0)
            or (max_classifications is not None and classified_attempts >= max_classifications)
            or index == total_rows
        )
        if is_progress_checkpoint:
            print(
                f"Progress: {index}/{total_rows} rows checked | "
                f"classified: {classified_attempts}"
                + (
                    f"/{max_classifications}" if max_classifications is not None else ""
                )
                + f" | succeeded: {succeeded_count} | failed: {failed_count} | skipped: {skipped_count} | "
                f"avg_total_tokens/request: {avg_total_tokens:.2f}"
            )

        if max_classifications is not None and classified_attempts >= max_classifications:
            break

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    print("Summary:")
    print(f"Total classified attempts: {classified_attempts}")
    print(f"Succeeded: {succeeded_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    final_avg_total_tokens = (
        usage_total_tokens_sum / usage_request_count
        if usage_request_count
        else 0
    )
    print(f"Average total_tokens/request: {final_avg_total_tokens:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify raw reviews and store labels.")
    parser.add_argument("--limit", type=int, default=None, help="Max number of reviews to classify this run")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N classified reviews")
    parser.add_argument("--sleep-seconds", type=float, default=5, help="Delay between classifications")
    parser.add_argument("--retries", type=int, default=3, help="Groq call attempts per review")
    parser.add_argument("--rate-limit-delay", type=float, default=60, help="Retry delay in seconds for 429 responses")
    parser.add_argument("--ignore-daily-limit-stop", action="store_true", help="Do not stop after 10 consecutive 429 failures")
    args = parser.parse_args()

    classify_and_store_all(
        max_classifications=args.limit,
        progress_every=max(1, args.progress_every),
        sleep_seconds=max(0, args.sleep_seconds),
        retries=max(1, args.retries),
        rate_limit_delay=max(0, args.rate_limit_delay),
        stop_on_daily_limit=not args.ignore_daily_limit_stop,
    )
