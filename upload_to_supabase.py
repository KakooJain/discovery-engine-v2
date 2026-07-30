import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL must be set in .env")

if not SUPABASE_SERVICE_ROLE_KEY and not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY must be set in .env")

supabase_key = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY
supabase = create_client(SUPABASE_URL, supabase_key)

DATA_PATH = Path("data/clean/reviews.json")


def _extract_failure_reason(response=None, exc=None):
    if exc is not None:
        return str(exc)
    if response is None:
        return "No response returned"

    error = getattr(response, "error", None)
    if error is not None:
        if hasattr(error, "message"):
            return str(error.message)
        if isinstance(error, dict):
            return json.dumps(error, sort_keys=True)
        return str(error)

    data = getattr(response, "data", None)
    if data:
        return None
    return "No inserted data returned"


def upload_reviews():
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    success_count = 0
    failure_count = 0
    failure_reasons = {}

    for index, record in enumerate(records, start=1):
        payload = {
            "source": record.get("source"),
            "app": record.get("app"),
            "text": record.get("text"),
            "rating": record.get("rating"),
            "review_date": record.get("date"),
        }

        try:
            response = supabase.table("raw_reviews").insert(payload).execute()
            data = getattr(response, "data", None)
            if data:
                success_count += 1
            else:
                failure_count += 1
                reason = _extract_failure_reason(response=response)
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                print(f"[FAIL {index}] {reason}")
        except Exception as exc:
            failure_count += 1
            reason = _extract_failure_reason(exc=exc)
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            print(f"[FAIL {index}] {reason}")

        if index % 100 == 0:
            print(f"Processed {index} records...")

    print("Summary:")
    print(f"Inserted successfully: {success_count}")
    print(f"Failed: {failure_count}")
    print("Unique failure reasons:")
    for reason, count in sorted(failure_reasons.items(), key=lambda item: item[0].lower()):
        print(f"- {reason} (count={count})")


if __name__ == "__main__":
    upload_reviews()
