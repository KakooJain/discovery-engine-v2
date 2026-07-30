import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL must be set in .env")

if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError(
        "SUPABASE_SERVICE_ROLE_KEY must be set in .env to reset Supabase tables. "
        "Use a service role key only in trusted environments."
    )

HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Accept": "application/json",
}

TABLES = ["analyzed_reviews", "raw_reviews"]


def request(method, endpoint, headers=None, body=None):
    data = None
    if body is not None:
        data = body.encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, method=method)
    for name, value in {**HEADERS, **(headers or {})}.items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, dict(resp.getheaders()), resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode("utf-8")


def count_rows(table):
    endpoint = f"{SUPABASE_URL}/rest/v1/{table}?select=id"
    status, headers, body = request("GET", endpoint, {"Prefer": "count=exact"})
    if status not in (200, 206):
        raise RuntimeError(f"Failed to count rows for {table}: {status} {body}")
    return headers.get("Content-Range", "unknown")


def delete_table(table):
    endpoint = f"{SUPABASE_URL}/rest/v1/{table}?id=not.is.null"
    status, headers, body = request("DELETE", endpoint, {"Prefer": "return=minimal"})
    if status not in (200, 204):
        raise RuntimeError(f"Delete failed for {table}: {status} {body}")
    return status, headers, body


if __name__ == "__main__":
    for table in TABLES:
        print(f"=== {table} ===")
        before = count_rows(table)
        print(f"before: {before}")
        status, headers, body = delete_table(table)
        print(f"delete status: {status}")
        print(f"delete body length: {len(body)}")
        after = count_rows(table)
        print(f"after: {after}\n")
