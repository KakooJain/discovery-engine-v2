import json
import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import create_client

from app import _fetch_all_rows, _normalize_analyzed_row
from research_pipeline import run_hybrid_retrieval


def precision_at_k(predicted_ids, expected_ids, k):
    if k <= 0:
        return 0.0
    top = predicted_ids[:k]
    if not top:
        return 0.0
    expected = set(expected_ids)
    hits = sum(1 for rid in top if rid in expected)
    return hits / float(k)


def load_eval_set(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_reviews():
    select_cols = "*,raw_reviews(text,app,source,rating)"
    rows = _fetch_all_rows("analyzed_reviews", select_cols)
    return [_normalize_analyzed_row(row) for row in rows]


def evaluate_case(case, reviews):
    question = case["question"]
    result = run_hybrid_retrieval(question, [dict(r) for r in reviews])
    top_ids = [r.get("id") for r in result.top_records if r.get("id") is not None]

    p5 = precision_at_k(top_ids, case.get("expected_relevant_record_ids", []), 5)
    p10 = precision_at_k(top_ids, case.get("expected_relevant_record_ids", []), 10)

    blocked_domains = set(case.get("excluded_domains", []))
    excluded_hits = [r for r in result.top_records[:15] if str(r.get("feedback_domain")) in blocked_domains]

    return {
        "question": question,
        "intents_detected": result.intents,
        "sufficiency": result.sufficiency,
        "top_15_ids": top_ids[:15],
        "precision_at_5": round(p5, 3),
        "precision_at_10": round(p10, 3),
        "excluded_domain_hits_top15": [
            {
                "id": row.get("id"),
                "feedback_domain": row.get("feedback_domain"),
                "research_relevance": row.get("research_relevance"),
                "score": row.get("final_reranking_score"),
            }
            for row in excluded_hits
        ],
    }


def main():
    root = Path(__file__).resolve().parent
    eval_path = root / "eval" / "research_eval_set.json"

    load_dotenv(root / ".env")
    if not os.getenv("SUPABASE_URL"):
        raise RuntimeError("SUPABASE_URL missing")

    _ = create_client(
        os.environ["SUPABASE_URL"],
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_ANON_KEY"],
    )

    dataset = load_eval_set(eval_path)
    reviews = load_reviews()

    reports = [evaluate_case(case, reviews) for case in dataset]

    print(json.dumps({"reports": reports}, indent=2))


if __name__ == "__main__":
    main()
