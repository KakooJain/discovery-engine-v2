import os
import json
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client

from retrieve import retrieve
from research_pipeline import (
    build_debug_payload,
    build_insufficient_evidence_response,
    build_synthesis_prompt,
    run_hybrid_retrieval,
)

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY must be set in .env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env")

client = Groq(api_key=GROQ_API_KEY, max_retries=1)
supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

app = Flask(__name__, template_folder="templates")
ASK_DEBUG_ENABLED = os.getenv("ASK_DEBUG_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
ASK_MODEL = os.getenv("ASK_MODEL", "llama-3.1-8b-instant")


class AskWorkflowError(Exception):
    def __init__(self, status_code, error_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


def _structured_ask_response(
    status,
    question,
    message,
    answer,
    sources_used,
    records_searched,
    evidence,
    error_code=None,
    confidence_level="LOW",
    selected_tag_filters=None,
    research_intents=None,
    sufficiency=None,
    provisional=False,
    corpus_status=None,
    claim_validation=None,
    retrieval_failed=False,
    debug_payload=None,
):
    payload = {
        "status": status,
        "question": question,
        "message": message,
        "error_code": error_code,
        "answer": answer,
        "sources_used": sources_used,
        "records_searched": records_searched,
        "records_used": len(evidence or []),
        "evidence": evidence or [],
        "confidence_level": confidence_level,
        "selected_tag_filters": selected_tag_filters or {},
        "research_intents": research_intents or [],
        "sufficiency": sufficiency or {},
        "provisional": bool(provisional),
        "corpus_status": corpus_status or {},
        "claim_validation": claim_validation or [],
        "retrieval_failed": bool(retrieval_failed),
        "fallback_used": False,
        "tag_matched_count": (sufficiency or {}).get("directly_relevant_records", 0),
        "total_retrieved_count": records_searched,
    }
    if debug_payload is not None:
        payload["debug"] = debug_payload
    return payload


def _is_quota_or_rate_limit_error(exc):
    text = str(exc or "").lower()
    return any(token in text for token in ["429", "quota", "rate limit", "rate_limit", "daily_limit_all_gemini_models_exhausted"])


def _is_timeout_error(exc):
    text = str(exc or "").lower()
    return any(token in text for token in ["timeout", "timed out", "deadline exceeded"])


def _is_model_unavailable_error(exc):
    text = str(exc or "").lower()
    return any(token in text for token in ["service unavailable", "temporarily unavailable", "model unavailable", "503"])


def _to_evidence_rows(records):
    evidence_rows = []
    for rec in records or []:
        evidence_rows.append(
            {
                "id": rec.get("id"),
                "app": rec.get("app"),
                "source": rec.get("source"),
                "text": str(rec.get("text", "") or "").strip(),
                "supporting_quote": str(rec.get("supporting_quote", "") or "").strip(),
                "research_relevance": rec.get("research_relevance"),
                "feedback_domain": rec.get("feedback_domain"),
                "final_reranking_score": rec.get("final_reranking_score"),
            }
        )
    return evidence_rows


def _build_limited_top_themes(evidence_rows):
    theme_rules = [
        (
            "Search and known-item purchase",
            ["search", "showing", "find", "reorder", "same", "known", "repeat"],
        ),
        (
            "Trust and authenticity concerns",
            ["trust", "authentic", "fake", "risky", "quality", "genuine"],
        ),
        (
            "Lack of detailed product information",
            ["detail", "information", "description", "ingredient", "review", "photo"],
        ),
        (
            "Price and value uncertainty",
            ["price", "expensive", "discount", "value", "cost"],
        ),
        (
            "Assortment and availability gaps",
            ["missing", "not available", "stock", "variety", "option"],
        ),
    ]

    counts = {label: 0 for label, _ in theme_rules}
    for row in evidence_rows:
        text = str(row.get("text", "") or "").lower()
        quote = str(row.get("supporting_quote", "") or "").lower()
        combined = f"{text} {quote}"
        for label, cues in theme_rules:
            if any(cue in combined for cue in cues):
                counts[label] += 1

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    selected = [label for label, count in ranked if count > 0][:3]
    if selected:
        return selected

    return [
        "Search and known-item purchase",
        "Trust and authenticity concerns",
        "Lack of detailed product information",
    ]


def _build_limited_supporting_excerpts(evidence_rows, max_items=3):
    excerpts = []
    for row in evidence_rows:
        quote = str(row.get("supporting_quote", "") or "").strip()
        text = str(row.get("text", "") or "").strip().replace("\n", " ")
        excerpt = quote or text
        if not excerpt:
            continue
        excerpt = excerpt[:160].strip()
        if excerpt in excerpts:
            continue
        excerpts.append(excerpt)
        if len(excerpts) >= max_items:
            break
    return excerpts


def _build_limited_answer(evidence_rows):
    relevant_count = len(evidence_rows)
    themes = _build_limited_top_themes(evidence_rows)
    excerpts = _build_limited_supporting_excerpts(evidence_rows, max_items=3)

    lines = []
    lines.append("Synthesis temporarily unavailable")
    lines.append("")
    lines.append(f"{relevant_count} relevant records retrieved")
    lines.append("")
    lines.append("Top matching themes:")
    for theme in themes:
        lines.append(f"• {theme}")
    lines.append("")
    lines.append("Supporting evidence:")
    if excerpts:
        for excerpt in excerpts:
            lines.append(f"[{excerpt}]")
    else:
        lines.append("[Review excerpt]")

    return "\n".join(lines)

TAG_FIELD_MAP = {
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

TAG_FIELDS = list(TAG_FIELD_MAP.keys())

SYSTEM_PROMPT = (
    "---\n"
    "You are a senior user researcher briefing Blinkit's product team. You analyse quick-commerce user feedback and produce sharp, decision-useful insights.\n\n"
    "OUTPUT FORMAT — follow exactly:\n\n"
    "**Insight 1: [Short, specific, claim-style headline]**\n"
    "[2-3 sentences explaining the pattern. Reference how many reviews support it using the stats provided. Be concrete.]\n"
    "> \"[verbatim user quote]\"\n"
    "> \"[verbatim user quote]\"\n"
    "> \"[verbatim user quote]\"\n"
    "[...up to five total, only as many as the evidence genuinely supports]\n\n"
    "**Insight 2: ...** [headline, summary, up to five supporting quotes]\n\n"
    "**Insight 3: ...** [headline, summary, up to five supporting quotes]\n\n"
    "RULES:\n"
    "- NEVER write \"review 6\", \"review 10\", or any numeric review reference. Quote users verbatim instead.\n"
    "- Every insight must be grounded in the supplied evidence. Do not invent patterns.\n"
    "- Use quantitative evidence, but do not use retrieved-set prevalence to claim how common behavior is in the overall user base.\n"
    "- Headlines must be claims, not topics. Write \"[A specific claim about user behaviour, not a topic label]\" not \"Habits and routines\".\n"
    "- Quotes must be copied exactly from the evidence, including Hinglish. Trim to the most telling fragment.\n"
    "- If fewer than five distinct reviews in the retrieved set genuinely support an insight, give only the ones that do and state the actual number. Never invent, pad, reuse the same quote twice, or stretch an unrelated quote to fit.\n"
    "- Each quote must independently support the specific claim in that insight's headline; do not include a quote merely because it appears in the retrieved set. Two or three strongly relevant quotes are better than five loosely related ones. Never reuse a quote across insights. If a quote is about a different topic than the headline claims, exclude it.\n"
    "- Quote cap by evidence size: if RETRIEVED_SET_SIZE is fewer than 10, include at most 2 quotes per insight; if fewer than 20, include at most 3; otherwise up to 5. Prefer fewer strong quotes over filling slots.\n"
    "- Quote uniqueness is mandatory: a quote may appear under only one insight in the entire answer. Do not repeat or lightly rephrase the same quote under multiple insights.\n"
    "- If the evidence does not support an insight, say so plainly rather than stretching unrelated complaints (e.g. pricing or delivery) into discovery conclusions.\n"
    "- If evidence genuinely doesn't support three distinct insights, give fewer.\n"
    "- Any statement about how common a pattern is MUST use CLASSIFIED CORPUS count and percentage together (for example, '412 of 2,961, 13.9%').\n"
    "- Retrieved-set counts are allowed only to describe quote evidence volume (for example, 'quotes below come from 6 retrieved reviews'); never use retrieved-set counts to claim prevalence.\n"
    "- Do not present an insight supported by fewer than 3 reviews.\n"
    "- No preamble. Start directly with Insight 1.\n"
    "---"
)


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


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_app_name(value):
    app_name = str(value or "").strip().lower()
    if "blinkit" in app_name:
        return "Blinkit"
    if "instamart" in app_name:
        return "Instamart"
    if "zepto" in app_name:
        return "Zepto"
    return "Other"


def _normalize_source_name(value):
    source_name = str(value or "").strip().lower()
    if "play" in source_name:
        return "play_store"
    if "app" in source_name:
        return "app_store"
    return "unknown"


def _normalize_tag_array(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _count_tag_field(rows, field_name):
    counts = {}
    for row in rows:
        for tag in _normalize_tag_array(row.get(field_name)):
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def _build_bar_items(counts, denominator):
    items = []
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        pct = (count / denominator * 100) if denominator else 0
        items.append({"label": label, "count": count, "percentage": round(pct, 1)})
    return items


def _build_stats_block(reviews):
    retrieved_count = len(reviews)

    corpus_rows = _fetch_all_rows(
        "analyzed_reviews",
        "behavioral_driver,discovery_barrier,discovery_channel,frustration_type,segment_marker,unmet_need,categories,sentiment_score",
    )
    corpus_count = len(corpus_rows)

    retrieved_tag_counts = {}
    corpus_tag_counts = {}

    for field in TAG_FIELDS:
        corpus_field_counts = {}
        for row in corpus_rows:
            values = row.get(field, [])
            if not isinstance(values, list):
                continue
            for tag in values:
                tag_name = str(tag).strip()
                if not tag_name:
                    continue
                corpus_field_counts[tag_name] = corpus_field_counts.get(tag_name, 0) + 1
                corpus_tag_counts[tag_name] = corpus_tag_counts.get(tag_name, 0) + 1

    tag_frequency_counts = {}
    for field in TAG_FIELDS:
        field_counts = {}
        for review in reviews:
            tags = review.get("tags", {}) if isinstance(review.get("tags", {}), dict) else {}
            values = tags.get(field, [])
            if not isinstance(values, list):
                continue
            for tag in values:
                tag_name = str(tag).strip()
                if not tag_name:
                    continue
                field_counts[tag_name] = field_counts.get(tag_name, 0) + 1
                retrieved_tag_counts[tag_name] = retrieved_tag_counts.get(tag_name, 0) + 1
        tag_frequency_counts[field] = dict(sorted(field_counts.items(), key=lambda item: (-item[1], item[0])))

    sentiment_distribution = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
    for review in reviews:
        value = review.get("sentiment_score")
        if value is None:
            continue
        try:
            score = int(value)
        except (TypeError, ValueError):
            continue
        if score in sentiment_distribution:
            sentiment_distribution[score] += 1

    sorted_flat_counts = sorted(
        retrieved_tag_counts.items(),
        key=lambda item: (-item[1], -corpus_tag_counts.get(item[0], 0), item[0]),
    )

    lines = [f"RETRIEVED EVIDENCE: {retrieved_count} reviews total"]
    lines.append(f"TAG COUNTS (out of these {retrieved_count} retrieved | {corpus_count} classified corpus):")
    if sorted_flat_counts:
        for tag, count in sorted_flat_counts:
            corpus_tag_count = corpus_tag_counts.get(tag, 0)
            corpus_pct = (corpus_tag_count / corpus_count * 100) if corpus_count else 0
            lines.append(
                f"  {tag}: {count} of {retrieved_count} retrieved | {corpus_tag_count} of {corpus_count} classified corpus ({corpus_pct:.1f}%)"
            )
    else:
        lines.append("  none: 0 of 0 retrieved | 0 of 0 classified corpus (0.0%)")

    lines.append(
        f"SENTIMENT (out of these {retrieved_count}): "
        f"-2:{sentiment_distribution[-2]}, -1:{sentiment_distribution[-1]}, 0:{sentiment_distribution[0]}, +1:{sentiment_distribution[1]}, +2:{sentiment_distribution[2]}"
    )
    lines.append(f"CLASSIFIED_CORPUS_TOTAL={corpus_count}")
    lines.append("RAW_TAG_COUNTS_JSON=" + json.dumps(tag_frequency_counts, ensure_ascii=True))
    lines.append(
        "RAW_SENTIMENT_JSON="
        + json.dumps({str(k): v for k, v in sentiment_distribution.items()}, ensure_ascii=True)
    )
    return "\n".join(lines)


def _build_evidence_payload(question, reviews, retrieved_set_size=None):
    if retrieved_set_size is None:
        retrieved_set_size = len(reviews)

    lines = [
        f"Question: {question}",
        f"RETRIEVED_SET_SIZE={retrieved_set_size}",
        _build_stats_block(reviews),
        "Evidence:",
    ]
    for index, review in enumerate(reviews, start=1):
        text = review.get("text", "").replace("\n", " ").strip()
        source = review.get("app", "unknown")
        quote = str(review.get("supporting_quote", "") or "").strip()
        tags = review.get("tags", {}) if isinstance(review.get("tags", {}), dict) else {}
        non_empty_tag_parts = []
        for field in TAG_FIELDS:
            values = tags.get(field, [])
            if isinstance(values, list) and values:
                non_empty_tag_parts.append(f"{field}={','.join(values)}")
        tag_summary = " | ".join(non_empty_tag_parts) if non_empty_tag_parts else "none"
        lines.append(
            f"{index}. [{source}] text={text} | tags={tag_summary} | supporting_quote={quote or 'n/a'}"
        )
    return "\n".join(lines)


def _ask_groq(question, reviews, retrieved_set_size):
    return _ask_groq_with_extra_rule(question, reviews, "", retrieved_set_size)


def _ask_groq_with_extra_rule(question, reviews, extra_rule, retrieved_set_size):
    user_payload = _build_evidence_payload(question, reviews, retrieved_set_size)
    if extra_rule:
        user_payload = user_payload + "\n\nValidator correction: " + extra_rule

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def _validate_generated_answer(answer):
    violations = []

    for match in re.finditer(r"\b(\d+)\s+of\s+(\d+)\b", answer):
        numerator = int(match.group(1))
        denominator = int(match.group(2))
        if numerator > denominator:
            violations.append(
                f"Invalid X of Y statement: {numerator} of {denominator} (numerator exceeds denominator)."
            )

    if re.search(r"\breview\s+\d+\b", answer, flags=re.IGNORECASE):
        violations.append("Numeric review references detected (e.g., 'review 6').")

    insight_sections = re.split(r"\*\*Insight\s+\d+:", answer)
    for section in insight_sections[1:]:
        section_text = section
        section_lines = [line for line in section_text.splitlines() if not line.lstrip().startswith(">")]
        section_text = "\n".join(section_lines)
        counts = [(int(m.group(1)), int(m.group(2))) for m in re.finditer(r"\b(\d+)\s+of\s+(\d+)\b", section_text)]
        if counts and max(n for n, _ in counts) < 3:
            violations.append("An insight cites fewer than 3 supporting reviews.")

    deduped = []
    seen = set()
    for violation in violations:
        if violation in seen:
            continue
        seen.add(violation)
        deduped.append(violation)
    return deduped


def _format_validator_retry_rule(violations):
    return (
        "Your previous answer violated these rules: "
        + " ; ".join(violations)
        + " . Rewrite the answer fully, satisfy every rule exactly, and keep only evidence-grounded claims."
    )


def _public_ask_error_message(exc):
    text = str(exc or "").lower()
    if "429" in text or "rate limit" in text or "quota" in text or "rate_limit" in text:
        return (
            "The language model's usage limit has been reached. "
            "Answers will be available again shortly - the dashboard below is unaffected and still fully usable."
        )

    connection_cues = [
        "timeout",
        "timed out",
        "connection",
        "connect",
        "temporarily unavailable",
        "service unavailable",
        "network",
        "dns",
    ]
    if any(cue in text for cue in connection_cues):
        return "Couldn't reach the language model. Please try again in a moment."

    return "Something went wrong generating this answer. Please try again."


def _parse_json_object(text):
    candidate = str(text or "").strip()
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not match:
            return {}
        return json.loads(match.group(0))


def _normalize_selector_output(payload):
    if not isinstance(payload, dict):
        return []

    selections = payload.get("selections")
    if not isinstance(selections, list):
        return []

    normalized = []
    seen = set()

    for item in selections[:2]:
        if not isinstance(item, dict):
            continue
        field = str(item.get("field", "")).strip()
        if field not in TAG_FIELD_MAP:
            continue
        tags = item.get("tags")
        if not isinstance(tags, list):
            continue
        allowed = TAG_FIELD_MAP[field]
        cleaned_tags = [str(tag).strip() for tag in tags if str(tag).strip() in allowed]
        cleaned_tags = sorted(set(cleaned_tags))[:4]
        if not cleaned_tags:
            continue
        key = (field, tuple(cleaned_tags))
        if key in seen:
            continue
        seen.add(key)
        normalized.append({"field": field, "tags": cleaned_tags})

    return normalized


def pick_retrieval_targets(question):
    selector_prompt = (
        "You map a research question to taxonomy tags. "
        "Select the 1-2 most relevant fields and specific tags for retrieval. "
        "Choose only from the allowed taxonomy values below. "
        "Return ONLY JSON in this shape: "
        "{\"selections\":[{\"field\":\"...\",\"tags\":[\"...\"]}]} with max 2 selections and max 4 tags per selection.\n\n"
        f"Allowed fields and tags: {json.dumps({k: sorted(list(v)) for k, v in TAG_FIELD_MAP.items()})}"
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": selector_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    parsed = _parse_json_object(response.choices[0].message.content)
    normalized = _normalize_selector_output(parsed)
    if normalized:
        return normalized

    # Safe default if selector fails: questions about discovery usually map here.
    return [{"field": "discovery_barrier", "tags": sorted(list(TAG_FIELD_MAP["discovery_barrier"]))[:3]}]


def _extract_raw_review(row):
    raw = row.get("raw_reviews")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, dict):
        return raw
    return {}


def _normalize_analyzed_row(row):
    raw = _extract_raw_review(row)
    tags = {}
    for field in TAG_FIELDS:
        values = row.get(field)
        tags[field] = [str(value).strip() for value in values] if isinstance(values, list) else []

    return {
        "id": row.get("raw_review_id"),
        "text": str(raw.get("text", "") or ""),
        "app": str(raw.get("app", "") or "unknown"),
        "source": _normalize_source_name(raw.get("source")),
        "supporting_quote": str(row.get("supporting_quote", "") or ""),
        "sentiment_score": row.get("sentiment_score"),
        "research_relevance": str(row.get("research_relevance", "") or ""),
        "feedback_domain": str(row.get("feedback_domain", "") or ""),
        "behavioral_schema": {
            "shopping_mission": str(row.get("shopping_mission", "") or "not_stated"),
            "habit_signal": str(row.get("habit_signal", "") or "not_stated"),
            "current_category": str(row.get("current_category", "") or "not_stated"),
            "category_tried": str(row.get("category_tried", "") or "not_stated"),
            "discovery_method": str(row.get("discovery_method", "") or "not_stated"),
            "exploration_barrier": str(row.get("exploration_barrier", "") or "not_stated"),
            "purchase_trigger": str(row.get("purchase_trigger", "") or "not_stated"),
            "information_needed": str(row.get("information_needed", "") or "not_stated"),
            "trust_signal": str(row.get("trust_signal", "") or "not_stated"),
            "perceived_risk": str(row.get("perceived_risk", "") or "not_stated"),
            "workaround": str(row.get("workaround", "") or "not_stated"),
            "user_context": str(row.get("user_context", "") or "not_stated"),
            "unmet_need": str(row.get("unmet_need_detail", "") or "not_stated"),
            "evidence_quote": str(row.get("evidence_quote", "") or "not_stated"),
            "classification_confidence": str(row.get("classification_confidence", "") or "medium"),
        },
        "tags": tags,
    }


def _fetch_all_analyzed_reviews_for_ask():
    select_cols = (
        "*,"
        "raw_reviews(text,app,source,rating)"
    )
    rows = _fetch_all_rows("analyzed_reviews", select_cols)
    return [_normalize_analyzed_row(row) for row in rows]


def _build_claim_validator_prompt(question, answer_text, records):
    evidence_lines = []
    for rec in records:
        rec_text = str(rec.get("text", "") or "").replace("\n", " ").strip()
        evidence_lines.append(f"id={rec.get('id')} | text={rec_text}")
    evidence_blob = "\n".join(evidence_lines)

    return (
        "Validate claims against evidence. Return ONLY JSON with key claims where each claim has: "
        "claim_text, supporting_record_ids, support_level(one_of strong|partial|none), contradiction_status(one_of contradicted|not_contradicted). "
        "Use only supplied IDs.\n\n"
        f"Question: {question}\n\n"
        f"Draft answer:\n{answer_text}\n\n"
        f"Evidence:\n{evidence_blob}"
    )


def _validate_claims_with_model(question, answer_text, records):
    if not records or not answer_text.strip():
        return []

    prompt = _build_claim_validator_prompt(question, answer_text, records)
    response = client.chat.completions.create(
        model=ASK_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a strict evidence validator.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = _parse_json_object(response.choices[0].message.content)
    claims = parsed.get("claims") if isinstance(parsed, dict) else []
    if not isinstance(claims, list):
        return []
    normalized = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        normalized.append(
            {
                "claim_text": str(claim.get("claim_text", "") or "").strip(),
                "supporting_record_ids": [rid for rid in claim.get("supporting_record_ids", []) if rid is not None],
                "support_level": str(claim.get("support_level", "none") or "none"),
                "contradiction_status": str(claim.get("contradiction_status", "not_contradicted") or "not_contradicted"),
            }
        )
    return normalized


def _generate_research_answer(question, records, sufficiency, classified_total, pending_count):
    if sufficiency.get("state") == "insufficient":
        return build_insufficient_evidence_response(question, sufficiency), []

    prompt = build_synthesis_prompt(question, records, sufficiency, classified_total, pending_count)
    completion = client.chat.completions.create(
        model=ASK_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research analyst. Answer only from provided evidence. "
                    "If evidence is limited, say so directly."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )
    answer = (completion.choices[0].message.content if completion.choices else "") or ""
    answer = answer.strip()
    claims = _validate_claims_with_model(question, answer, records)

    weak_claims = [
        claim for claim in claims
        if claim.get("support_level") == "none"
        or claim.get("contradiction_status") == "contradicted"
        or len(claim.get("supporting_record_ids", [])) < 2
    ]

    if weak_claims:
        weak_summary = "; ".join(claim.get("claim_text", "")[:120] for claim in weak_claims[:5])
        rewrite_prompt = (
            prompt
            + "\n\nRe-write the answer by removing or softening unsupported claims. "
            + "Unsupported claims detected: "
            + weak_summary
            + ". Ensure each major claim has at least 2 supporting record IDs."
        )
        completion_retry = client.chat.completions.create(
            model=ASK_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict evidence-grounded analyst.",
                },
                {
                    "role": "user",
                    "content": rewrite_prompt,
                },
            ],
            temperature=0,
        )
        answer = (completion_retry.choices[0].message.content if completion_retry.choices else answer) or answer
        answer = answer.strip()
        claims = _validate_claims_with_model(question, answer, records)

    return answer, claims


def _count_failed_records():
    failed_path = Path(__file__).resolve().parent / "failed_ids.txt"
    if not failed_path.exists():
        return 0
    unique_ids = set()
    with failed_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                unique_ids.add(value)
    return len(unique_ids)


def _rank_reviews_tfidf(question, reviews, top_n=40):
    valid = [review for review in reviews if str(review.get("text", "")).strip()]
    if not valid:
        return []

    texts = [review.get("text", "") for review in valid]
    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([question])
    similarities = cosine_similarity(query_vector, doc_matrix).ravel()
    ranked_indices = similarities.argsort()[::-1][: min(top_n, len(valid))]
    return [valid[index] for index in ranked_indices]


def _fetch_by_field_overlap(field, tags):
    if not tags:
        return []
    select_cols = (
        "raw_review_id,behavioral_driver,discovery_barrier,discovery_channel,"
        "frustration_type,segment_marker,unmet_need,categories,sentiment_score,"
        "supporting_quote,raw_reviews(text,app)"
    )
    response = (
        supabase.table("analyzed_reviews")
        .select(select_cols)
        .overlaps(field, tags)
        .execute()
    )
    rows = getattr(response, "data", None) or []
    return [_normalize_analyzed_row(row) for row in rows]


def retrieve_tagged_reviews(question, selections):
    merged = {}
    for selection in selections:
        field = selection.get("field")
        tags = selection.get("tags", [])
        if field not in TAG_FIELD_MAP or not isinstance(tags, list):
            continue
        for review in _fetch_by_field_overlap(field, tags):
            review_id = review.get("id")
            if review_id is None:
                continue
            merged[review_id] = review
    return _rank_reviews_tfidf(question, list(merged.values()), top_n=40)


def retrieve_fallback_reviews(question):
    select_cols = (
        "raw_review_id,behavioral_driver,discovery_barrier,discovery_channel,"
        "frustration_type,segment_marker,unmet_need,categories,sentiment_score,"
        "supporting_quote,raw_reviews(text,app)"
    )
    all_analyzed_rows = _fetch_all_rows("analyzed_reviews", select_cols)
    all_analyzed_reviews = [_normalize_analyzed_row(row) for row in all_analyzed_rows]
    ranked = _rank_reviews_tfidf(question, all_analyzed_reviews, top_n=40)
    if ranked:
        return ranked

    # Last-resort fallback to existing corpus retriever.
    raw_ranked = retrieve(question, top_n=40)
    fallback_rows = []
    for row in raw_ranked:
        fallback_rows.append(
            {
                "id": row.get("id"),
                "text": str(row.get("text", "") or ""),
                "app": str(row.get("app", "") or "unknown"),
                "supporting_quote": "",
                "sentiment_score": None,
                "tags": {field: [] for field in TAG_FIELDS},
            }
        )
    return fallback_rows


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
@app.route("/healthz")
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True)
    if not payload or "question" not in payload:
        response_payload = _structured_ask_response(
            status="error",
            question="",
            message="Missing question field",
            answer=None,
            sources_used=None,
            records_searched=0,
            evidence=[],
            error_code="invalid_question",
            retrieval_failed=True,
        )
        return jsonify(response_payload), 400

    question = str(payload.get("question", "")).strip()
    if not question:
        response_payload = _structured_ask_response(
            status="error",
            question=question,
            message="Question cannot be empty",
            answer=None,
            sources_used=None,
            records_searched=0,
            evidence=[],
            error_code="invalid_question",
            retrieval_failed=True,
        )
        return jsonify(response_payload), 400

    try:
        debug_requested = bool(payload.get("debug", False)) and ASK_DEBUG_ENABLED

        all_reviews = _fetch_all_analyzed_reviews_for_ask()
        retrieval_result = run_hybrid_retrieval(question, all_reviews)

        top_records = retrieval_result.top_records
        evidence_rows = _to_evidence_rows(top_records)
        total_retrieved_count = len(retrieval_result.candidates)
        records_used = len(top_records)
        classified_total = len(all_reviews)
        total_raw_reviews = len(_fetch_all_rows("raw_reviews", "id"))
        pending_count = max(0, total_raw_reviews - classified_total)
        failed_count = _count_failed_records()

        try:
            answer, claim_checks = _generate_research_answer(
                question,
                top_records,
                retrieval_result.sufficiency,
                classified_total,
                pending_count,
            )
            if not isinstance(answer, str):
                raise AskWorkflowError(
                    status_code=503,
                    error_code="malformed_response",
                    message="Synthesis service returned an invalid response format.",
                )
            if claim_checks is None:
                claim_checks = []
        except Exception as llm_exc:
            debug_payload = build_debug_payload(retrieval_result) if debug_requested else None
            corpus_status = {
                "total_raw_records": total_raw_reviews,
                "classified_records": classified_total,
                "pending_records": pending_count,
                "failed_records": failed_count,
                "percentage_classified": round((classified_total / total_raw_reviews * 100), 1) if total_raw_reviews else 0,
            }

            if isinstance(llm_exc, AskWorkflowError):
                response_payload = _structured_ask_response(
                    status="error",
                    question=question,
                    message=llm_exc.message,
                    answer=None,
                    sources_used=records_used,
                    records_searched=total_retrieved_count,
                    evidence=evidence_rows,
                    error_code=llm_exc.error_code,
                    confidence_level="LOW",
                    selected_tag_filters=retrieval_result.selected_filters,
                    research_intents=retrieval_result.intents,
                    sufficiency=retrieval_result.sufficiency,
                    provisional=pending_count > 0,
                    corpus_status=corpus_status,
                    retrieval_failed=False,
                    debug_payload=debug_payload,
                )
                return jsonify(response_payload), llm_exc.status_code

            if _is_quota_or_rate_limit_error(llm_exc):
                limited_answer = _build_limited_answer(evidence_rows)
                response_payload = _structured_ask_response(
                    status="limited",
                    question=question,
                    message="The language-model quota has been reached. Retrieved evidence is shown below, but synthesis is temporarily unavailable.",
                    answer=limited_answer,
                    sources_used=records_used,
                    records_searched=total_retrieved_count,
                    evidence=evidence_rows,
                    error_code="llm_quota_exceeded",
                    confidence_level="LOW",
                    selected_tag_filters=retrieval_result.selected_filters,
                    research_intents=retrieval_result.intents,
                    sufficiency=retrieval_result.sufficiency,
                    provisional=pending_count > 0,
                    corpus_status=corpus_status,
                    retrieval_failed=False,
                    debug_payload=debug_payload,
                )
                return jsonify(response_payload), 429

            if _is_timeout_error(llm_exc):
                response_payload = _structured_ask_response(
                    status="error",
                    question=question,
                    message="Synthesis timed out. Retrieved evidence is shown below.",
                    answer=None,
                    sources_used=records_used,
                    records_searched=total_retrieved_count,
                    evidence=evidence_rows,
                    error_code="request_timeout",
                    confidence_level="LOW",
                    selected_tag_filters=retrieval_result.selected_filters,
                    research_intents=retrieval_result.intents,
                    sufficiency=retrieval_result.sufficiency,
                    provisional=pending_count > 0,
                    corpus_status=corpus_status,
                    retrieval_failed=False,
                    debug_payload=debug_payload,
                )
                return jsonify(response_payload), 408

            if _is_model_unavailable_error(llm_exc):
                response_payload = _structured_ask_response(
                    status="error",
                    question=question,
                    message="The model is temporarily unavailable. Retrieved evidence is shown below.",
                    answer=None,
                    sources_used=records_used,
                    records_searched=total_retrieved_count,
                    evidence=evidence_rows,
                    error_code="model_unavailable",
                    confidence_level="LOW",
                    selected_tag_filters=retrieval_result.selected_filters,
                    research_intents=retrieval_result.intents,
                    sufficiency=retrieval_result.sufficiency,
                    provisional=pending_count > 0,
                    corpus_status=corpus_status,
                    retrieval_failed=False,
                    debug_payload=debug_payload,
                )
                return jsonify(response_payload), 503

            response_payload = _structured_ask_response(
                status="error",
                question=question,
                message="The request could not be completed. Retrieved evidence is shown below.",
                answer=None,
                sources_used=records_used,
                records_searched=total_retrieved_count,
                evidence=evidence_rows,
                error_code="synthesis_failed",
                confidence_level="LOW",
                selected_tag_filters=retrieval_result.selected_filters,
                research_intents=retrieval_result.intents,
                sufficiency=retrieval_result.sufficiency,
                provisional=pending_count > 0,
                corpus_status=corpus_status,
                retrieval_failed=False,
                debug_payload=debug_payload,
            )
            return jsonify(response_payload), 500

        confidence_level = "LOW"
        if retrieval_result.sufficiency.get("state") == "sufficient":
            confidence_level = "HIGH"
        elif retrieval_result.sufficiency.get("state") == "limited":
            confidence_level = "MEDIUM"

        provisional = pending_count > 0
        if provisional:
            answer += "\n\nData status: provisional (classification is incomplete)."

        response_payload = _structured_ask_response(
            status="success",
            question=question,
            message="OK",
            answer=answer,
            sources_used=records_used,
            records_searched=total_retrieved_count,
            evidence=evidence_rows,
            error_code=None,
            confidence_level=confidence_level,
            selected_tag_filters=retrieval_result.selected_filters,
            research_intents=retrieval_result.intents,
            sufficiency=retrieval_result.sufficiency,
            provisional=provisional,
            corpus_status={
                "total_raw_records": total_raw_reviews,
                "classified_records": classified_total,
                "pending_records": pending_count,
                "failed_records": failed_count,
                "percentage_classified": round((classified_total / total_raw_reviews * 100), 1) if total_raw_reviews else 0,
            },
            claim_validation=claim_checks,
            retrieval_failed=False,
            debug_payload=build_debug_payload(retrieval_result) if debug_requested else None,
        )
        return jsonify(response_payload)
    except Exception as exc:
        app.logger.exception("/ask failed")
        response_payload = _structured_ask_response(
            status="error",
            question=question,
            message=_public_ask_error_message(exc),
            answer=None,
            sources_used=None,
            records_searched=0,
            evidence=[],
            error_code="internal_server_error",
            retrieval_failed=True,
        )
        return jsonify(response_payload), 500


@app.route("/dashboard-data", methods=["GET"])
def dashboard_data():
    raw_reviews = _fetch_all_rows("raw_reviews", "id,app,rating")
    analyzed_reviews = _fetch_all_rows(
        "analyzed_reviews",
        "raw_review_id,frustration_type,discovery_barrier,behavioral_driver,discovery_channel,unmet_need",
    )

    total_reviews = len(raw_reviews)
    reviewed_analyzed = len(analyzed_reviews)

    app_counts = {"Blinkit": 0, "Instamart": 0, "Zepto": 0}
    ratings = []
    for row in raw_reviews:
        app_name = _normalize_app_name(row.get("app"))
        if app_name in app_counts:
            app_counts[app_name] += 1

        rating_value = _to_float(row.get("rating"))
        if rating_value is not None:
            ratings.append(rating_value)

    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

    frustration_counts = _count_tag_field(analyzed_reviews, "frustration_type")
    barrier_counts = _count_tag_field(analyzed_reviews, "discovery_barrier")
    driver_counts = _count_tag_field(analyzed_reviews, "behavioral_driver")
    channel_counts = _count_tag_field(analyzed_reviews, "discovery_channel")
    unmet_need_counts = _count_tag_field(analyzed_reviews, "unmet_need")

    return jsonify(
        {
            "stats": {
                "total_reviews": total_reviews,
                "app_counts": app_counts,
                "average_rating": avg_rating,
                "reviews_analyzed": reviewed_analyzed,
            },
            "bars": {
                "top_frustrations": _build_bar_items(frustration_counts, reviewed_analyzed),
                "discovery_barriers": _build_bar_items(barrier_counts, reviewed_analyzed),
                "behavioral_drivers": _build_bar_items(driver_counts, reviewed_analyzed),
                "discovery_channels": _build_bar_items(channel_counts, reviewed_analyzed),
                "unmet_needs": _build_bar_items(unmet_need_counts, reviewed_analyzed),
            },
        }
    )


@app.route("/dashboard-data-v2", methods=["GET"])
def dashboard_data_v2():
    raw_reviews = _fetch_all_rows("raw_reviews", "id,app,rating,source,text")
    analyzed_reviews = _fetch_all_rows(
        "analyzed_reviews",
        "raw_review_id,behavioral_driver,discovery_barrier,discovery_channel,frustration_type,segment_marker,unmet_need,categories,sentiment_score,supporting_quote,analyzed_at",
    )

    total_reviews = len(raw_reviews)
    classified_count = len(analyzed_reviews)
    progress_pct = round((classified_count / total_reviews * 100), 1) if total_reviews else 0
    pending_count = max(0, total_reviews - classified_count)
    failed_count = _count_failed_records()

    ratings = []
    source_counts = {"play_store": 0, "app_store": 0, "unknown": 0}
    app_counts = {"Blinkit": 0, "Instamart": 0, "Zepto": 0, "Other": 0}
    for row in raw_reviews:
        rating_value = _to_float(row.get("rating"))
        if rating_value is not None:
            ratings.append(rating_value)

        normalized_source = _normalize_source_name(row.get("source"))
        source_counts[normalized_source] = source_counts.get(normalized_source, 0) + 1

        normalized_app = _normalize_app_name(row.get("app"))
        app_counts[normalized_app] = app_counts.get(normalized_app, 0) + 1

    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

    sentiment_distribution = {-2: 0, -1: 0, 0: 0, 1: 0, 2: 0}
    sentiment_values = []
    for row in analyzed_reviews:
        try:
            score = int(row.get("sentiment_score"))
        except (TypeError, ValueError):
            continue
        if score in sentiment_distribution:
            sentiment_distribution[score] += 1
            sentiment_values.append(score)

    avg_sentiment = round(sum(sentiment_values) / len(sentiment_values), 2) if sentiment_values else None

    source_total = source_counts["play_store"] + source_counts["app_store"]
    source_split = {
        "play_store": {
            "count": source_counts["play_store"],
            "percentage": round((source_counts["play_store"] / source_total * 100), 1) if source_total else 0,
            "label": "Play Store",
        },
        "app_store": {
            "count": source_counts["app_store"],
            "percentage": round((source_counts["app_store"] / source_total * 100), 1) if source_total else 0,
            "label": "App Store",
        },
    }

    sentiment_segments = []
    for score in [-2, -1, 0, 1, 2]:
        count = sentiment_distribution.get(score, 0)
        pct = round((count / classified_count * 100), 1) if classified_count else 0
        sentiment_segments.append({"score": score, "count": count, "percentage": pct})

    tag_panels = {}
    for field in TAG_FIELDS:
        field_counts = _count_tag_field(analyzed_reviews, field)
        items = _build_bar_items(field_counts, classified_count)
        tag_panels[field] = {
            "items": items,
            "empty": len(items) == 0,
            "total_tag_assignments": sum(field_counts.values()),
        }

    segment_tags = sorted(TAG_FIELD_MAP["segment_marker"])
    barrier_tags = sorted(TAG_FIELD_MAP["discovery_barrier"])
    matrix = {segment: {barrier: 0 for barrier in barrier_tags} for segment in segment_tags}
    non_zero_cells = 0

    for row in analyzed_reviews:
        segments = _normalize_tag_array(row.get("segment_marker"))
        barriers = _normalize_tag_array(row.get("discovery_barrier"))
        valid_segments = [tag for tag in segments if tag in TAG_FIELD_MAP["segment_marker"]]
        valid_barriers = [tag for tag in barriers if tag in TAG_FIELD_MAP["discovery_barrier"]]
        for segment in valid_segments:
            for barrier in valid_barriers:
                if matrix[segment][barrier] == 0:
                    non_zero_cells += 1
                matrix[segment][barrier] += 1

    heatmap_values = []
    max_heat = 0
    for segment in segment_tags:
        row_values = []
        for barrier in barrier_tags:
            value = matrix[segment][barrier]
            row_values.append(value)
            if value > max_heat:
                max_heat = value
        heatmap_values.append(row_values)

    total_cells = len(segment_tags) * len(barrier_tags)
    sparse_ratio = round((non_zero_cells / total_cells), 3) if total_cells else 0

    return jsonify(
        {
            "header_stats": {
                "total_reviews": total_reviews,
                "classified_so_far": classified_count,
                "progress_percentage": progress_pct,
                "pending_records": pending_count,
                "failed_records": failed_count,
                "play_store_count": source_counts["play_store"],
                "app_store_count": source_counts["app_store"],
                "average_rating": avg_rating,
                "average_sentiment": avg_sentiment,
                "provisional": pending_count > 0,
            },
            "source_split": source_split,
            "sentiment_distribution": {
                "segments": sentiment_segments,
                "classified_total": classified_count,
            },
            "tag_panels": tag_panels,
            "heatmap": {
                "rows": segment_tags,
                "columns": barrier_tags,
                "values": heatmap_values,
                "max_value": max_heat,
                "is_sparse": sparse_ratio < 0.2,
                "sparsity_ratio": sparse_ratio,
            },
            "methodology_note": "Tag percentages are calculated over the classified corpus size.",
            "app_counts": app_counts,
            "corpus_status": {
                "total_raw_records": total_reviews,
                "classified_records": classified_count,
                "pending_records": pending_count,
                "failed_records": failed_count,
                "percentage_classified": progress_pct,
            },
        }
    )


@app.route("/reviews-by-tag", methods=["GET"])
def reviews_by_tag():
    field = str(request.args.get("field", "")).strip()
    tag = str(request.args.get("tag", "")).strip()

    if field not in TAG_FIELD_MAP:
        return jsonify({"error": "Invalid field"}), 400
    if tag not in TAG_FIELD_MAP[field]:
        return jsonify({"error": "Invalid tag"}), 400

    select_cols = (
        "raw_review_id,sentiment_score,supporting_quote,analyzed_at,"
        "raw_reviews(text,app,rating,source)"
    )
    response = (
        supabase.table("analyzed_reviews")
        .select(select_cols)
        .overlaps(field, [tag])
        .order("analyzed_at", desc=True)
        .limit(20)
        .execute()
    )
    rows = getattr(response, "data", None) or []

    review_items = []
    for row in rows:
        raw = _extract_raw_review(row)
        review_items.append(
            {
                "raw_review_id": row.get("raw_review_id"),
                "text": str(raw.get("text", "") or "").strip(),
                "source": _normalize_source_name(raw.get("source")),
                "app": _normalize_app_name(raw.get("app")),
                "rating": _to_float(raw.get("rating")),
                "sentiment_score": row.get("sentiment_score"),
                "supporting_quote": str(row.get("supporting_quote", "") or "").strip(),
            }
        )

    return jsonify(
        {
            "field": field,
            "tag": tag,
            "count_returned": len(review_items),
            "reviews": review_items,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))
