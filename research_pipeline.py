import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

RESEARCH_INTENTS = [
    "repetitive_purchase",
    "exploration_barrier",
    "discovery_behaviour",
    "habit_formation",
    "information_need",
    "frustration",
    "experimental_segment",
    "unmet_need",
    "category_opportunity",
    "general_feedback",
]

RESEARCH_RELEVANCE_ALLOWED = {
    "discovery_relevant",
    "indirectly_relevant",
    "operational_only",
    "irrelevant",
}

FEEDBACK_DOMAIN_ALLOWED = {
    "category_discovery",
    "shopping_habit",
    "purchase_mission",
    "trust_and_quality",
    "product_information",
    "price_and_value",
    "recommendation",
    "assortment",
    "impulse_purchase",
    "workaround",
    "delivery",
    "refund",
    "customer_support",
    "app_technical",
    "other",
}

HYBRID_WEIGHTS = {
    "semantic": 0.40,
    "keyword": 0.20,
    "taxonomy": 0.20,
    "quality": 0.15,
    "source_diversity": 0.05,
}

INTENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "exploration_barrier": {
        "preferred_taxonomy_fields": [
            "discovery_barrier",
            "behavioral_schema.exploration_barrier",
            "behavioral_schema.perceived_risk",
            "behavioral_schema.information_needed",
            "behavioral_schema.trust_signal",
            "behavioral_schema.workaround",
        ],
        "required_relevance": ["discovery_relevant", "indirectly_relevant"],
        "eligible_domains": [
            "category_discovery",
            "trust_and_quality",
            "product_information",
            "price_and_value",
            "assortment",
            "shopping_habit",
        ],
        "excluded_domains": ["delivery", "refund", "customer_support", "app_technical"],
        "query_expansions": [
            "hard to try new categories",
            "hesitation risk trust unfamiliar products",
            "missing information before trying",
        ],
        "min_direct_evidence": 5,
        "min_avg_score": 0.45,
    },
    "repetitive_purchase": {
        "preferred_taxonomy_fields": ["behavioral_driver", "behavioral_schema.habit_signal", "behavioral_schema.shopping_mission"],
        "required_relevance": ["discovery_relevant", "indirectly_relevant"],
        "eligible_domains": ["shopping_habit", "purchase_mission", "price_and_value", "assortment"],
        "excluded_domains": ["delivery", "refund", "customer_support", "app_technical"],
        "query_expansions": ["same items repeat reorder routine", "habit repeat purchase behavior"],
        "min_direct_evidence": 5,
        "min_avg_score": 0.42,
    },
    "discovery_behaviour": {
        "preferred_taxonomy_fields": ["discovery_channel", "behavioral_schema.discovery_method", "behavioral_schema.purchase_trigger"],
        "required_relevance": ["discovery_relevant", "indirectly_relevant"],
        "eligible_domains": ["category_discovery", "recommendation", "impulse_purchase", "product_information"],
        "excluded_domains": ["delivery", "refund", "customer_support", "app_technical"],
        "query_expansions": ["how users discover products", "recommendation search social discovery"],
        "min_direct_evidence": 5,
        "min_avg_score": 0.42,
    },
    "habit_formation": {
        "preferred_taxonomy_fields": ["behavioral_driver", "behavioral_schema.habit_signal", "behavioral_schema.current_category"],
        "required_relevance": ["discovery_relevant", "indirectly_relevant"],
        "eligible_domains": ["shopping_habit", "purchase_mission"],
        "excluded_domains": ["delivery", "refund", "customer_support", "app_technical"],
        "query_expansions": ["habit routine repeat purchases", "lock in same category"],
        "min_direct_evidence": 4,
        "min_avg_score": 0.40,
    },
    "information_need": {
        "preferred_taxonomy_fields": ["discovery_barrier", "behavioral_schema.information_needed", "behavioral_schema.trust_signal"],
        "required_relevance": ["discovery_relevant", "indirectly_relevant"],
        "eligible_domains": ["product_information", "trust_and_quality", "category_discovery"],
        "excluded_domains": ["delivery", "refund", "customer_support", "app_technical"],
        "query_expansions": ["what information is needed before trying category", "details quality trust"],
        "min_direct_evidence": 4,
        "min_avg_score": 0.40,
    },
    "frustration": {
        "preferred_taxonomy_fields": ["frustration_type", "discovery_barrier", "behavioral_schema.exploration_barrier"],
        "required_relevance": ["discovery_relevant", "indirectly_relevant", "operational_only"],
        "eligible_domains": list(FEEDBACK_DOMAIN_ALLOWED),
        "excluded_domains": [],
        "query_expansions": ["recurring frustrations", "pain points repeated complaints"],
        "min_direct_evidence": 4,
        "min_avg_score": 0.35,
    },
    "experimental_segment": {
        "preferred_taxonomy_fields": ["segment_marker", "behavioral_schema.category_tried", "behavioral_schema.purchase_trigger"],
        "required_relevance": ["discovery_relevant", "indirectly_relevant"],
        "eligible_domains": ["category_discovery", "shopping_habit", "impulse_purchase", "recommendation"],
        "excluded_domains": ["delivery", "refund", "customer_support", "app_technical"],
        "query_expansions": ["users likely to experiment", "new category trial users"],
        "min_direct_evidence": 4,
        "min_avg_score": 0.40,
    },
    "unmet_need": {
        "preferred_taxonomy_fields": ["unmet_need", "behavioral_schema.unmet_need", "behavioral_schema.workaround"],
        "required_relevance": ["discovery_relevant", "indirectly_relevant"],
        "eligible_domains": ["assortment", "category_discovery", "product_information", "workaround"],
        "excluded_domains": ["delivery", "refund", "customer_support", "app_technical"],
        "query_expansions": ["unmet needs missing options", "requests for categories or info"],
        "min_direct_evidence": 4,
        "min_avg_score": 0.40,
    },
    "category_opportunity": {
        "preferred_taxonomy_fields": ["categories", "behavioral_schema.category_tried", "behavioral_schema.current_category"],
        "required_relevance": ["discovery_relevant", "indirectly_relevant"],
        "eligible_domains": ["category_discovery", "assortment", "purchase_mission", "shopping_habit"],
        "excluded_domains": ["delivery", "refund", "customer_support", "app_technical"],
        "query_expansions": ["new category opportunities", "cross category expansion"],
        "min_direct_evidence": 4,
        "min_avg_score": 0.40,
    },
    "general_feedback": {
        "preferred_taxonomy_fields": ["behavioral_driver", "discovery_barrier", "frustration_type"],
        "required_relevance": ["discovery_relevant", "indirectly_relevant", "operational_only"],
        "eligible_domains": list(FEEDBACK_DOMAIN_ALLOWED),
        "excluded_domains": [],
        "query_expansions": ["user feedback patterns"],
        "min_direct_evidence": 3,
        "min_avg_score": 0.30,
    },
}

STOP_WORDS = {
    "the", "is", "a", "an", "to", "of", "and", "for", "on", "in", "from", "do", "does", "what", "why", "how", "are", "users", "user", "with", "or", "it", "that", "this", "be", "as", "at", "by", "today", "new",
}

VERY_SHORT_WORDS = {"ok", "bad", "good", "nice", "worst", "best"}

OPERATIONAL_TERMS = {
    "delivery", "late", "delay", "refund", "return", "support", "customer care", "agent", "app crash", "bug", "payment failed", "cancel order",
}

DISCOVERY_TERMS = {
    "new category", "try", "explore", "discover", "recommend", "suggest", "search", "first time", "unfamiliar", "risk", "hard to find", "not showing", "missing details",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize(text: str) -> List[str]:
    return [tok for tok in re.findall(r"[a-zA-Z0-9]+", normalize_text(text)) if tok and tok not in STOP_WORDS]


def detect_intents(question: str) -> List[str]:
    q = normalize_text(question)
    intents = []
    rules = [
        ("repetitive_purchase", ["repeated", "repeat", "same categories", "same category", "again and again"]),
        ("exploration_barrier", ["prevent", "barrier", "hesitate", "exploring", "new categories", "why don't"]),
        ("discovery_behaviour", ["discover", "discovery", "find products", "recommendation", "search"]),
        ("habit_formation", ["habit", "routine", "habit formation", "lock in"]),
        ("information_need", ["information", "before trying", "details", "need to know"]),
        ("frustration", ["frustration", "problem", "complaint", "issue"]),
        ("experimental_segment", ["experiment", "more likely", "early adopter", "try new"]),
        ("unmet_need", ["unmet", "missing", "need", "lack", "wish"]),
        ("category_opportunity", ["opportunity", "expand", "which category", "cross category"]),
    ]
    for intent, cues in rules:
        if any(cue in q for cue in cues):
            intents.append(intent)

    if not intents:
        intents = ["general_feedback"]

    if "exploration_barrier" in intents and "information_need" not in intents:
        intents.append("information_need")

    return intents[:3]


def merge_intent_config(intents: List[str]) -> Dict[str, Any]:
    merged = {
        "preferred_taxonomy_fields": [],
        "required_relevance": [],
        "eligible_domains": set(),
        "excluded_domains": set(),
        "query_expansions": [],
        "min_direct_evidence": 3,
        "min_avg_score": 0.35,
    }

    for intent in intents:
        cfg = INTENT_CONFIG.get(intent) or INTENT_CONFIG["general_feedback"]
        merged["preferred_taxonomy_fields"].extend(cfg.get("preferred_taxonomy_fields", []))
        merged["required_relevance"].extend(cfg.get("required_relevance", []))
        merged["eligible_domains"].update(cfg.get("eligible_domains", []))
        merged["excluded_domains"].update(cfg.get("excluded_domains", []))
        merged["query_expansions"].extend(cfg.get("query_expansions", []))
        merged["min_direct_evidence"] = max(merged["min_direct_evidence"], cfg.get("min_direct_evidence", 3))
        merged["min_avg_score"] = max(merged["min_avg_score"], cfg.get("min_avg_score", 0.35))

    merged["preferred_taxonomy_fields"] = sorted(set(merged["preferred_taxonomy_fields"]))
    merged["required_relevance"] = sorted(set(merged["required_relevance"]))
    merged["query_expansions"] = sorted(set(merged["query_expansions"]))
    merged["eligible_domains"] = sorted(set(merged["eligible_domains"]))
    merged["excluded_domains"] = sorted(set(merged["excluded_domains"]))
    return merged


def infer_feedback_domain(text: str, tags: Dict[str, List[str]]) -> str:
    t = normalize_text(text)
    if any(x in t for x in ["refund", "return"]):
        return "refund"
    if any(x in t for x in ["delivery", "late", "delay", "rider"]):
        return "delivery"
    if any(x in t for x in ["support", "customer care", "helpline", "agent"]):
        return "customer_support"
    if any(x in t for x in ["crash", "bug", "app", "login", "payment failed"]):
        return "app_technical"
    if any(x in t for x in ["search", "discover", "new category", "recommend"]):
        return "category_discovery"
    if any(x in t for x in ["habit", "routine", "always buy", "same"]):
        return "shopping_habit"
    if any(x in t for x in ["price", "expensive", "cheap", "discount"]):
        return "price_and_value"
    if any(x in t for x in ["quality", "fresh", "trust", "authentic"]):
        return "trust_and_quality"
    if any(x in t for x in ["information", "details", "ingredients", "spec"]):
        return "product_information"
    if tags.get("unmet_need"):
        return "assortment"
    return "other"


def infer_research_relevance(text: str, domain: str) -> str:
    t = normalize_text(text)
    discovery_hits = sum(1 for cue in DISCOVERY_TERMS if cue in t)
    operational_hits = sum(1 for cue in OPERATIONAL_TERMS if cue in t)
    bridge = any(cue in t for cue in ["because", "therefore", "so i avoid", "so i don't", "won't try", "stop trying", "so i avoid trying", "not willing to try"])

    if discovery_hits > 0 and operational_hits == 0:
        return "discovery_relevant"
    if discovery_hits > 0 and operational_hits > 0:
        return "indirectly_relevant" if bridge else "operational_only"
    if domain in {"delivery", "refund", "customer_support", "app_technical"}:
        return "operational_only"
    if len(t) < 20:
        return "irrelevant"
    return "indirectly_relevant"


def has_exploration_barrier_signal(record: Dict[str, Any], text: str) -> bool:
    tags = record.get("tags", {}) or {}
    schema = record.get("behavioral_schema", {}) or {}

    if tags.get("discovery_barrier"):
        return True

    for field in ["exploration_barrier", "perceived_risk", "information_needed", "workaround"]:
        value = str(schema.get(field, "not_stated") or "not_stated").strip().lower()
        if value and value != "not_stated":
            return True

    t = normalize_text(text)
    barrier_terms = [
        "hard", "difficult", "can't", "cannot", "risk", "afraid", "hesitate", "uncertain",
        "not showing", "missing", "expensive", "overpriced", "no details", "don't trust", "dont trust",
    ]
    return any(term in t for term in barrier_terms)


def extract_behavioral_schema(text: str, tags: Dict[str, List[str]], supporting_quote: str = "") -> Dict[str, Any]:
    t = normalize_text(text)
    schema = {
        "shopping_mission": "not_stated",
        "habit_signal": "not_stated",
        "current_category": "not_stated",
        "category_tried": "not_stated",
        "discovery_method": "not_stated",
        "exploration_barrier": "not_stated",
        "purchase_trigger": "not_stated",
        "information_needed": "not_stated",
        "trust_signal": "not_stated",
        "perceived_risk": "not_stated",
        "workaround": "not_stated",
        "user_context": "not_stated",
        "unmet_need": "not_stated",
        "evidence_quote": supporting_quote.strip() or "not_stated",
        "classification_confidence": "medium",
    }

    if any(cue in t for cue in ["daily", "weekly", "every", "routine", "again"]):
        schema["habit_signal"] = "repeat purchase routine"
    if any(cue in t for cue in ["grocery", "milk", "vegetable", "snack", "pharmacy"]):
        schema["current_category"] = "explicitly mentioned category"
    if any(cue in t for cue in ["tried", "first time", "new category", "experiment"]):
        schema["category_tried"] = "user mentions trying a new category"
    if any(cue in t for cue in ["search", "recommend", "friend", "instagram", "youtube"]):
        schema["discovery_method"] = "explicit discovery channel mention"
    if any(cue in t for cue in ["hard to find", "confusing", "don't trust", "risk", "uncertain"]):
        schema["exploration_barrier"] = "explicit exploration barrier"
    if any(cue in t for cue in ["discount", "offer", "urgent", "convenient", "fast"]):
        schema["purchase_trigger"] = "explicit trigger mention"
    if any(cue in t for cue in ["need details", "ingredients", "expiry", "source", "quality proof"]):
        schema["information_needed"] = "explicit information requirement"
    if any(cue in t for cue in ["trusted", "authentic", "brand", "fresh"]):
        schema["trust_signal"] = "explicit trust cue"
    if any(cue in t for cue in ["fear", "risk", "worry", "might"]):
        schema["perceived_risk"] = "explicit risk mention"
    if any(cue in t for cue in ["instead", "so i", "i switched", "workaround"]):
        schema["workaround"] = "explicit workaround mention"
    if any(cue in t for cue in ["office", "family", "kids", "pet", "bachelor"]):
        schema["user_context"] = "explicit personal context"
    if tags.get("unmet_need") or any(cue in t for cue in ["please add", "missing", "not available"]):
        schema["unmet_need"] = "explicit unmet need mention"

    filled = sum(1 for k, v in schema.items() if k != "classification_confidence" and v != "not_stated")
    if filled <= 2:
        schema["classification_confidence"] = "low"
    elif filled >= 6:
        schema["classification_confidence"] = "high"

    return schema


def evidence_quality_score(text: str, domain: str, schema: Dict[str, Any]) -> float:
    t = normalize_text(text)
    if len(re.sub(r"[^a-z0-9]+", "", t)) < 20:
        return 0.05

    score = 0.35
    meaningful = len([w for w in tokenize(t) if w not in VERY_SHORT_WORDS])
    if meaningful >= 6:
        score += 0.15
    if any(x in t for x in ["because", "so", "therefore", "hence"]):
        score += 0.15
    if any(schema[field] != "not_stated" for field in ["exploration_barrier", "purchase_trigger", "workaround", "information_needed"]):
        score += 0.20
    if any(schema[field] != "not_stated" for field in ["current_category", "category_tried", "habit_signal"]):
        score += 0.10
    if domain in {"delivery", "refund", "customer_support", "app_technical"}:
        score -= 0.20
    if re.fullmatch(r"(good|bad|nice|worst|best|ok|awesome|poor)[!. ]*", t):
        score = min(score, 0.10)
    if any(phrase in t for phrase in ["good service", "good quality", "on time delivery", "superfast"]) and "because" not in t:
        score -= 0.20

    return max(0.0, min(1.0, score))


def min_max_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def compute_semantic_scores(query: str, docs: List[str]) -> List[float]:
    if not docs:
        return []
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        mat = vec.fit_transform(docs)
        qv = vec.transform([query])
        return cosine_similarity(qv, mat).ravel().tolist()
    except ValueError:
        return [0.0 for _ in docs]


def compute_keyword_scores(query_terms: List[str], docs: List[str]) -> List[float]:
    if not docs:
        return []
    doc_term_counts = [Counter(tokenize(doc)) for doc in docs]
    df = Counter()
    for counts in doc_term_counts:
        for term in counts.keys():
            df[term] += 1

    n_docs = max(1, len(docs))
    scores = []
    for counts in doc_term_counts:
        score = 0.0
        for term in query_terms:
            if term not in counts:
                continue
            idf = math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1
            score += counts[term] * idf
        scores.append(score)
    return scores


def taxonomy_match_score(record: Dict[str, Any], preferred_fields: List[str]) -> Tuple[float, List[str]]:
    tags = record.get("tags", {})
    schema = record.get("behavioral_schema", {})
    matched = []
    score = 0.0

    for field in preferred_fields:
        if field.startswith("behavioral_schema."):
            k = field.split(".", 1)[1]
            if schema.get(k, "not_stated") != "not_stated":
                matched.append(field)
                score += 1.0
            continue
        values = tags.get(field, []) if isinstance(tags, dict) else []
        if isinstance(values, list) and values:
            matched.append(field)
            score += 1.0

    if preferred_fields:
        score = score / float(len(preferred_fields))
    return max(0.0, min(1.0, score)), matched


@dataclass
class PipelineResult:
    question: str
    normalized_question: str
    intents: List[str]
    selected_filters: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    top_records: List[Dict[str, Any]]
    excluded_records: List[Dict[str, Any]]
    sufficiency: Dict[str, Any]


def _record_text(record: Dict[str, Any]) -> str:
    quote = str(record.get("supporting_quote") or "")
    text = str(record.get("text") or "")
    return (text + " " + quote).strip()


def run_hybrid_retrieval(question: str, records: List[Dict[str, Any]]) -> PipelineResult:
    normalized_question = normalize_text(question)
    intents = detect_intents(normalized_question)
    merged = merge_intent_config(intents)

    expanded_query = " ".join([normalized_question] + merged["query_expansions"])
    query_terms = tokenize(expanded_query)

    enriched_records = []
    excluded_records = []
    for record in records:
        text = _record_text(record)
        domain = str(record.get("feedback_domain") or infer_feedback_domain(text, record.get("tags", {})))
        relevance = str(record.get("research_relevance") or infer_research_relevance(text, domain))
        schema = record.get("behavioral_schema") or extract_behavioral_schema(text, record.get("tags", {}), record.get("supporting_quote", ""))

        record["feedback_domain"] = domain if domain in FEEDBACK_DOMAIN_ALLOWED else "other"
        record["research_relevance"] = relevance if relevance in RESEARCH_RELEVANCE_ALLOWED else "irrelevant"
        record["behavioral_schema"] = schema

        exclusion_reasons = []
        if merged["required_relevance"] and record["research_relevance"] not in merged["required_relevance"]:
            exclusion_reasons.append("relevance_class_not_allowed")
        if merged["eligible_domains"] and record["feedback_domain"] not in merged["eligible_domains"]:
            exclusion_reasons.append("domain_not_eligible")
        if merged["excluded_domains"] and record["feedback_domain"] in merged["excluded_domains"] and record["research_relevance"] != "discovery_relevant":
            exclusion_reasons.append("domain_excluded_for_intent")

        if "exploration_barrier" in intents and not has_exploration_barrier_signal(record, text):
            exclusion_reasons.append("missing_exploration_barrier_signal")

        if exclusion_reasons:
            excluded_records.append(
                {
                    "id": record.get("id"),
                    "text": text[:200],
                    "reasons": exclusion_reasons,
                    "research_relevance": record["research_relevance"],
                    "feedback_domain": record["feedback_domain"],
                }
            )
            continue

        enriched_records.append(record)

    docs = [_record_text(r) for r in enriched_records]
    semantic_raw = compute_semantic_scores(expanded_query, docs)
    keyword_raw = compute_keyword_scores(query_terms, docs)

    taxonomy_raw = []
    quality_raw = []
    match_fields_all = []
    for record in enriched_records:
        tax_score, match_fields = taxonomy_match_score(record, merged["preferred_taxonomy_fields"])
        q_score = evidence_quality_score(_record_text(record), record["feedback_domain"], record["behavioral_schema"])
        taxonomy_raw.append(tax_score)
        quality_raw.append(q_score)
        match_fields_all.append(match_fields)

    semantic_norm = min_max_normalize(semantic_raw)
    keyword_norm = min_max_normalize(keyword_raw)
    taxonomy_norm = min_max_normalize(taxonomy_raw)
    quality_norm = min_max_normalize(quality_raw)

    source_counts = Counter(str(r.get("source") or "unknown") for r in enriched_records)
    diversity_raw = [1.0 / math.sqrt(max(1, source_counts[str(r.get("source") or "unknown")])) for r in enriched_records]
    diversity_norm = min_max_normalize(diversity_raw)

    scored = []
    for idx, record in enumerate(enriched_records):
        final = (
            HYBRID_WEIGHTS["semantic"] * semantic_norm[idx]
            + HYBRID_WEIGHTS["keyword"] * keyword_norm[idx]
            + HYBRID_WEIGHTS["taxonomy"] * taxonomy_norm[idx]
            + HYBRID_WEIGHTS["quality"] * quality_norm[idx]
            + HYBRID_WEIGHTS["source_diversity"] * diversity_norm[idx]
        )

        rerank = final + 0.05 * (1.0 if record["research_relevance"] == "discovery_relevant" else 0.0)
        rerank += 0.03 * min(1.0, len(match_fields_all[idx]) / 3.0)

        scored.append(
            {
                **record,
                "semantic_score": round(semantic_norm[idx], 4),
                "keyword_score": round(keyword_norm[idx], 4),
                "taxonomy_score": round(taxonomy_norm[idx], 4),
                "quality_score": round(quality_norm[idx], 4),
                "source_diversity_score": round(diversity_norm[idx], 4),
                "final_score": round(final, 4),
                "final_reranking_score": round(rerank, 4),
                "match_fields": match_fields_all[idx],
            }
        )

    scored.sort(key=lambda item: item["final_reranking_score"], reverse=True)
    initial_candidates = scored[:30]
    top_records = initial_candidates[:15]

    directly = [r for r in initial_candidates if r.get("research_relevance") == "discovery_relevant"]
    indirectly = [r for r in initial_candidates if r.get("research_relevance") == "indirectly_relevant"]
    avg_score = sum(r.get("final_reranking_score", 0.0) for r in top_records) / max(1, len(top_records))
    source_count = len(set(str(r.get("source") or "unknown") for r in initial_candidates))

    category_counts = Counter(r.get("research_relevance", "irrelevant") for r in initial_candidates)
    domain_counts = Counter(r.get("feedback_domain", "other") for r in initial_candidates)

    required_direct = merged["min_direct_evidence"]
    required_avg = merged["min_avg_score"]

    if len(directly) >= required_direct and avg_score >= required_avg:
        state = "sufficient"
    elif len(directly) >= max(2, required_direct - 2):
        state = "limited"
    else:
        state = "insufficient"

    sufficiency = {
        "state": state,
        "total_eligible_records": len(initial_candidates),
        "directly_relevant_records": len(directly),
        "indirectly_relevant_records": len(indirectly),
        "source_count": source_count,
        "average_reranking_score": round(avg_score, 4),
        "classification_coverage": dict(category_counts),
        "source_distribution": dict(Counter(r.get("source") or "unknown" for r in initial_candidates)),
        "feedback_domain_distribution": dict(domain_counts),
        "minimum_requirements": {
            "directly_relevant_records": required_direct,
            "average_score": required_avg,
            "independent_support_records": 2,
        },
    }

    selected_filters = {
        "intents": intents,
        "preferred_taxonomy_fields": merged["preferred_taxonomy_fields"],
        "required_relevance": merged["required_relevance"],
        "eligible_domains": merged["eligible_domains"],
        "excluded_domains": merged["excluded_domains"],
        "query_expansions": merged["query_expansions"],
    }

    return PipelineResult(
        question=question,
        normalized_question=normalized_question,
        intents=intents,
        selected_filters=selected_filters,
        candidates=initial_candidates,
        top_records=top_records,
        excluded_records=excluded_records,
        sufficiency=sufficiency,
    )


def build_synthesis_prompt(question: str, records: List[Dict[str, Any]], sufficiency: Dict[str, Any], classified_total: int, pending_count: int) -> str:
    lines = []
    lines.append(f"Question: {question}")
    lines.append(f"Eligible retrieved records: {len(records)}")
    lines.append(f"Classified corpus size: {classified_total}")
    lines.append(f"Pending classification count: {pending_count}")
    lines.append(f"Evidence sufficiency state: {sufficiency.get('state')}")
    lines.append("\nEvidence records:")
    for rec in records:
        text = _record_text(rec).replace("\n", " ").strip()
        lines.append(
            f"- id={rec.get('id')} | app={rec.get('app')} | source={rec.get('source')} | relevance={rec.get('research_relevance')} | domain={rec.get('feedback_domain')} | score={rec.get('final_reranking_score')} | text={text}"
        )

    lines.append("\nReturn ONLY valid JSON matching this schema:")
    lines.append("{")
    lines.append('  "question": "",')
    lines.append('  "insights": [')
    lines.append('    {')
    lines.append('      "title": "",')
    lines.append('      "summary": "",')
    lines.append('      "supporting_record_ids": [],')
    lines.append('      "supporting_evidence": [')
    lines.append('        {')
    lines.append('          "excerpt": "",')
    lines.append('          "source": "",')
    lines.append('          "date": "",')
    lines.append('          "source_url": ""')
    lines.append('        }')
    lines.append('      ]')
    lines.append('    }')
    lines.append('  ]')
    lines.append("}")

    lines.append("\nRules:")
    lines.append("- Use only the supplied evidence records.")
    lines.append("- Return exactly 2 or 3 insights when evidence supports them. If evidence is too thin for multiple distinct insights, return one strong insight.")
    lines.append("- Each insight must be meaningfully distinct and answer the specific research question.")
    lines.append("- Each insight summary must be 2-3 sentences and explain behavioural meaning, not generic themes.")
    lines.append("- Do not repeat the title sentence inside the summary.")
    lines.append("- Each insight should have 2-3 supporting excerpts when available.")
    lines.append("- Each supporting excerpt must be verbatim from evidence; do not paraphrase or invent excerpts.")
    lines.append("- Do not include markdown tokens such as ** or ## in visible fields.")
    lines.append("- Avoid generic headlines like 'Product information' or 'Quality concerns'. Use concrete behavioural claims.")

    return "\n".join(lines)


def build_insufficient_evidence_response(question: str, sufficiency: Dict[str, Any]) -> str:
    return (
        "1. Direct answer\n"
        f"Evidence is {sufficiency.get('state')} for answering '{question}' with confidence.\n\n"
        "2. Evidence-backed themes\n"
        "Only partial signals are available in the currently classified corpus.\n\n"
        "3. Evidence count for each theme\n"
        f"Directly relevant: {sufficiency.get('directly_relevant_records', 0)}; "
        f"Indirectly relevant: {sufficiency.get('indirectly_relevant_records', 0)}; "
        f"Eligible records: {sufficiency.get('total_eligible_records', 0)}.\n\n"
        "4. Supporting excerpts and source links\n"
        "Insufficient direct evidence to provide stable excerpts for major claims.\n\n"
        "5. Interpretation clearly labelled as interpretation\n"
        "Interpretation: More classification coverage or stronger discovery-specific records are required.\n\n"
        "6. Contradictory or minority evidence\n"
        "Operational-only feedback dominates the current candidate set.\n\n"
        "7. Evidence-strength rating\n"
        f"{sufficiency.get('state').upper()}\n\n"
        "8. Product implication\n"
        "Improve discovery-focused data capture before making major product decisions.\n\n"
        "9. Data limitations\n"
        "Corpus is partially classified and the current question needs more direct discovery evidence."
    )


def build_debug_payload(result: PipelineResult) -> Dict[str, Any]:
    top = []
    for rec in result.top_records:
        top.append(
            {
                "id": rec.get("id"),
                "semantic_score": rec.get("semantic_score"),
                "keyword_score": rec.get("keyword_score"),
                "taxonomy_score": rec.get("taxonomy_score"),
                "quality_score": rec.get("quality_score"),
                "final_reranking_score": rec.get("final_reranking_score"),
                "match_fields": rec.get("match_fields", []),
                "research_relevance": rec.get("research_relevance"),
                "feedback_domain": rec.get("feedback_domain"),
            }
        )

    return {
        "original_question": result.question,
        "normalized_question": result.normalized_question,
        "detected_research_intent": result.intents,
        "taxonomy_tags_selected": result.selected_filters.get("preferred_taxonomy_fields", []),
        "filters_applied": result.selected_filters,
        "eligible_records": result.sufficiency.get("total_eligible_records", 0),
        "top_retrieved_records": top,
        "excluded_records": result.excluded_records,
    }
