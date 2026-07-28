import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(query, top_n=10):
    data_path = Path("data/clean/reviews.json")
    with data_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not records:
        return []

    texts = [record.get("text", "") for record in records]
    vectorizer = TfidfVectorizer()
    doc_matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, doc_matrix).ravel()
    ranked_indices = similarities.argsort()[::-1][: min(top_n, len(records))]

    results = []
    for index in ranked_indices:
        result = dict(records[index])
        result["score"] = float(similarities[index])
        results.append(result)

    return results


if __name__ == "__main__":
    results = retrieve("cancel order refund problem", top_n=5)
    for result in results:
        print(f"score={result['score']:.4f} | app={result['app']} | text={result['text']}")
