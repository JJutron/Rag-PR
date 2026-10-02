from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FALLBACK = "제공된 공식 문서에서 확인할 수 없습니다. 최신 정보는 공식 고객센터에서 확인해 주세요."
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


@dataclass(frozen=True)
class Hit:
    chunk: dict[str, Any]
    score: float


def load_env(path: str | Path | None = None) -> None:
    env_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("\"'")


def get_client():
    load_env()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai 패키지가 없습니다. `python -m pip install -e .`를 실행하세요.") from exc
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY가 필요합니다. .env.example을 참고하세요.")
    return OpenAI(api_key=api_key)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    denom = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denom if denom else 0.0


def bm25_scores(query: str, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    query_terms = tokenize(query)
    documents = [tokenize(chunk["text"]) for chunk in chunks]
    if not query_terms or not documents:
        return [0.0] * len(documents)

    average_length = sum(map(len, documents)) / len(documents) or 1.0
    document_frequency = {
        term: sum(term in document for document in documents)
        for term in set(query_terms)
    }
    scores: list[float] = []
    for document in documents:
        score = 0.0
        frequencies = {term: document.count(term) for term in set(query_terms)}
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            count = document_frequency[term]
            inverse_frequency = math.log(1 + (len(documents) - count + 0.5) / (count + 0.5))
            denominator = frequency + k1 * (1 - b + b * len(document) / average_length)
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if math.isclose(low, high):
        return [0.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def filter_chunks(chunks: list[dict[str, Any]], filters: dict[str, str] | None) -> list[dict[str, Any]]:
    if not filters:
        return chunks
    return [
        chunk
        for chunk in chunks
        if all(not value or chunk.get(key) == value for key, value in filters.items())
    ]


def retrieve(
    index: dict[str, Any],
    question: str,
    query_embedding: list[float],
    mode: str = "hybrid",
    top_k: int = 4,
    alpha: float = 0.5,
    filters: dict[str, str] | None = None,
) -> list[Hit]:
    if mode not in {"dense", "hybrid"}:
        raise ValueError("mode must be dense or hybrid")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    chunks = filter_chunks(index["chunks"], filters)
    dense = [cosine(query_embedding, chunk["embedding"]) for chunk in chunks]
    scores = dense
    if mode == "hybrid":
        sparse = bm25_scores(question, chunks)
        dense_normalized, sparse_normalized = normalize(dense), normalize(sparse)
        scores = [
            alpha * dense_score + (1 - alpha) * sparse_score
            for dense_score, sparse_score in zip(dense_normalized, sparse_normalized)
        ]

    ranked = sorted(zip(chunks, scores), key=lambda item: item[1], reverse=True)
    return [Hit(chunk=chunk, score=score) for chunk, score in ranked[:top_k]]


def embed(client, texts: list[str], model: str) -> list[list[float]]:
    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def load_index(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"인덱스가 없습니다: {path}. 먼저 ingest를 실행하세요.")
    return json.loads(path.read_text(encoding="utf-8"))


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[문서 {number}] source={hit.chunk['source']} | site={hit.chunk['site']} | "
        f"category={hit.chunk['category']} | collected_at={hit.chunk['collected_at']}\n{hit.chunk['text']}"
        for number, hit in enumerate(hits, 1)
    )


def answer(client, question: str, hits: list[Hit], model: str) -> str:
    context = format_context(hits)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "당신은 자연휴양림 규정 안내 도우미다. 제공된 공식 문서 조각만 근거로 답한다. "
                    "문서에서 답을 확인할 수 없거나 실시간 정보가 필요하면 다른 설명 없이 정확히 다음 문장만 답한다: "
                    f"{FALLBACK} 답을 할 수 있으면 적용 대상과 예외를 구분하고, 근거 문서 번호를 [문서 N] 형식으로 표시한다."
                ),
            },
            {"role": "user", "content": f"질문: {question}\n\n공식 문서:\n{context}"},
        ],
    )
    return response.output_text.strip()


def ask(
    question: str,
    index_path: str | Path = "data/index.json",
    mode: str = "hybrid",
    top_k: int = 4,
    alpha: float = 0.5,
    filters: dict[str, str] | None = None,
) -> tuple[str, list[Hit]]:
    client = get_client()
    index = load_index(index_path)
    query_embedding = embed(client, [question], index["embedding_model"])[0]
    hits = retrieve(index, question, query_embedding, mode, top_k, alpha, filters)
    if not hits:
        return FALLBACK, []
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
    return answer(client, question, hits, model), hits


def main() -> None:
    parser = argparse.ArgumentParser(description="공식 자연휴양림 문서 기반 RAG 질의")
    parser.add_argument("question")
    parser.add_argument("--index", default="data/index.json")
    parser.add_argument("--mode", choices=("dense", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5, help="Hybrid에서 Dense 검색 비중")
    parser.add_argument("--site", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--collected-at", default="")
    args = parser.parse_args()
    filters = {
        "site": args.site,
        "category": args.category,
        "collected_at": args.collected_at,
    }
    result, hits = ask(args.question, args.index, args.mode, args.top_k, args.alpha, filters)
    print(result)
    if result != FALLBACK:
        print("\n검색 근거")
        for hit in hits:
            print(f"- {hit.chunk['source']}#{hit.chunk['id']} ({hit.score:.3f})")


if __name__ == "__main__":
    main()
