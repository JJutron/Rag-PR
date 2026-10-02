from __future__ import annotations

import argparse

from src.rag import embed, get_client, load_index, retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="검색 결과를 답변 생성 전에 직접 확인")
    parser.add_argument("question")
    parser.add_argument("--index", default="data/index.json")
    parser.add_argument("--mode", choices=("dense", "hybrid"), default="hybrid")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--site", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--collected-at", default="")
    args = parser.parse_args()

    index = load_index(args.index)
    query_embedding = embed(get_client(), [args.question], index["embedding_model"])[0]
    hits = retrieve(
        index,
        args.question,
        query_embedding,
        args.mode,
        args.top_k,
        args.alpha,
        {"site": args.site, "category": args.category, "collected_at": args.collected_at},
    )
    for number, hit in enumerate(hits, 1):
        preview = hit.chunk["text"].replace("\n", " ")[:240]
        print(f"{number}. {hit.score:.3f} {hit.chunk['source']}#{hit.chunk['id']}\n   {preview}")


if __name__ == "__main__":
    main()
