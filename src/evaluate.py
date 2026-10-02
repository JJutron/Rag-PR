from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.rag import FALLBACK, answer, embed, get_client, load_index, retrieve


def included(hits, expected_source: str) -> str:
    if not expected_source:
        return "미지정"
    return "예" if any(expected_source in hit.chunk["source"] for hit in hits) else "아니오"


def escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def evaluate(index_path: str, questions_path: str, output_path: str, top_k: int, alpha: float) -> None:
    client = get_client()
    index = load_index(index_path)
    questions = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
    rows = []

    for item in questions:
        question = item["question"]
        query_embedding = embed(client, [question], index["embedding_model"])[0]
        results = {}
        for mode in ("dense", "hybrid"):
            hits = retrieve(index, question, query_embedding, mode, top_k, alpha, item.get("filters"))
            response = answer(client, question, hits, model) if hits else FALLBACK
            results[mode] = {
                "hits": hits,
                "answer": response,
                "included": included(hits, item.get("expected_source", "")),
            }

        baseline, improved = results["dense"], results["hybrid"]
        if "미지정" in {baseline["included"], improved["included"]}:
            judgement = "수동 검토"
        elif baseline["included"] == improved["included"]:
            judgement = "동일"
        else:
            judgement = "개선" if improved["included"] == "예" else "악화"
        rows.append((item, baseline, improved, judgement))

    lines = [
        "# RAG 평가 결과",
        "",
        f"- 인덱스: `{index_path}`",
        f"- 질문 수: {len(rows)}",
        f"- Top-K: {top_k}",
        f"- Hybrid Dense 가중치(alpha): {alpha}",
        "",
        "정답 문서가 지정되지 않은 문항과 답변의 근거 일치는 사람이 검색 문서 원문과 대조해 판정합니다.",
        "",
        "| ID | 유형 | 질문 | Baseline Top-K | 개선 Top-K | 정답 포함(B/I) | 판단 |",
        "|---|---|---|---|---|---|---|",
    ]
    for item, baseline, improved, judgement in rows:
        baseline_sources = ", ".join(hit.chunk["source"] for hit in baseline["hits"])
        improved_sources = ", ".join(hit.chunk["source"] for hit in improved["hits"])
        lines.append(
            f"| {item['id']} | {escape(item['type'])} | {escape(item['question'])} | "
            f"{escape(baseline_sources)} | {escape(improved_sources)} | "
            f"{baseline['included']}/{improved['included']} | {judgement} |"
        )

    for item, baseline, improved, judgement in rows:
        lines.extend(
            [
                "",
                f"## {item['id']}. {item['question']}",
                "",
                f"- 유형: {item['type']}",
                f"- 판단: {judgement}",
                "",
                "### Baseline 답변",
                "",
                baseline["answer"],
                "",
                "### 개선 답변",
                "",
                improved["answer"],
                "",
                "### 수동 검토",
                "",
                "- [ ] 검색 문서가 질문과 관련 있다.",
                "- [ ] 답변이 검색 문서의 근거와 일치한다.",
                "- [ ] 문서에 없는 내용은 추측하지 않는다.",
            ]
        )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"평가 결과를 {target}에 저장했습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense Baseline과 Hybrid 개선 버전을 동일 질문으로 비교")
    parser.add_argument("--index", default="data/index.json")
    parser.add_argument("--questions", default="evaluation/questions.json")
    parser.add_argument("--output", default="results/evaluation.md")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(args.index, args.questions, args.output, args.top_k, args.alpha)


if __name__ == "__main__":
    main()
