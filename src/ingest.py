from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from src.rag import embed, get_client


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size must be positive and overlap must be smaller than size")
    clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return [clean[start : start + size] for start in range(0, len(clean), size - overlap) if clean[start : start + size]]


def metadata(path: Path) -> tuple[str, str, str]:
    parts = path.stem.split("__")
    return tuple((parts + ["unknown", "general", "unknown"])[:3])  # type: ignore[return-value]


def load_chunks(data_dir: str | Path, size: int, overlap: int) -> list[dict]:
    root = Path(data_dir)
    documents = [
        path
        for path in sorted(root.rglob("*"))
        if path.suffix.lower() in {".txt", ".md"} and path.name.lower() != "readme.md"
    ]
    chunks: list[dict] = []
    for path in documents:
        site, category, collected_at = metadata(path)
        source = str(path.relative_to(root))
        for number, text in enumerate(chunk_text(path.read_text(encoding="utf-8"), size, overlap)):
            identity = hashlib.sha1(f"{source}:{number}:{text}".encode()).hexdigest()[:12]
            chunks.append(
                {
                    "id": identity,
                    "text": text,
                    "source": source,
                    "site": site,
                    "category": category,
                    "collected_at": collected_at,
                }
            )
    return chunks


def build_index(data_dir: str, output: str, size: int, overlap: int) -> dict:
    chunks = load_chunks(data_dir, size, overlap)
    if not chunks:
        raise SystemExit(f"{data_dir}에 .txt 또는 .md 공식 문서가 없습니다.")
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    vectors = embed(get_client(), [chunk["text"] for chunk in chunks], model)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    index = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model,
        "chunk_size": size,
        "chunk_overlap": overlap,
        "chunks": chunks,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="공식 문서를 JSON 벡터 인덱스로 변환")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="data/index.json")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()
    index = build_index(args.data_dir, args.output, args.chunk_size, args.overlap)
    print(f"{len(index['chunks'])}개 chunk를 {args.output}에 저장했습니다.")


if __name__ == "__main__":
    main()
