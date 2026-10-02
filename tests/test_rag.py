import os
import tempfile
import unittest
from pathlib import Path

from src.ingest import chunk_text, load_chunks, metadata
from src.rag import load_env, retrieve


class RagTest(unittest.TestCase):
    def test_chunk_overlap(self):
        chunks = chunk_text("abcdefghij", size=6, overlap=2)
        self.assertEqual(chunks, ["abcdef", "efghij", "ij"])

    def test_hybrid_recovers_exact_keyword(self):
        index = {
            "chunks": [
                {
                    "id": "correct",
                    "text": "RV42 추첨제 결제 기한 안내",
                    "source": "correct.txt",
                    "site": "common",
                    "category": "reservation",
                    "collected_at": "2026-10-02",
                    "embedding": [0.8, 0.2],
                },
                {
                    "id": "wrong",
                    "text": "일반 예약 이용 안내",
                    "source": "wrong.txt",
                    "site": "common",
                    "category": "reservation",
                    "collected_at": "2026-10-02",
                    "embedding": [1.0, 0.0],
                },
            ]
        }
        dense = retrieve(index, "RV42 결제 기한", [1.0, 0.0], mode="dense", top_k=1)
        hybrid = retrieve(index, "RV42 결제 기한", [1.0, 0.0], mode="hybrid", top_k=1, alpha=0.25)
        self.assertEqual(dense[0].chunk["id"], "wrong")
        self.assertEqual(hybrid[0].chunk["id"], "correct")

    def test_metadata_filter(self):
        index = {
            "chunks": [
                {"id": "a", "text": "규정", "source": "a", "site": "alpha", "category": "general", "collected_at": "2026", "embedding": [1.0]},
                {"id": "b", "text": "규정", "source": "b", "site": "beta", "category": "general", "collected_at": "2026", "embedding": [1.0]},
            ]
        }
        hits = retrieve(index, "규정", [1.0], filters={"site": "beta"})
        self.assertEqual([hit.chunk["id"] for hit in hits], ["b"])

    def test_filename_metadata(self):
        self.assertEqual(metadata(__import__("pathlib").Path("forest__refund__2026-10-02.md")), ("forest", "refund", "2026-10-02"))

    def test_load_env_overrides_existing_key(self):
        previous = os.environ.get("OPENAI_API_KEY")

        def restore() -> None:
            if previous is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = previous

        self.addCleanup(restore)
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("OPENAI_API_KEY=from-file\n", encoding="utf-8")
            os.environ["OPENAI_API_KEY"] = "from-shell"
            load_env(env_file)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "from-file")

    def test_official_data_has_traceable_metadata(self):
        data_dir = Path(__file__).parents[1] / "data"
        chunks = load_chunks(data_dir, size=10_000, overlap=0)
        sources = {chunk["source"] for chunk in chunks}
        self.assertEqual(len(sources), 8)
        self.assertTrue(all(chunk["collected_at"] == "2026-10-02" for chunk in chunks))
        self.assertTrue(
            all(
                "https://www.foresttrip.go.kr/" in chunk["text"]
                or "https://www.hwadamsup.com/" in chunk["text"]
                for chunk in chunks
            )
        )


if __name__ == "__main__":
    unittest.main()
