"""
시장성 평가 RAG 파이프라인

임베딩 모델: BAAI/bge-m3
벡터스토어:  Qdrant (http://localhost:6333)
PDF 적재:    market_data/ 디렉토리의 PDF 자동 ingestion
"""
import os
import glob
import hashlib
from typing import Optional

import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

_MARKET_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "market_data",
)

# PDF 페이지를 몇 페이지씩 묶어 하나의 청크로 만들지 설정
_PAGES_PER_CHUNK = 2


class MarketEvalRAG:
    """
    BAAI/bge-m3 + Qdrant 기반 시장성 평가 RAG 파이프라인.

    초기화 시 market_data/ 의 PDF를 자동으로 Qdrant에 적재합니다.
    이미 적재된 파일은 재적재하지 않습니다 (파일명 기반 중복 체크).
    """

    MODEL_NAME      = "BAAI/bge-m3"
    COLLECTION_NAME = "market_eval"
    VECTOR_SIZE     = 1024  # BAAI/bge-m3 출력 차원

    def __init__(
        self,
        data_dir: str = _MARKET_DATA_DIR,
        qdrant_url: str = "http://localhost:6333",
    ):
        self._data_dir = data_dir
        self._embedder: Optional[SentenceTransformer] = None

        # Qdrant 클라이언트
        self._qdrant = QdrantClient(url=qdrant_url)
        self._ensure_collection()

        # market_data/ PDF 자동 적재
        self._ingest_pdfs()

    # ──────────────────────────────────────────
    # 내부 초기화
    # ──────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """컬렉션이 없으면 생성합니다."""
        existing = [c.name for c in self._qdrant.get_collections().collections]
        if self.COLLECTION_NAME not in existing:
            self._qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"  📦 Qdrant 컬렉션 생성: {self.COLLECTION_NAME}")

    def _ingest_pdfs(self) -> None:
        """market_data/ 의 PDF를 청크로 분할하여 Qdrant에 적재합니다."""
        pdf_files = glob.glob(os.path.join(self._data_dir, "*.pdf"))
        if not pdf_files:
            return

        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            # 파일명 기반 중복 체크
            existing = self._qdrant.count(
                collection_name=self.COLLECTION_NAME,
                count_filter=Filter(
                    must=[FieldCondition(
                        key="source",
                        match=MatchValue(value=filename),
                    )]
                ),
            ).count
            if existing > 0:
                print(f"  ⏭️  이미 적재됨: {filename} ({existing}청크)")
                continue

            print(f"  📄 PDF 적재 중: {filename}")
            chunks = self._pdf_to_chunks(pdf_path)
            if not chunks:
                continue

            embeddings = self.embedder.encode(
                chunks, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

            points = [
                PointStruct(
                    id=int(hashlib.md5(f"{filename}_{i}".encode()).hexdigest(), 16) % (10**18),
                    vector=embeddings[i],
                    payload={"text": chunks[i], "source": filename, "chunk_index": i},
                )
                for i in range(len(chunks))
            ]
            self._qdrant.upsert(collection_name=self.COLLECTION_NAME, points=points)
            print(f"  ✅ 적재 완료: {filename} ({len(chunks)}청크)")

    def _pdf_to_chunks(self, pdf_path: str) -> list[str]:
        """PDF를 페이지 단위로 읽어 _PAGES_PER_CHUNK 페이지씩 묶어 반환합니다."""
        chunks = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]

            for i in range(0, len(pages), _PAGES_PER_CHUNK):
                text = "\n".join(pages[i:i + _PAGES_PER_CHUNK]).strip()
                if text:
                    chunks.append(text)
        except Exception as e:
            print(f"  ⚠️  PDF 파싱 오류 ({os.path.basename(pdf_path)}): {e}")
        return chunks

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print(f"  📥 임베딩 모델 로드: {self.MODEL_NAME}")
            self._embedder = SentenceTransformer(self.MODEL_NAME)
        return self._embedder

    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
    ) -> None:
        """문서를 Qdrant에 추가(upsert)합니다."""
        embeddings = self.embedder.encode(
            documents, normalize_embeddings=True
        ).tolist()

        points = [
            PointStruct(
                id=int(hashlib.md5(ids[i].encode()).hexdigest(), 16) % (10**18),
                vector=embeddings[i],
                payload={"text": documents[i], **(metadatas[i] if metadatas else {})},
            )
            for i in range(len(documents))
        ]
        self._qdrant.upsert(collection_name=self.COLLECTION_NAME, points=points)

    def retrieve(self, query: str, n_results: int = 5) -> list[str]:
        """쿼리와 코사인 유사도가 높은 문서를 반환합니다."""
        total = self._qdrant.count(collection_name=self.COLLECTION_NAME).count
        if total == 0:
            return []

        query_embedding = self.embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()[0]

        results = self._qdrant.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=min(n_results, total),
        ).points
        return [r.payload["text"] for r in results]

    @property
    def doc_count(self) -> int:
        return self._qdrant.count(collection_name=self.COLLECTION_NAME).count
