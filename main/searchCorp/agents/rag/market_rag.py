"""
시장성 평가 RAG 파이프라인

임베딩 모델: BAAI/bge-m3
벡터스토어:  ChromaDB (persistent)
"""
import os
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

# 벡터스토어 저장 경로 (searchCorp/market_data/)
_MARKET_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "market_data",
)


class MarketEvalRAG:
    """
    BAAI/bge-m3 기반 시장성 평가 RAG 파이프라인.

    문서 추가:
        rag = MarketEvalRAG()
        rag.add_documents(
            documents=["반도체 시장 보고서 내용..."],
            ids=["report_2024_q1"],
            metadatas=[{"source": "IDC", "year": 2024}],
        )

    검색:
        context = rag.retrieve("AI칩 시장 경쟁 현황")
    """

    MODEL_NAME = "BAAI/bge-m3"
    COLLECTION_NAME = "market_eval"

    def __init__(self, data_dir: str = _MARKET_DATA_DIR):
        os.makedirs(data_dir, exist_ok=True)

        # 지연 로딩 (처음 embed 호출 시 모델 로드)
        self._embedder: Optional[SentenceTransformer] = None

        # ChromaDB persistent 클라이언트
        self._chroma = chromadb.PersistentClient(path=data_dir)
        self._collection = self._chroma.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def embedder(self) -> SentenceTransformer:
        """지연 로딩: 처음 사용 시 BAAI/bge-m3 모델 다운로드 및 로드"""
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
        """
        시장 데이터 문서를 벡터스토어에 추가(upsert)합니다.

        Args:
            documents:  문서 텍스트 목록
            ids:        각 문서의 고유 ID
            metadatas:  문서 메타데이터 (source, year, category 등)
        """
        embeddings = self.embedder.encode(
            documents, normalize_embeddings=True
        ).tolist()

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in documents],
        )

    def retrieve(self, query: str, n_results: int = 5) -> list[str]:
        """
        쿼리와 코사인 유사도가 높은 시장 데이터 문서를 반환합니다.

        벡터스토어에 문서가 없으면 빈 리스트를 반환합니다.
        """
        total = self._collection.count()
        if total == 0:
            return []

        query_embedding = self.embedder.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, total),
        )
        return results["documents"][0]

    @property
    def doc_count(self) -> int:
        """벡터스토어에 저장된 문서 수"""
        return self._collection.count()
