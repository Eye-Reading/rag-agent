"""
성공 롤모델 DNA RAG 파이프라인

임베딩 모델 : BAAI/bge-m3
벡터스토어  : Qdrant http://localhost:6333  (market_eval 과 동일 인스턴스, 별도 컬렉션)
컬렉션      : dna_rolemodel

적재 대상   : NVIDIA · Qualcomm · AMD
중복 방지   : 기업명 기반 payload 필터로 이미 적재된 롤모델은 재적재하지 않음

중요:
- searchCorp 의 SearchAgent가 쓰는 StartupInfo 스키마 관점으로 데이터를 수집합니다.
- 롤모델 데이터가 컬렉션에 이미 있으면 LLM 요청 없이 재사용합니다.
- 즉, 초기 세팅(첫 적재)에서만 LLM 요청이 발생하고 이후 실행은 캐시 기반으로 동작합니다.
"""
import json
import hashlib
from typing import Optional

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
)


# ──────────────────────────────────────────
# searchCorp 정합성 원칙
# - search_agent.py 의 _get_startup_list 프롬프트 방식(웹 검색 → JSON 정리)을 재사용한
#   2단계 LLM 호출로 롤모델 기업 데이터를 수집합니다.
# - 최종 구조는 searchCorp 의 StartupInfo 형태와 동일한 필드를 사용합니다.
# - 임베딩 텍스트는 StartupInfo를 동일한 관점으로 직렬화하여 생성합니다.
# ──────────────────────────────────────────

_ROLEMODEL_COMPANIES = ["NVIDIA", "Qualcomm", "AMD"]


class DnaRoleModelRAG:
    """
    BAAI/bge-m3 + Qdrant 기반 성공 롤모델 DNA 유사도 검색 파이프라인.

    - 기존 market_eval 컬렉션과 동일한 Qdrant 인스턴스를 공유하되
      'dna_rolemodel' 컬렉션으로 데이터를 격리합니다.
    - 초기화 시 NVIDIA · Qualcomm · AMD 3사 데이터를
      자동으로 upsert합니다 (기업명 기반 중복 체크).
    """

    MODEL_NAME      = "BAAI/bge-m3"
    COLLECTION_NAME = "dna_rolemodel"
    VECTOR_SIZE     = 1024  # BAAI/bge-m3 출력 차원

    def __init__(self, qdrant_url: str = "http://localhost:6333"):
        self._embedder: Optional[SentenceTransformer] = None
        self._qdrant = QdrantClient(url=qdrant_url)
        self._llm = OpenAI()

        self._ensure_collection()
        self._upsert_rolemodels()

    # ──────────────────────────────────────────
    # 초기화 내부 메서드
    # ──────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """컬렉션이 없으면 생성합니다."""
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if self.COLLECTION_NAME not in existing:
            self._qdrant.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"  📦 Qdrant 컬렉션 생성: {self.COLLECTION_NAME}")

    def _upsert_rolemodels(self) -> None:
        """누락된 롤모델만 LLM으로 수집해 Qdrant에 적재합니다."""
        missing_companies: list[str] = []
        for company_name in _ROLEMODEL_COMPANIES:
            existing_count = self._qdrant.count(
                collection_name=self.COLLECTION_NAME,
                count_filter=Filter(
                    must=[FieldCondition(
                        key="company",
                        match=MatchValue(value=company_name),
                    )]
                ),
            ).count
            if existing_count > 0:
                print(f"  ⏭️  롤모델 이미 적재됨: {company_name}")
            else:
                missing_companies.append(company_name)

        if not missing_companies:
            return

        print(f"  🔎 롤모델 기업 정보 수집 시작: {', '.join(missing_companies)}")
        rolemodel_startups = self._fetch_rolemodel_startups(missing_companies)
        startup_map = {s.get("name", "").lower(): s for s in rolemodel_startups}

        for company_name in missing_companies:
            startup = startup_map.get(company_name.lower())
            if not startup:
                print(f"  ⚠️  수집 실패: {company_name}")
                continue

            startup_text = self._serialize_startup_info(startup)
            embedding = self.embedder.encode(
                [startup_text], normalize_embeddings=True, show_progress_bar=False
            ).tolist()[0]

            point_id = int(hashlib.md5(company_name.encode()).hexdigest(), 16) % (10 ** 18)

            self._qdrant.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "company": company_name,
                            "text": startup_text,
                            "startupInfo": startup,
                            "source": "searchcorp-compatible-llm-search",
                        },
                    )
                ],
            )
            print(f"  ✅ 적재 완료: {company_name}")

    def _fetch_rolemodel_startups(self, target_companies: list[str]) -> list[dict]:
        """
        searchCorp 의 _get_startup_list 와 동일한 2단계 패턴으로 롤모델 기업 정보를 수집합니다.
        1) gpt-4o-search-preview 로 웹 검색
        2) gpt-4o 로 StartupInfo 형태 JSON 정규화
        """
        schema_example = json.dumps({
            "startupId": "rolemodel_nvidia",
            "name": "NVIDIA",
            "foundedYear": "1993",
            "domain": "AI반도체",
            "location": "미국",
            "stage": "상장사",
            "team": {
                "founderCount": 3,
                "founderProfiles": [{
                    "name": "Jensen Huang",
                    "role": "CEO",
                    "education": "Oregon State University",
                    "priorCompanies": ["LSI Logic", "AMD"],
                    "domainYears": 10,
                    "hasExitExperience": False,
                }],
            },
            "funding": {
                "totalFunding": "미공개",
                "latestRound": "상장",
                "latestValuation": "시가총액 기준",
                "keyInvestors": ["Sequoia Capital"],
            },
            "traction": {
                "revenueYear": ["2024: ..."],
                "arrGrowthRate": "미공개",
                "keyCustomers": ["Meta", "Microsoft"],
            },
        }, ensure_ascii=False, indent=2)

        companies_text = ", ".join(target_companies)
        system_msg = {
            "role": "system",
            "content": (
                "당신은 반도체 산업 리서치 전문가입니다. "
                "지정된 기업만 조사하고, searchCorp StartupInfo 스키마로 구조화하세요. "
                "정보가 불확실하면 '미공개' 또는 빈 리스트를 사용하세요."
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                "다음 기업만 웹에서 조사해 StartupInfo 스키마로 정리해주세요.\n"
                f"대상 기업: {companies_text}\n\n"
                "요구 사항:\n"
                "1) 각 기업당 하나의 JSON 객체\n"
                "2) name 필드는 대상 기업명과 정확히 일치\n"
                "3) startupId는 rolemodel_ 접두어 사용 (예: rolemodel_nvidia)\n"
                "4) foundedYear, domain, location, stage, team, funding, traction 필드를 모두 포함\n"
                "5) 최종 응답은 JSON 배열만 출력\n"
            ),
        }

        search_response = self._llm.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[system_msg, user_msg],
        )
        search_content = search_response.choices[0].message.content or ""

        response = self._llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                system_msg,
                user_msg,
                {"role": "assistant", "content": search_content},
                {
                    "role": "user",
                    "content": (
                        "위 검색 결과를 StartupInfo 스키마 배열로 정규화하세요. "
                        "다른 텍스트 없이 JSON 배열만 반환하세요.\n\n"
                        f"스키마 예시:\n{schema_example}"
                    ),
                },
            ],
        )

        text = response.choices[0].message.content or ""
        start, end = text.find("["), text.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError(f"롤모델 StartupInfo JSON 파싱 실패: {text[:300]}")

        return json.loads(text[start:end])

    def _serialize_startup_info(self, startup: dict) -> str:
        """searchCorp StartupInfo와 동일 관점으로 임베딩용 텍스트를 구성합니다."""
        return json.dumps(
            {
                "startupId": startup.get("startupId", ""),
                "name": startup.get("name", ""),
                "foundedYear": startup.get("foundedYear", "미공개"),
                "domain": startup.get("domain", "미공개"),
                "location": startup.get("location", "미공개"),
                "stage": startup.get("stage", "미공개"),
                "team": startup.get("team", {"founderCount": 0, "founderProfiles": []}),
                "funding": startup.get(
                    "funding",
                    {
                        "totalFunding": "미공개",
                        "latestRound": "미공개",
                        "latestValuation": "미공개",
                        "keyInvestors": [],
                    },
                ),
                "traction": startup.get(
                    "traction",
                    {
                        "revenueYear": [],
                        "arrGrowthRate": "미공개",
                        "keyCustomers": [],
                    },
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print(f"  📥 임베딩 모델 로드: {self.MODEL_NAME}")
            self._embedder = SentenceTransformer(self.MODEL_NAME)
        return self._embedder

    def search_similar(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> list[ScoredPoint]:
        """
        query_text 를 임베딩하여 롤모델 3사와의 코사인 유사도를 반환합니다.

        Args:
            query_text: 스타트업 원본 데이터를 텍스트로 직렬화한 문자열
            top_k     : 반환할 최대 유사 롤모델 수 (기본 3 — 전체 롤모델 수와 동일)

        Returns:
            ScoredPoint 리스트 — score(코사인 유사도 0.0~1.0), payload["company"] 포함
        """
        total = self._qdrant.count(collection_name=self.COLLECTION_NAME).count
        if total == 0:
            return []

        query_embedding = self.embedder.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        ).tolist()[0]

        return self._qdrant.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_embedding,
            limit=min(top_k, total),
            with_payload=True,
        ).points
