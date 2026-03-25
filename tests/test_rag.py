"""Unit tests for RAG modules: chunking, embeddings, store, retriever."""

import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------- Sentence Splitting (no model needed) ----------

class TestSentenceSplitting:
    def test_split_chinese(self):
        from wagent.rag.chunking import split_sentences
        text = "第一句话。第二句话！第三句话？"
        result = split_sentences(text)
        assert len(result) == 3

    def test_split_english(self):
        from wagent.rag.chunking import split_sentences
        text = "First sentence. Second sentence! Third?"
        result = split_sentences(text)
        assert len(result) == 3

    def test_split_mixed(self):
        from wagent.rag.chunking import split_sentences
        text = "Transformer 模型架构。Self-attention 机制非常重要！你怎么看？"
        result = split_sentences(text)
        assert len(result) == 3


# ---------- Embeddings (requires model download) ----------

class TestEmbeddings:
    @pytest.mark.slow
    def test_embed_texts_shape(self):
        from wagent.rag.embeddings import embed_texts
        vectors = embed_texts(["测试文本", "hello world"])
        assert vectors.shape == (2, 1024)

    @pytest.mark.slow
    def test_embed_query_shape(self):
        from wagent.rag.embeddings import embed_query
        vec = embed_query("什么是RAG？")
        assert vec.shape == (1024,)

    @pytest.mark.slow
    def test_embeddings_normalized(self):
        from wagent.rag.embeddings import embed_texts
        vectors = embed_texts(["测试"])
        norm = np.linalg.norm(vectors[0])
        assert abs(norm - 1.0) < 0.01


# ---------- Qdrant Store (local file mode) ----------

class TestQdrantStore:
    @pytest.mark.slow
    def test_ensure_collection(self):
        from wagent.rag.store import get_qdrant_client, ensure_collection, COLLECTION_NAME
        with tempfile.TemporaryDirectory() as tmp:
            from unittest.mock import patch
            with patch("wagent.config.Settings.qdrant_abs_path", new_callable=lambda: property(lambda self: Path(tmp))):
                from wagent.rag.store import get_qdrant_client
                from qdrant_client import QdrantClient
                client = QdrantClient(path=tmp)
                ensure_collection(client)
                names = [c.name for c in client.get_collections().collections]
                assert COLLECTION_NAME in names

    @pytest.mark.slow
    def test_add_and_search(self):
        from wagent.rag.embeddings import embed_texts
        from wagent.rag.store import add_chunks, search, ensure_collection, COLLECTION_NAME
        from qdrant_client import QdrantClient

        with tempfile.TemporaryDirectory() as tmp:
            client = QdrantClient(path=tmp)
            ensure_collection(client)

            texts = [
                "RAG 检索增强生成是一种结合检索和生成的技术",
                "Transformer 的注意力机制基于 Query Key Value",
                "向量数据库用于存储和检索高维向量",
            ]
            metas = [
                {"source": "manual", "topic": "rag_pipeline"},
                {"source": "manual", "topic": "transformer_architecture"},
                {"source": "manual", "topic": "vector_database"},
            ]
            embeddings = embed_texts(texts)
            n = add_chunks(client, texts, metas, embeddings)
            assert n == 3

            results = search(client, "RAG 是什么", top_k=2)
            assert len(results) > 0
            assert results[0]["topic"] in ["rag_pipeline", "transformer_architecture", "vector_database"]

    @pytest.mark.slow
    def test_check_duplicate(self):
        from wagent.rag.embeddings import embed_texts
        from wagent.rag.store import add_chunks, check_duplicate, ensure_collection
        from qdrant_client import QdrantClient

        with tempfile.TemporaryDirectory() as tmp:
            client = QdrantClient(path=tmp)
            ensure_collection(client)

            text = "RAG 检索增强生成是结合信息检索与大语言模型的一种技术范式"
            embeddings = embed_texts([text])
            add_chunks(client, [text], [{"source": "manual"}], embeddings)

            assert check_duplicate(client, text, threshold=0.90) is True
            assert check_duplicate(client, "完全不相关的话题：今天天气很好", threshold=0.95) is False


# ---------- HybridRetriever BM25 ----------

class TestHybridRetrieverBM25:
    def test_build_bm25_index(self):
        from wagent.rag.retriever import HybridRetriever
        retriever = HybridRetriever()
        docs = [
            {"text": "RAG 检索增强生成"},
            {"text": "Transformer 自注意力机制"},
            {"text": "向量数据库 Qdrant"},
        ]
        retriever.build_bm25_index(docs)
        assert retriever._bm25_index is not None

    def test_bm25_search(self):
        from wagent.rag.retriever import HybridRetriever
        retriever = HybridRetriever()
        docs = [
            {"text": "RAG 检索增强生成技术"},
            {"text": "Transformer 自注意力机制原理"},
            {"text": "向量数据库 Qdrant 使用方法"},
        ]
        retriever.build_bm25_index(docs)
        results = retriever.bm25_search("RAG 检索", top_k=2)
        assert len(results) > 0
        assert "RAG" in results[0]["text"]

    def test_bm25_empty_without_index(self):
        from wagent.rag.retriever import HybridRetriever
        retriever = HybridRetriever()
        results = retriever.bm25_search("anything")
        assert results == []


# ---------- Agent Schemas ----------

class TestSchemas:
    def test_interview_scorecard_serialization(self):
        from wagent.agents.schemas import InterviewScorecard, TopicScoreItem
        sc = InterviewScorecard(
            overall_score=7.5,
            topic_scores=[TopicScoreItem(topic="rag", score=8.0, notes="good")],
            strengths=["deep understanding"],
            weaknesses=["lacks practical experience"],
            recommendation="hire",
            summary="Good candidate",
            battle_scars_index=6.0,
            first_principles_score=7.0,
            star_completeness=5.5,
            followup_resilience=7.0,
            deep_analysis="Strong episode on RAG project.",
        )
        d = sc.model_dump()
        assert d["battle_scars_index"] == 6.0
        assert d["followup_resilience"] == 7.0
        sc2 = InterviewScorecard(**d)
        assert sc2.overall_score == 7.5

    def test_interview_plan_serialization(self):
        from wagent.agents.schemas import InterviewPlan
        plan = InterviewPlan(
            resume_topics=["rag_pipeline", "agent_architecture"],
            random_topics=["llm_training"],
            resume_question_count=10,
            random_question_count=5,
        )
        d = plan.model_dump()
        assert len(d["resume_topics"]) == 2
        plan2 = InterviewPlan(**d)
        assert plan2.total_questions == 15

    def test_answer_evaluation_defaults(self):
        from wagent.agents.schemas import AnswerEvaluation
        ae = AnswerEvaluation(score=5.0)
        assert ae.depth == "basic"
        assert ae.should_follow_up is False
