"""Evaluation runner – automated interview simulation with metrics collection.

Runs N simulated interviews with virtual candidates of different profiles,
collects metrics from the Harness tracer, and produces an evaluation report.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from wagent.agents.graph import build_interview_graph, create_graph_agent
from wagent.agents.interviewer import evaluate_answer, generate_question
from wagent.agents.schemas import InterviewPlan
from wagent.evaluation.virtual_candidate import VirtualCandidate
from wagent.harness.harness import InterviewHarness
from wagent.harness.state import InterviewState, QAPair, QuestionMode
from wagent.rag.retriever import HybridRetriever
from wagent.rag.store import get_qdrant_client, ensure_collection, collection_stats, COLLECTION_NAME

logger = logging.getLogger(__name__)

DUMMY_RESUME = """\
张三 - AI 工程师
教育: 某大学计算机科学硕士
技能: Python, PyTorch, Transformer, RAG系统开发, LangChain, 向量数据库
项目经历:
1. 基于 RAG 的智能客服系统 - 使用 Qdrant + BGE embedding 构建知识检索
2. 多 Agent 协作框架 - 基于 LangGraph 实现任务分解与协同
3. LLM 微调平台 - LoRA 微调 + 评估流水线
"""


class EvaluationRunner:
    """Run automated interview simulations and collect metrics."""

    def __init__(self, output_dir: str = "logs/evaluations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_single(
        self,
        *,
        profile: str = "average",
        resume_text: str = DUMMY_RESUME,
        max_turns: int = 10,
    ) -> dict[str, Any]:
        """Run a single simulated interview."""
        candidate = VirtualCandidate(profile=profile)
        state = InterviewState()

        # Build LangGraph graph
        compiled_graph = build_interview_graph()

        # Route phase
        route_result = await compiled_graph.ainvoke({
            "interview_state": state,
            "resume_text": resume_text,
            "phase": "routing",
        })
        plan: InterviewPlan = route_result["plan"]

        # Set up harness with graph agent
        harness = InterviewHarness()
        graph_agent = create_graph_agent(compiled_graph)
        harness.set_agent(graph_agent)
        harness.set_system_prompt(
            "你是一位资深的 AI/大模型技术面试官。使用中文提问。每次问一个聚焦的技术问题。"
        )

        retriever = HybridRetriever()
        try:
            client = get_qdrant_client()
            ensure_collection(client)
            total = collection_stats(client).get("total_points") or 2000
            points, _ = client.scroll(
                collection_name=COLLECTION_NAME, limit=total, with_payload=True
            )
            if points:
                docs = [{"text": p.payload.get("text", ""), **p.payload} for p in points if p.payload.get("text")]
                retriever.build_bm25_index(docs)
        except Exception:
            pass

        current_topic = plan.resume_topics[0] if plan.resume_topics else "general"
        harness.set_agent_context(plan=plan, retriever=retriever, current_topic=current_topic)

        kb_ctx = self._retrieve_context(retriever, current_topic)
        question = await generate_question(state, plan, knowledge_context=kb_ctx)

        for turn in range(max_turns):
            context = "\n".join(
                f"Q: {qa.question}\nA: {qa.answer}" for qa in state.qa_history[-3:]
            )
            answer = await candidate.answer(question, context)

            response = await harness.turn(state, answer)

            evaluation = await evaluate_answer(question, answer, current_topic)
            qa = QAPair(
                question=question,
                answer=answer,
                topic=current_topic,
                mode=state.progress.current_mode,
                score=evaluation.score,
            )
            state.qa_history.append(qa)
            state.progress.questions_asked += 1

            if current_topic not in state.progress.covered_topics:
                state.progress.covered_topics.append(current_topic)

            should_end = await harness.suggest_end(state)
            if should_end:
                break

            if not evaluation.should_follow_up or evaluation.score < 6:
                if state.progress.current_mode == QuestionMode.RESUME_DRIVEN:
                    pending = [t for t in plan.resume_topics if t not in state.progress.covered_topics]
                    current_topic = pending[0] if pending else "general"
                else:
                    remaining = [t for t in plan.random_topics if t not in state.progress.covered_topics]
                    current_topic = remaining[0] if remaining else "general"

            harness.set_agent_context(current_topic=current_topic)

            kb_ctx = self._retrieve_context(retriever, current_topic)
            question = response if response and not response.startswith("[面试系统]") else await generate_question(state, plan, knowledge_context=kb_ctx)

        # Judge via LangGraph
        judge_result = await compiled_graph.ainvoke({
            "interview_state": state,
            "phase": "judging",
        })
        scorecard = judge_result["scorecard"]

        harness_summary = harness.finalize(state)

        result = {
            "profile": profile,
            "questions_asked": state.progress.questions_asked,
            "topics_covered": state.progress.covered_topics,
            "coverage_ratio": state.covered_topic_ratio,
            "mode_distribution": {
                "resume_driven": sum(1 for qa in state.qa_history if qa.mode == QuestionMode.RESUME_DRIVEN),
                "random_bagu": sum(1 for qa in state.qa_history if qa.mode == QuestionMode.RANDOM_BAGU),
            },
            "avg_score": sum(qa.score for qa in state.qa_history if qa.score) / max(len(state.qa_history), 1),
            "scorecard": scorecard.model_dump(),
            "harness_metrics": harness_summary,
            "timestamp": datetime.now().isoformat(),
        }

        return result

    @staticmethod
    def _retrieve_context(retriever: HybridRetriever, topic: str) -> str:
        """Retrieve knowledge context for a topic via hybrid search."""
        try:
            results = retriever.retrieve(topic, topic_filter=None)
            if results:
                snippets = [r["text"][:300] for r in results[:3]]
                return "\n---\n".join(snippets)
        except Exception:
            pass
        return ""

    async def run_batch(
        self,
        *,
        profiles: list[str] | None = None,
        runs_per_profile: int = 1,
        max_turns: int = 8,
    ) -> dict[str, Any]:
        """Run batch evaluations across multiple candidate profiles."""
        profiles = profiles or ["expert", "average", "poor"]
        all_results = []

        for profile in profiles:
            for i in range(runs_per_profile):
                logger.info("Eval run: profile=%s, run=%d/%d", profile, i + 1, runs_per_profile)
                result = await self.run_single(profile=profile, max_turns=max_turns)
                all_results.append(result)

        report = {
            "total_runs": len(all_results),
            "profiles_tested": profiles,
            "per_profile": {},
            "timestamp": datetime.now().isoformat(),
        }

        for profile in profiles:
            profile_results = [r for r in all_results if r["profile"] == profile]
            if profile_results:
                report["per_profile"][profile] = {
                    "runs": len(profile_results),
                    "avg_score": sum(r["avg_score"] for r in profile_results) / len(profile_results),
                    "avg_coverage": sum(r["coverage_ratio"] for r in profile_results) / len(profile_results),
                    "avg_questions": sum(r["questions_asked"] for r in profile_results) / len(profile_results),
                    "recommendations": [
                        r["scorecard"].get("recommendation", "N/A") for r in profile_results
                    ],
                }

        report_path = self.output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"report": report, "results": all_results}, f, ensure_ascii=False, indent=2)

        logger.info("Evaluation report saved to %s", report_path)
        return report
