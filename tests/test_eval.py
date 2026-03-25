"""Evaluation layer tests — validates the evaluation runner infrastructure
using mock agents so no real LLM API connection is needed."""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wagent.evaluation.runner import EvaluationRunner, DUMMY_RESUME
from wagent.evaluation.virtual_candidate import VirtualCandidate, CANDIDATE_PROFILES
from wagent.agents.schemas import (
    AnswerEvaluation,
    InterviewPlan,
    InterviewScorecard,
    SkillExtraction,
    TopicScoreItem,
)
from wagent.harness.state import InterviewState


class TestVirtualCandidate:
    def test_all_profiles_exist(self):
        for profile in ["expert", "average", "poor"]:
            vc = VirtualCandidate(profile=profile)
            assert vc.profile == profile

    def test_invalid_profile_raises(self):
        with pytest.raises(ValueError):
            VirtualCandidate(profile="nonexistent")

    def test_profile_prompts_differ(self):
        prompts = {VirtualCandidate(p)._system_prompt for p in ["expert", "average", "poor"]}
        assert len(prompts) == 3


class TestEvaluationRunner:
    @pytest.mark.asyncio
    async def test_single_run_with_mocks(self):
        """Run a single evaluation with mocked LLM calls."""
        runner = EvaluationRunner(output_dir=tempfile.mkdtemp())

        mock_plan = InterviewPlan(
            resume_topics=["rag_pipeline", "agent_architecture"],
            random_topics=["llm_training"],
            resume_question_count=2,
            random_question_count=1,
        )
        mock_extraction = SkillExtraction(
            candidate_name="TestCandidate",
            mapped_topics=["rag_pipeline", "agent_architecture"],
        )
        mock_eval = AnswerEvaluation(
            score=7.0, depth="intermediate", should_follow_up=True
        )
        mock_scorecard = InterviewScorecard(
            overall_score=7.5,
            topic_scores=[TopicScoreItem(topic="rag", score=8.0)],
            strengths=["Good RAG knowledge"],
            weaknesses=["Lacks depth in Agent systems"],
            recommendation="hire",
            summary="Good candidate overall.",
            battle_scars_index=6.0,
            first_principles_score=7.0,
            star_completeness=5.0,
            followup_resilience=7.5,
        )

        with patch("wagent.evaluation.runner.route_resume", new_callable=AsyncMock) as mock_route, \
             patch("wagent.evaluation.runner.generate_question", new_callable=AsyncMock) as mock_gen_q, \
             patch("wagent.evaluation.runner.evaluate_answer", new_callable=AsyncMock) as mock_eval_ans, \
             patch("wagent.evaluation.runner.judge_interview", new_callable=AsyncMock) as mock_judge, \
             patch("wagent.evaluation.runner.interviewer_turn", new_callable=AsyncMock) as mock_turn:

            mock_route.return_value = mock_plan
            mock_route.side_effect = lambda state, text: _setup_state_and_return(state, mock_plan)
            mock_gen_q.return_value = "请解释 RAG 的工作原理。"
            mock_eval_ans.return_value = mock_eval
            mock_judge.return_value = mock_scorecard
            mock_turn.return_value = {
                "response": "好的，下一个问题：Transformer 的自注意力机制是什么？",
                "input_tokens": 100,
                "output_tokens": 50,
            }

            with patch.object(VirtualCandidate, "answer", new_callable=AsyncMock) as mock_answer:
                mock_answer.return_value = "RAG 是检索增强生成，结合了信息检索和大模型。"

                result = await runner.run_single(
                    profile="average",
                    resume_text=DUMMY_RESUME,
                    max_turns=3,
                )

        assert result["profile"] == "average"
        assert result["questions_asked"] >= 1
        assert "scorecard" in result
        assert result["scorecard"]["overall_score"] == 7.5
        assert result["scorecard"]["battle_scars_index"] == 6.0
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_batch_run_with_mocks(self):
        """Run batch evaluation and verify report output."""
        tmpdir = tempfile.mkdtemp()
        runner = EvaluationRunner(output_dir=tmpdir)

        mock_plan = InterviewPlan(
            resume_topics=["rag_pipeline"],
            random_topics=["llm_training"],
        )
        mock_scorecard = InterviewScorecard(
            overall_score=5.0,
            recommendation="lean_hire",
            summary="Average candidate.",
        )

        with patch("wagent.evaluation.runner.route_resume", new_callable=AsyncMock) as mock_route, \
             patch("wagent.evaluation.runner.generate_question", new_callable=AsyncMock) as mock_gen_q, \
             patch("wagent.evaluation.runner.evaluate_answer", new_callable=AsyncMock) as mock_eval_ans, \
             patch("wagent.evaluation.runner.judge_interview", new_callable=AsyncMock) as mock_judge, \
             patch("wagent.evaluation.runner.interviewer_turn", new_callable=AsyncMock) as mock_turn:

            mock_route.side_effect = lambda state, text: _setup_state_and_return(state, mock_plan)
            mock_gen_q.return_value = "请解释 Transformer。"
            mock_eval_ans.return_value = AnswerEvaluation(score=5.0)
            mock_judge.return_value = mock_scorecard
            mock_turn.return_value = {
                "response": "下一题",
                "input_tokens": 50,
                "output_tokens": 25,
            }

            with patch.object(VirtualCandidate, "answer", new_callable=AsyncMock) as mock_answer:
                mock_answer.return_value = "Transformer 是一种深度学习模型。"

                report = await runner.run_batch(
                    profiles=["expert", "poor"],
                    runs_per_profile=1,
                    max_turns=2,
                )

        assert report["total_runs"] == 2
        assert "expert" in report["per_profile"]
        assert "poor" in report["per_profile"]

        report_files = list(Path(tmpdir).glob("*.json"))
        assert len(report_files) == 1
        with open(report_files[0]) as f:
            data = json.load(f)
        assert "report" in data
        assert "results" in data
        assert len(data["results"]) == 2


def _setup_state_and_return(state: InterviewState, plan: InterviewPlan) -> InterviewPlan:
    """Helper to set up state when route_resume is called."""
    state.resume_topics = plan.resume_topics
    state.progress.pending_resume_topics = list(plan.resume_topics)
    state.progress.current_phase = "interviewing"
    return plan
