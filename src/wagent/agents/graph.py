"""LangGraph interview graph – wires Router → Interviewer → Judge.

This is the "inner loop" that runs inside the Harness's outer control loop.
The graph handles agent reasoning and tool calls; the Harness handles
middleware, compaction, budget, and tracing.

Topology:
  START → phase_router
    ├─ "routing"      → route_node      → END
    ├─ "interviewing"  → interview_node  → END
    ├─ "judging"       → judge_node      → END
    └─ else            → END
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from wagent.agents.interviewer import INTERVIEWER_SYSTEM_PROMPT, _build_resume_section
from wagent.agents.judge import judge_interview
from wagent.agents.router import route_resume
from wagent.agents.schemas import InterviewPlan
from wagent.harness.state import InterviewState
from wagent.llm import get_llm

logger = logging.getLogger(__name__)


class GraphState(TypedDict, total=False):
    interview_state: Any          # InterviewState pydantic instance
    plan: Any                     # InterviewPlan instance
    phase: str                    # "routing" | "interviewing" | "judging" | "done"
    resume_text: str
    candidate_message: str
    current_topic: str
    retriever: Any                # HybridRetriever instance
    messages: list                # list[BaseMessage] from the harness
    response: str
    scorecard: Any                # InterviewScorecard instance
    input_tokens: int
    output_tokens: int
    knowledge_context: str


# ────────────────────── Node functions ──────────────────────

async def route_node(state: GraphState) -> dict:
    """Router node: parse resume and generate interview plan."""
    interview_state: InterviewState = state["interview_state"]
    resume_text = state.get("resume_text", "")

    plan = await route_resume(interview_state, resume_text)
    return {"plan": plan, "phase": "interviewing"}


async def interview_node(state: GraphState) -> dict:
    """Interview node: use plan + RAG context to generate a response."""
    interview_state: InterviewState = state["interview_state"]
    plan: InterviewPlan | None = state.get("plan")
    current_topic: str = state.get("current_topic", "general")
    retriever = state.get("retriever")
    messages: list[BaseMessage] = state.get("messages", [])

    kb_section = ""
    if retriever and current_topic:
        try:
            results = retriever.retrieve(current_topic, topic_filter=None)
            if results:
                snippets = [r["text"][:300] for r in results[:3]]
                kb_section = "相关知识库内容:\n" + "\n---\n".join(snippets)
        except Exception as e:
            logger.debug("RAG retrieval in interview_node failed: %s", e)

    if plan:
        mode = interview_state.progress.current_mode
        covered = interview_state.progress.covered_topics
        resume_section = _build_resume_section(interview_state.resume_text, mode)
        system_prompt = INTERVIEWER_SYSTEM_PROMPT.format(
            random_topics=plan.random_topics,
            mode=mode.value,
            current_topic=current_topic,
            covered=covered,
            q_count=interview_state.progress.questions_asked,
            total_questions=plan.total_questions,
            resume_section=resume_section,
            knowledge_context=kb_section,
        )
        enriched: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        enriched.extend(m for m in messages if not isinstance(m, SystemMessage))
    else:
        enriched = list(messages)
        if kb_section:
            enriched.append(SystemMessage(content=f"[知识库上下文]\n{kb_section}"))

    llm = get_llm(temperature=0.4, max_tokens=8192)
    response = await llm.ainvoke(enriched)

    input_tokens = response.response_metadata.get("token_usage", {}).get("prompt_tokens", 0)
    output_tokens = response.response_metadata.get("token_usage", {}).get("completion_tokens", 0)

    return {
        "response": response.content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


async def judge_node(state: GraphState) -> dict:
    """Judge node: generate final scorecard."""
    interview_state: InterviewState = state["interview_state"]
    scorecard = await judge_interview(interview_state)
    return {"scorecard": scorecard, "phase": "done"}


# ────────────────────── Graph construction ──────────────────────

def _phase_router(state: GraphState) -> str:
    """Conditional entry-point: dispatch to the correct node based on phase."""
    phase = state.get("phase", "routing")
    if phase == "routing":
        return "route"
    if phase == "interviewing":
        return "interview"
    if phase == "judging":
        return "judge"
    return END


def build_interview_graph():
    """Build and compile the LangGraph StateGraph for the interview pipeline."""
    graph = StateGraph(GraphState)

    graph.add_node("route", route_node)
    graph.add_node("interview", interview_node)
    graph.add_node("judge", judge_node)

    graph.add_conditional_edges(
        START,
        _phase_router,
        {"route": "route", "interview": "interview", "judge": "judge", END: END},
    )

    graph.add_edge("route", END)
    graph.add_edge("interview", END)
    graph.add_edge("judge", END)

    return graph.compile()


# ────────────────────── Harness-compatible wrapper ──────────────────────

def create_graph_agent(compiled_graph):
    """Return an AgentCallable that delegates to the LangGraph compiled graph.

    Signature: async (InterviewState, list[BaseMessage], dict) -> dict
    """
    async def agent_fn(
        state: InterviewState,
        messages: list[BaseMessage],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        graph_input: GraphState = {
            "interview_state": state,
            "plan": context.get("plan"),
            "current_topic": context.get("current_topic", "general"),
            "retriever": context.get("retriever"),
            "messages": messages,
            "phase": "interviewing",
            "response": "",
            "input_tokens": 0,
            "output_tokens": 0,
        }
        result = await compiled_graph.ainvoke(graph_input)
        return {
            "response": result.get("response", ""),
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
        }

    return agent_fn
