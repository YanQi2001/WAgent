"""Interactive Q&A session – knowledge query mode with RAG + optional web search fallback."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from rich.console import Console
from rich.panel import Panel

from wagent.cli.prompt_utils import prompt_input
from wagent.llm import get_llm
from wagent.rag.retriever import HybridRetriever
from wagent.rag.store import COLLECTION_NAME, collection_stats, ensure_collection, get_qdrant_client

logger = logging.getLogger(__name__)
console = Console()

QA_SYSTEM_PROMPT = """\
你是一位 AI/大模型技术领域的知识助手。

回答策略（按优先级）：
1. 如果提供了参考资料（context），以参考资料为起点，结合你自身的知识给出完整、深入的回答。
   参考资料可能不完全匹配问题，不必局限于参考资料的范围，来自参考资料的部分标注来源编号即可
2. 如果参考资料不足以完整回答，先引用已有资料，再大量补充你自身的知识，
   确保回答完整有深度
3. 如果完全没有参考资料，直接用你自身的知识回答，
   但在开头声明「以下回答基于模型自身知识，未经知识库验证」

核心规则：
- 使用中文回答
- 回答要准确、有深度、结构清晰
- 对不确定的部分明确标注
- 绝不编造不存在的论文或具体数据
- 每次只回答用户最新提出的问题，历史对话仅作为背景参考
"""

RAG_SCORE_THRESHOLD = 0.3


def _init_retriever() -> HybridRetriever:
    """Initialize the hybrid retriever with BM25 index."""
    retriever = HybridRetriever()
    try:
        client = get_qdrant_client()
        ensure_collection(client)
        total = collection_stats(client).get("total_points") or 2000
        points, _ = client.scroll(
            collection_name=COLLECTION_NAME, limit=total, with_payload=True
        )
        if points:
            docs = [
                {"text": p.payload.get("text", ""), **p.payload}
                for p in points
                if p.payload.get("text")
            ]
            retriever.build_bm25_index(docs)
            console.print(f"[green]知识库已加载[/green] ({len(docs)} chunks)")
        else:
            console.print("[yellow]知识库为空[/yellow]")
    except Exception as e:
        console.print(f"[yellow]知识库加载失败: {e}[/yellow]")
    return retriever


def _retrieve_and_format(retriever: HybridRetriever, question: str) -> tuple[str, float]:
    """Retrieve relevant context and return (formatted_context, max_score).

    Returns empty string and 0.0 if nothing relevant found.
    """
    try:
        results = retriever.retrieve(question, topic_filter=None)
        if not results:
            return "", 0.0

        max_score = max(r.get("rerank_score", r.get("rrf_score", 0.0)) for r in results)
        snippets = []
        for i, r in enumerate(results[:5], 1):
            source = r.get("source", "unknown")
            topic = r.get("topic", "general")
            text = r["text"][:500]
            snippets.append(f"[来源{i}: {source}/{topic}]\n{text}")

        context = "\n\n---\n\n".join(snippets)
        return context, max_score
    except Exception as e:
        logger.debug("QA retrieval failed: %s", e)
        return "", 0.0


async def _web_search_fallback(question: str) -> str:
    """Search Bing + Xiaohongshu with a tech-domain prefix and return merged context.

    Also triggers PDF auto-download + ingest for any PDF URLs found in Bing results.
    """
    tech_query = f"AI 大模型面试 {question}"
    all_results: list[dict[str, Any]] = []

    # Bing search
    try:
        from wagent.mcp_servers.bing_server import BingMCPServer
        bing_results = await BingMCPServer().search(tech_query, max_results=3)
        all_results.extend(bing_results)

        if bing_results:
            try:
                from wagent.mcp_servers.pdf_downloader import process_search_results_for_pdfs
                pdf_reports = await process_search_results_for_pdfs(bing_results)
                for r in pdf_reports:
                    if r["success"]:
                        console.print(
                            f"[green]自动发现并入库 PDF: {r['path']} (+{r['chunks_added']} chunks)[/green]"
                        )
            except Exception as e:
                logger.debug("PDF auto-ingest check failed: %s", e)
    except ImportError:
        logger.debug("Bing MCP not available")
    except Exception as e:
        logger.warning("Bing search failed: %s", e)

    # Xiaohongshu search
    try:
        from wagent.mcp_servers.xiaohongshu_server import XiaohongshuMCPServer
        xhs_results = await XiaohongshuMCPServer().search(tech_query, max_results=3)
        all_results.extend(xhs_results)
    except ImportError:
        logger.debug("Xiaohongshu MCP not available")
    except Exception as e:
        logger.warning("Xiaohongshu search failed: %s", e)

    if not all_results:
        return ""

    snippets = []
    for r in all_results[:6]:
        title = r.get("title", "")
        snippet = r.get("snippet", "") or r.get("content", "")
        url = r.get("url", "")
        if title or snippet:
            snippets.append(f"[{title}]({url})\n{snippet[:500]}")
    return "\n\n---\n\n".join(snippets)


FOCUS_INSTRUCTION = "请只回答用户最新提出的问题，不要重复之前已经回答过的内容。如果当前问题引用了之前的对话，可以简要提及但不要完整重复。"
HISTORY_SUMMARY_INSTRUCTION = "以下是之前的对话历史摘要（仅供参考，不要重复回答这些问题）："


async def _answer_question(
    question: str,
    context: str,
    has_context: bool,
    conversation_history: list[dict[str, str]],
) -> str:
    """Generate an answer using the LLM with optional RAG context.

    Uses Hybrid Memory: older turns are compressed to 1-line summaries,
    recent 2 turns are kept with truncated answers, and a FOCUS instruction
    ensures the model only answers the latest question.
    """
    llm = get_llm(temperature=0.2, max_tokens=8192)

    messages = [SystemMessage(content=QA_SYSTEM_PROMPT + "\n\n" + FOCUS_INSTRUCTION)]

    if len(conversation_history) > 2:
        older = conversation_history[:-2]
        summary_lines = []
        for turn in older[-6:]:
            q_short = turn["question"][:80]
            a_short = turn["answer"][:100].replace("\n", " ")
            summary_lines.append(f"- Q: {q_short} → A: {a_short}...")
        messages.append(SystemMessage(
            content=f"{HISTORY_SUMMARY_INSTRUCTION}\n" + "\n".join(summary_lines)
        ))

    recent = conversation_history[-2:]
    for turn in recent:
        messages.append(HumanMessage(content=turn["question"]))
        a_truncated = turn["answer"][:500] + ("..." if len(turn["answer"]) > 500 else "")
        messages.append(AIMessage(content=a_truncated))

    if has_context:
        messages.append(SystemMessage(
            content=(
                "以下是从知识库检索到的参考资料，可能不完全匹配当前问题。"
                "请结合参考资料和你自身的知识给出完整回答；"
                "来自参考资料的部分标注来源编号，来自自身知识的部分说明即可。\n\n"
                f"{context}"
            )
        ))
    else:
        messages.append(SystemMessage(content="当前没有参考资料，请根据回答策略第 3 条，用你自身的知识回答。"))

    messages.append(HumanMessage(content=question))

    response = await llm.ainvoke(messages)
    return response.content


# Pending topics queue shared across the system
_pending_search_topics: list[str] = []


def get_pending_search_topics() -> list[str]:
    return list(_pending_search_topics)


def add_pending_search_topic(topic: str) -> None:
    if topic not in _pending_search_topics:
        _pending_search_topics.append(topic)


async def run_qa_session() -> None:
    """Run an interactive Q&A session."""
    console.print(Panel("[bold]AI 知识问答模式[/bold] — 基于 RAG 的技术问答", style="blue"))
    console.print("输入技术问题获取答案。输入 /end 退出，输入 /interview 切换到面试模式。\n")

    retriever = _init_retriever()
    conversation_history: list[dict[str, str]] = []

    while True:
        question = await prompt_input("你的问题", "qa")
        question = question.strip()

        if not question:
            continue

        if question.lower() == "/end":
            console.print("[yellow]退出问答模式[/yellow]")
            break

        if question.lower() == "/interview":
            console.print("[yellow]切换到面试模式...[/yellow]")
            return  # caller handles mode switch

        # RAG retrieval
        console.print("[dim]正在检索知识库...[/dim]")
        context, max_score = _retrieve_and_format(retriever, question)

        has_context = bool(context) and max_score > RAG_SCORE_THRESHOLD

        if not has_context:
            console.print("[yellow]知识库中未找到高相关度内容。[/yellow]")

            from wagent.cli.smart_prompt import prompt_and_parse
            decision = await prompt_and_parse(
                "[bold green]是否联网搜索？[/bold green]（可输入：搜一下 / 不用，直接回答 / 跳过）",
                situation="用户提问了一个技术问题，但知识库中未找到高相关度内容。可以联网搜索补充信息。",
                actions=[
                    {"id": "search", "desc": "联网搜索补充信息"},
                    {"id": "skip", "desc": "不搜索，直接基于已有知识回答"},
                ],
            )

            if decision["action"] == "search":
                console.print("[dim]正在联网搜索...[/dim]")
                web_context = await _web_search_fallback(question)
                if web_context:
                    context = web_context
                    has_context = True
                    console.print("[green]联网搜索完成，已获取补充资料[/green]")
                else:
                    console.print("[yellow]联网搜索也未找到相关内容[/yellow]")

            add_pending_search_topic(question)

        # Generate answer
        console.print("[dim]正在生成回答...[/dim]")
        answer = await _answer_question(question, context, has_context, conversation_history)

        console.print(Panel(answer, title="回答", style="cyan"))

        conversation_history.append({"question": question, "answer": answer})

    if _pending_search_topics:
        console.print(
            f"\n[dim]已记录 {len(_pending_search_topics)} 个待补充知识点，"
            f"将在下次知识库更新时优先搜索[/dim]"
        )
