"""Background scheduler for automated knowledge base updates.

Uses APScheduler with AsyncIOScheduler to run updates every 3 hours.
Consumes pending_search_topics from Q&A fallback and resume GAP analysis first,
then runs the standard GAP analysis.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


async def _scheduled_update() -> None:
    """The scheduled update job that runs every 3 hours."""
    from wagent.mcp_servers.updater import KnowledgeUpdater

    logger.info("Scheduled knowledge base update starting...")

    updater = KnowledgeUpdater()

    # 1. Consume pending search topics (from Q&A fallback and resume GAP analysis)
    try:
        from wagent.cli.qa_session import get_pending_search_topics, _pending_search_topics

        pending = get_pending_search_topics()
        if pending:
            logger.info("Processing %d pending search topics", len(pending))
            report = await updater.fill_gaps(pending)
            logger.info(
                "Pending topics processed: searched=%d, added=%d",
                report["searched"],
                report["added"],
            )
            _pending_search_topics.clear()
    except Exception as e:
        logger.warning("Failed to process pending topics: %s", e)

    # 2. Run standard update
    try:
        report = await updater.run_update()
        logger.info(
            "Scheduled update complete: queries=%d, raw=%d, extracted=%d, added=%d",
            report["searched_queries"],
            report["raw_results"],
            report["relevant_extracted"],
            report["chunks_added"],
        )
    except Exception as e:
        logger.error("Scheduled update failed: %s", e)

    # 3. Review taxonomy — auto-accept in scheduled mode
    try:
        review = await updater.review_taxonomy(auto_accept=True)
        if review["accepted"] and review["proposed_topics"]:
            logger.info(
                "Taxonomy review: added %d new topics %s, starting reclassification",
                len(review["proposed_topics"]),
                review["proposed_topics"],
            )
            reclass = await updater.reclassify_general_chunks()
            logger.info(
                "Reclassification: reclassified=%d, still_general=%d",
                reclass["reclassified"],
                reclass["still_general"],
            )
        else:
            logger.info("Taxonomy review: no new topics added — %s", review["reason"])
    except Exception as e:
        logger.error("Taxonomy review/reclassification failed: %s", e)


def create_scheduler() -> AsyncIOScheduler:
    """Create and configure the APScheduler instance."""
    scheduler = AsyncIOScheduler()

    scheduler.add_job(
        _scheduled_update,
        trigger=CronTrigger(hour="*/6"),
        id="knowledge_update",
        name="Knowledge Base Update (every 6h)",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    return scheduler


async def run_scheduler() -> None:
    """Start the scheduler and run forever."""
    scheduler = create_scheduler()
    scheduler.start()
    logger.info(
        "Scheduler started. Knowledge base updates every 6 hours."
    )

    try:
        while True:
            await asyncio.sleep(3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        scheduler.shutdown()
        logger.info("Scheduler shut down.")
