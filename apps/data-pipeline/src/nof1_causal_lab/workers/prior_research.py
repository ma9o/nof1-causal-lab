"""model-spec Worker: Per-Parameter Prior Research.

Each worker researches a single parameter using:
1. Targeted Exa literature search (cacheable, run once)
2. LLM prior elicitation based on evidence (can be retried with feedback)
"""

from __future__ import annotations

import logging

import httpx

from nof1_causal_lab.utils.openrouter_client import acquire_limiter

logger = logging.getLogger(__name__)


async def search_parameter_literature(
    query: str,
) -> list[dict[str, str]]:
    """Search Exa for literature relevant to a query.

    This is separated from elicitation so results can be cached and reused
    across retry loops without re-hitting the Exa API.

    Uses Exa deep search with structured output schema.

    Args:
        query: Search query string

    Returns:
        List of source dicts with title, url, snippet, effect_size
    """
    from nof1_causal_lab.utils.config import get_secret_async

    api_key = await get_secret_async("EXA_API_KEY")
    if not api_key:
        return []

    try:
        from exa_py import AsyncExa

        exa = AsyncExa(api_key=api_key)

        exa_query = (
            f"Empirical effect sizes related to: {query}. "
            "Meta-analyses, systematic reviews, standardized effect sizes."
        )

        await acquire_limiter("exa")

        result = await exa.search_and_contents(
            exa_query,
            num_results=5,
            type="auto",
            highlights=True,
            category="research paper",
        )

        sources = []
        for r in result.results:
            sources.append(
                {
                    "title": r.title or "",
                    "url": r.url or "",
                    "snippet": " ".join(r.highlights) if r.highlights else (r.text or "")[:500],
                    "effect_size": "",
                }
            )
        return sources

    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Exa search failed; continuing without search results: %s", exc)
        return []
