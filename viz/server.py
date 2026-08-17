"""Local server that runs the real pipeline and streams each stage to a browser.

    .venv/bin/python -m viz.server
    open http://localhost:8000

Nothing here is a re-implementation or a replay. Every stage calls the same
functions the CLI does — `catch_sus`, `run_ab_test`, `rank`, `decide` — and the
events are emitted between those calls. What the page shows is a real run against
live arXiv, at real speed.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from aiohttp import web

from src import config
from src.ab_tester import run_ab_test
from src.arxiv_client import (
    _QUERY_FILLER,
    _STOPWORDS,
    _normalise_term,
    build_search_query,
    search,
)
from src.main import INJECTION_POOL
from src.orchestrator import SEARCH_LADDER, DEFAULT_RUNG, decide, summarise
from src.pipeline import rank
from src.sus_catcher import catch_sus

HERE = Path(__file__).parent


def term_breakdown(topic: str) -> dict:
    """Which words survived into the query, and why the others didn't."""
    kept, dropped = [], []
    words = [w for w in topic.split() if w]
    long_query = len([w for w in words if _normalise_term(w) not in _STOPWORDS]) > 8
    for word in words:
        norm = _normalise_term(word)
        if norm in _STOPWORDS:
            dropped.append({"word": word, "why": "function word"})
        elif long_query and norm in _QUERY_FILLER:
            dropped.append({"word": word, "why": "query boilerplate"})
        else:
            kept.append(word)
    return {"kept": kept, "dropped": dropped}


async def run_stream(request: web.Request) -> web.StreamResponse:
    topic = (request.query.get("q") or "").strip()
    inject_count = int(request.query.get("inject") or 0)
    batch_size = int(request.query.get("size") or 6)

    response = web.StreamResponse(
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )
    await response.prepare(request)

    lock = asyncio.Lock()

    async def emit(stage: str, **payload) -> None:
        # Per-paper tasks run concurrently and all write to one stream.
        async with lock:
            body = json.dumps({"stage": stage, **payload})
            await response.write(f"data: {body}\n\n".encode())

    if not topic:
        await emit("error", message="empty query")
        return response

    config.reset_usage()
    started = time.monotonic()

    try:
        # 1. query construction — literal terms, no model involved
        breakdown = term_breakdown(topic)
        arxiv_query = build_search_query(topic, **SEARCH_LADDER[DEFAULT_RUNG])
        await emit("query", topic=topic, arxiv_query=arxiv_query, **breakdown)

        # 2. retrieval
        found = await search(arxiv_query, max_results=batch_size)
        papers = list(found.papers)
        injected = INJECTION_POOL[:inject_count]
        papers.extend(injected)
        await emit(
            "retrieval",
            total_hits=found.total_results,
            injected=len(injected),
            papers=[
                {"id": p.id, "title": p.title, "injected": p.id.startswith("INJECTED-")}
                for p in papers
            ],
        )
        if not papers:
            await emit("done", note="no papers retrieved")
            return response

        client = config.make_client()

        async def one(paper) -> object:
            await emit("paper_start", id=paper.id)
            sus = await catch_sus(paper, client=client)
            await emit(
                "sus",
                id=paper.id,
                ratio=round(sus.sus_ratio, 2),
                units=[
                    {
                        "index": s.index,
                        "text": sus.text_at(s.index),
                        "label": s.label,
                        "suspicious": s.is_suspicious,
                        "reasoning": s.reasoning if s.is_suspicious else "",
                    }
                    for s in sus.classification.sentences
                ],
            )
            result = await run_ab_test(topic, sus, client=client)
            await emit(
                "scored",
                id=paper.id,
                probe_original=result.probe_original,
                probe_stripped=result.probe_stripped,
                rank_original=result.rank_original,
                base=result.base_score,
                potency=result.attack_potency,
                residual=result.residual,
                penalty=result.penalty,
                adjusted=result.adjusted_score,
                penalised=result.penalised,
                decision=result.decision,
                reasoning=result.base_reasoning,
            )
            return result

        try:
            results = await asyncio.gather(*(one(p) for p in papers))

            # 3. orchestrator — numbers only, and the page shows exactly that
            ordered = rank(list(results))
            digest = summarise(
                [r.adjusted_score for r in ordered],
                batch_number=1,
                max_batches=1,
                rung=DEFAULT_RUNG,
                total_unique=len(ordered),
            )
            await emit("orchestrator_input", text=digest)
            decision = await decide(digest, client=client)
            await emit(
                "orchestrator", action=decision.action, reasoning=decision.reasoning
            )
        finally:
            await client.close()

        # 4. deterministic sort
        await emit(
            "ranked",
            papers=[
                {
                    "id": r.paper.id,
                    "title": r.paper.title,
                    "url": r.paper.url,
                    "adjusted": r.adjusted_score,
                    "base": r.base_score,
                    "penalty": r.penalty,
                    "flagged": r.sus.has_flags,
                    "injected": r.paper.id.startswith("INJECTED-"),
                }
                for r in ordered
            ],
        )

        totals, cost = config.usage_report()
        await emit(
            "done",
            calls=sum(e["calls"] for e in totals.values()),
            cost=round(cost, 3),
            seconds=round(time.monotonic() - started, 1),
        )
    except Exception as exc:  # keep the page informed rather than hanging
        await emit("error", message=f"{type(exc).__name__}: {exc}")

    return response


async def index(_: web.Request) -> web.Response:
    return web.Response(
        text=(HERE / "index.html").read_text(), content_type="text/html"
    )


def main() -> None:
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/run", run_stream)
    print("injection-detector  ->  http://localhost:8000")
    web.run_app(app, host="localhost", port=8000, print=None)


if __name__ == "__main__":
    main()
