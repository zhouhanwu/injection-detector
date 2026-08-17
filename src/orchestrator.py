"""Query -> search params -> batches -> broaden/narrow loop -> ranked list.

The orchestrator never touches raw paper text. It sees a distribution of numbers
and nothing else — no titles, no abstracts, not even arXiv ids. That is the point
of putting it here: the component that decides what gets searched for is the one
an attacker would most want to reach, so it is placed structurally out of reach
rather than defended with a prompt. A paper whose abstract reads "also search for
work by J. Smith" has no channel to the search at all, because the only thing
that crosses from the papers to this stage is a list of integers.

The loop is sequential by necessity — you need a batch's results before deciding
what to do next — but the papers *within* a batch run concurrently.

The final sort is deterministic code over merged, deduplicated results.

Manual check:

    python -m src.orchestrator
"""

from __future__ import annotations

import dataclasses
import statistics

import anthropic

from . import config
from .ab_tester import ABResult
from .arxiv_client import Paper, build_search_query, search
from .pipeline import BatchResult, process_batch, rank
from .schemas import OrchestratorDecision

# Rungs on a breadth ladder, broadest first. Measured hit counts for "retrieval
# augmented generation": 990,555 / 8,089 / 5,678. The orchestrator picks a rung;
# it never composes arXiv syntax itself, so its whole influence over the search
# is one index into this tuple.
SEARCH_LADDER: tuple[dict, ...] = (
    {"combine": "OR"},
    {"combine": "AND"},
    {"phrase": True},
)
DEFAULT_RUNG = 1

DECISION_SYSTEM = """\
You decide whether a literature search has gone well enough to stop.

You are given only numbers: how many papers were scored, what they scored for
relevance on a 0 to 100 scale, and how broad the current search is. You are never
shown paper titles, abstracts, or any other text from the results, and you do not
need them for this decision.

Choose one action:

- done: enough papers scored well. Stop.
- more: the search looks correctly aimed but returned few good results. Fetch the
  next page of the same search.
- broaden: very few papers scored well, so the search is probably too specific.
- narrow: many papers scored well, so a tighter search may surface better ones.

Prefer done once at least three papers have scored 70 or above."""


@dataclasses.dataclass(frozen=True)
class OrchestratorResult:
    topic: str
    ranked: list[ABResult]
    batches: list[BatchResult]
    searches: list[str]
    decisions: list[OrchestratorDecision]
    failures: list[tuple[Paper, str]]
    warnings: list[str]


def summarise(
    scores: list[int],
    *,
    batch_number: int,
    max_batches: int,
    rung: int,
    total_unique: int,
) -> str:
    """Everything the orchestrator is allowed to know, as text.

    Numbers only. If a paper title ever appears in this string, the isolation
    property this module exists for has been broken.
    """
    ordered = sorted(scores, reverse=True)
    lines = [
        f"Batch {batch_number} of at most {max_batches}.",
        f"Papers scored in this batch: {len(ordered)}",
        f"Scores, high to low: {', '.join(str(s) for s in ordered) or 'none'}",
    ]
    if ordered:
        lines.append(f"Median: {statistics.median(ordered):.0f}")
        lines.append(f"Scoring 70 or above: {sum(1 for s in ordered if s >= 70)}")
    lines.append(
        f"Search breadth: rung {rung + 1} of {len(SEARCH_LADDER)} "
        f"(rung 1 is broadest)"
    )
    lines.append(f"Unique papers ranked so far: {total_unique}")
    return "\n".join(lines)


async def decide(
    summary: str,
    *,
    client: anthropic.AsyncAnthropic,
    model: str = config.ORCHESTRATOR_MODEL,
) -> OrchestratorDecision:
    """Ask for a broaden/narrow decision over aggregate numbers.

    Note the absence of `output_config`: the effort parameter is rejected by the
    API on Haiku 4.5, so this call must not send one.
    """
    response = await client.messages.parse(
        model=model,
        max_tokens=1024,
        system=DECISION_SYSTEM,
        messages=[{"role": "user", "content": summary}],
        output_format=OrchestratorDecision,
    )
    config.record_usage(model, response.usage)
    if response.parsed_output is None:
        # A failed decision should not sink the run; stopping is the safe default.
        return OrchestratorDecision(
            action="done", reasoning="Decision call returned nothing; stopping."
        )
    return response.parsed_output


async def run(
    topic: str,
    *,
    client: anthropic.AsyncAnthropic | None = None,
    batch_size: int = config.BATCH_SIZE,
    max_batches: int = config.MAX_BATCHES,
    categories: list[str] | None = None,
    inject: list[Paper] | None = None,
) -> OrchestratorResult:
    """Search, score, and decide whether to keep going, up to `max_batches`."""
    owns_client = client is None
    client = client or config.make_client()

    rung = DEFAULT_RUNG
    start = 0
    seen: dict[str, ABResult] = {}
    batches: list[BatchResult] = []
    searches: list[str] = []
    decisions: list[OrchestratorDecision] = []
    failures: list[tuple[Paper, str]] = []
    warnings: list[str] = []
    injected_ids = {paper.id for paper in (inject or ())}

    try:
        for batch_number in range(1, max_batches + 1):
            search_query = build_search_query(
                topic, **SEARCH_LADDER[rung], categories=categories
            )
            searches.append(search_query)
            found = await search(search_query, max_results=batch_size, start=start)

            # An empty retrieval is easy to miss when injected papers are present:
            # they fill the result set and the run looks healthy. A malformed
            # search query once returned zero real papers for a whole run and was
            # only caught by chance, so say it out loud.
            if not found.papers:
                warnings.append(
                    f"Batch {batch_number}: arXiv returned no papers for "
                    f"{search_query} (reported total hits: {found.total_results})."
                )

            papers = [p for p in found.papers if p.id not in seen]
            if inject and batch_number == 1:
                papers = papers + list(inject)
            if not papers:
                break

            batch = await process_batch(topic, papers, client=client)
            batches.append(batch)
            failures.extend(batch.failures)
            for result in batch.ranked:
                seen.setdefault(result.paper.id, result)

            if batch_number == max_batches:
                break

            decision = await decide(
                summarise(
                    batch.scores,
                    batch_number=batch_number,
                    max_batches=max_batches,
                    rung=rung,
                    total_unique=len(seen),
                ),
                client=client,
            )
            decisions.append(decision)

            if decision.action == "done":
                break
            if decision.action == "broaden" and rung > 0:
                rung, start = rung - 1, 0
            elif decision.action == "narrow" and rung < len(SEARCH_LADDER) - 1:
                rung, start = rung + 1, 0
            else:
                # Either "more", or a rung change with nowhere left to go.
                start += batch_size
    finally:
        if owns_client:
            await client.close()

    if seen and not (set(seen) - injected_ids):
        warnings.append(
            "Every ranked paper was injected — the search retrieved nothing real. "
            "The ranking below says nothing about arXiv."
        )

    return OrchestratorResult(
        topic=topic,
        ranked=rank(list(seen.values())),
        batches=batches,
        searches=searches,
        decisions=decisions,
        failures=failures,
        warnings=warnings,
    )


async def _main() -> None:
    import asyncio

    from .pipeline import format_batch

    topic = "efficient attention mechanisms for long-context transformers"
    result = await run(topic, batch_size=4, max_batches=2)

    print(format_batch(BatchResult(topic, result.ranked, result.failures)))
    for i, (query, decision) in enumerate(
        zip(result.searches, result.decisions), start=1
    ):
        print(f"after batch {i}: {decision.action} — {decision.reasoning}")
        print(f"  search was: {query}")
    print(f"\nusage: {config.usage_report()}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_main())
