"""arXiv API search -> candidate papers (title, abstract, id, url).

Plain-text metadata only; no PDF fetching. The arXiv API returns Atom XML whose
`title` and `summary` fields are already plain text, which is the whole data
surface this system defends. (A consequence worth being precise about: there is
no visual layer here, so white-on-white-text tricks don't apply to this input.
That's a property of the source, not a defense this code provides.)

Manual check:

    python -m src.arxiv_client --query "quantum error correction" --max-results 5
"""

from __future__ import annotations

import asyncio
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from typing import Sequence

import httpx

from . import config

ARXIV_API_URL = "https://export.arxiv.org/api/query"
USER_AGENT = "injection-detector/0.1 (https://github.com/zhouhanwu/injection-detector)"

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"

# arXiv's terms of use ask for roughly one request every three seconds. The
# orchestrator's loop can issue several searches per run, so the interval is
# enforced here rather than trusted to callers.
MIN_REQUEST_INTERVAL = 3.0

# Queries arrive as natural language ("efficient attention mechanisms for
# long-context transformers"), and ANDing every token including "for" matched
# nothing at all on arXiv — the default search path returned zero results.
# Dropping function words leaves the terms that carry the topic.
_STOPWORDS = frozenset(
    """a an and are as at be by for from how in into is it of on or that the
    their there these this to using via with within we what when which while""".split()
)

_rate_limit_lock = asyncio.Lock()
_last_request_at = 0.0


class ArxivError(RuntimeError):
    """The arXiv API could not be reached, or returned something unusable."""


@dataclass(frozen=True)
class Paper:
    """One candidate paper. This is the only text the scoring stages ever see."""

    id: str  # arXiv id including version, e.g. "2401.01234v2"
    title: str
    abstract: str
    url: str
    authors: tuple[str, ...] = ()
    primary_category: str | None = None
    published: str | None = None

    def with_abstract(self, abstract: str) -> "Paper":
        """Copy of this paper with a different abstract.

        Used by the A/B tester to build the flagged-spans-removed variant without
        mutating the original.
        """
        return replace(self, abstract=abstract)


@dataclass(frozen=True)
class SearchResult:
    papers: list[Paper]
    total_results: int
    search_query: str
    start: int


def _clean(text: str | None) -> str:
    """Collapse the line wrapping arXiv puts in titles and abstracts."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def build_search_query(
    topic: str,
    *,
    phrase: bool = False,
    combine: str = "AND",
    field: str = "all",
    categories: Sequence[str] | None = None,
) -> str:
    """Turn a plain-language topic into arXiv search syntax.

    The orchestrator uses the knobs to broaden or narrow between batches:
    `phrase=True` or a `categories` filter narrows; `combine="OR"` broadens.

    Note that arXiv's `cat:` matches any category listed on a paper, not only its
    primary one, so a cs.CL filter will still surface papers whose
    `primary_category` reads cs.CR. That is arXiv's behaviour, not a bug here.
    """
    topic = _clean(topic)
    if not topic:
        raise ValueError("topic is empty")

    if phrase:
        query = f'{field}:"{topic.replace(chr(34), "")}"'
    else:
        terms = [t for t in re.split(r"\s+", topic) if t]
        content_terms = [t for t in terms if t.lower().strip(".,;:") not in _STOPWORDS]
        # If a query is nothing but function words, searching for them beats
        # searching for nothing.
        terms = content_terms or terms
        joiner = f" {combine.upper()} "
        query = joiner.join(f"{field}:{t}" for t in terms)
        if len(terms) > 1:
            query = f"({query})"

    if categories:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        query = f"{query} AND ({cats})"
    return query


async def _throttle() -> None:
    global _last_request_at
    async with _rate_limit_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < MIN_REQUEST_INTERVAL and _last_request_at > 0:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - elapsed)
        _last_request_at = time.monotonic()


async def _get(client: httpx.AsyncClient, params: dict, attempts: int = 3) -> str:
    """GET the arXiv API with throttling and a couple of retries."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        await _throttle()
        try:
            response = await client.get(ARXIV_API_URL, params=params)
            response.raise_for_status()
            return response.text
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            last_error = exc
            if attempt < attempts - 1:
                await asyncio.sleep(2 ** attempt)
    raise ArxivError(f"arXiv API request failed after {attempts} attempts: {last_error}")


def _parse_entry(entry: ET.Element) -> Paper | None:
    raw_id = _clean(entry.findtext(f"{_ATOM}id"))
    # arXiv reports query problems as a single entry pointing at its error docs.
    if not raw_id or "/api/errors" in raw_id:
        return None

    arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
    title = _clean(entry.findtext(f"{_ATOM}title"))
    abstract = _clean(entry.findtext(f"{_ATOM}summary"))
    if not title or not abstract:
        return None

    authors = tuple(
        _clean(name.text)
        for author in entry.findall(f"{_ATOM}author")
        if (name := author.find(f"{_ATOM}name")) is not None and _clean(name.text)
    )
    primary = entry.find(f"{_ARXIV}primary_category")

    return Paper(
        id=arxiv_id,
        title=title,
        abstract=abstract,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        authors=authors,
        primary_category=primary.get("term") if primary is not None else None,
        published=_clean(entry.findtext(f"{_ATOM}published")) or None,
    )


def parse_feed(xml_text: str) -> tuple[list[Paper], int]:
    """Parse an arXiv Atom feed into papers plus the total hit count."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise ArxivError(f"could not parse the arXiv response as XML: {exc}") from exc

    total_text = root.findtext(f"{_OPENSEARCH}totalResults") or "0"
    try:
        total = int(total_text)
    except ValueError:
        total = 0

    papers = [p for entry in root.findall(f"{_ATOM}entry") if (p := _parse_entry(entry))]
    return papers, total


async def search(
    search_query: str,
    *,
    max_results: int = config.BATCH_SIZE,
    start: int = 0,
    sort_by: str = "relevance",
    sort_order: str = "descending",
    client: httpx.AsyncClient | None = None,
) -> SearchResult:
    """Run one arXiv search and return the parsed candidates.

    `search_query` is arXiv search syntax — build it with `build_search_query`.
    """
    params = {
        "search_query": search_query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=30.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    )
    try:
        xml_text = await _get(client, params)
    finally:
        if owns_client:
            await client.aclose()

    papers, total = parse_feed(xml_text)
    return SearchResult(
        papers=papers, total_results=total, search_query=search_query, start=start
    )


async def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Manual check of the arXiv client.")
    parser.add_argument("--query", required=True, help="plain-language topic")
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument("--phrase", action="store_true", help="exact-phrase search")
    parser.add_argument("--category", action="append", dest="categories")
    args = parser.parse_args()

    search_query = build_search_query(
        args.query, phrase=args.phrase, categories=args.categories
    )
    print(f"search_query: {search_query}\n")

    result = await search(search_query, max_results=args.max_results)
    print(f"{len(result.papers)} of {result.total_results} total hits\n")
    for i, paper in enumerate(result.papers, 1):
        authors = ", ".join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors += " et al."
        print(f"{i}. [{paper.id}] {paper.title}")
        print(f"   {authors}  ({paper.primary_category})")
        print(f"   {paper.url}")
        print(f"   {paper.abstract[:220]}...\n")


if __name__ == "__main__":
    asyncio.run(_main())
