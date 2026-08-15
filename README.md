# injection-detector

An injection-resistant arXiv paper ranker: retrieve candidate papers for a research
query, rank them by relevance, and defend against papers whose titles/abstracts
contain prompt-injection attempts (fake instructions, roleplay/hypothetical framing,
self-referential "rank me higher" claims) — without ever treating that text as
instructions.

> **Status: work in progress.** The repo is public from the first commit and is
> being built in the order laid out in `PLAN.md`. This README is a placeholder;
> the full architecture write-up, defense rationale, eval numbers, and an honest
> "what this doesn't catch" section land at the end of the build.

## Scope

Title + abstract only, via the arXiv API's plain-text metadata. No PDF parsing —
which also means visual-layer attacks (white-on-white hidden text in a PDF) don't
apply to this data source. That's a property of the scope, not a defense being
claimed.

## Setup

```sh
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env
```

## Run

```sh
.venv/bin/python -m src.main --query "<research topic>"
.venv/bin/python -m tests.test_eval
```

## Layout

| Path | Role |
| --- | --- |
| `src/arxiv_client.py` | arXiv search → candidate papers (title, abstract, id, url) |
| `src/schemas.py` | Structured-output schemas for the scorer and sus catcher |
| `src/scorer.py` | Isolated per-paper relevance scoring |
| `src/sus_catcher.py` | Isolated per-paper injection / self-reference detection |
| `src/ab_tester.py` | Strip flagged spans, rescore, apply the penalty |
| `src/pipeline.py` | Per-paper: scorer → sus catcher → A/B → report |
| `src/orchestrator.py` | Query → batches → broaden/narrow loop → ranked list |
| `src/main.py` | CLI entrypoint |
| `redteam/` | Offline red-team attack generator + tuning/eval splits |
| `tests/test_eval.py` | Grades the held-out eval set, reports accuracy + FP rate |
| `notes.md` | Scope decisions, honest limits, what I'd do with more time |
