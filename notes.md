# Notes — scope decisions, limits, and what I'd do with more time

Running log kept during the build. Written to be honest rather than flattering.

## Scope decisions

**Title + abstract only, from the arXiv API's plain-text metadata.**
This matches the brief's stated scope and avoids PDF parsing. A consequence worth
stating plainly: visual-layer attacks (white-on-white text, 1pt font, text hidden
behind an image) simply don't exist in this data source — the API returns plain
text. That is *not* a defense this system provides; it's a property of the input.
Full-PDF ingestion is deliberately out of scope, not overlooked.

**The orchestrator never reads raw paper text.**
It sees aggregate numeric scores only. This is a structural defense: the component
that decides "broaden / narrow / done" is unreachable from anything an attacker
can write into an abstract.

## Build environment

- Python 3.11 in a local `.venv` (the system Python here is 3.9).
- `ANTHROPIC_API_KEY` is loaded from a gitignored `.env` via `python-dotenv`.
- Models: `claude-sonnet-5` for the scorer / sus catcher / A-B rescore;
  `claude-haiku-4-5` for the orchestrator's broaden-narrow decision.
  `output_config.effort` is set explicitly on the Sonnet calls (the API default is
  `high`, which is more than these short classification calls need). Note that
  `effort` is *not* accepted on Haiku 4.5, so the orchestrator call omits it.

## Honest limits

_To be filled in as the build surfaces them — plus the eval numbers, reported once,
whatever they turn out to be._

## With more time

_To be filled in._
