"""Central configuration: models, effort levels, thresholds, API key loading.

Everything tunable lives here rather than at the call sites, so the model and
threshold choices are auditable in one place.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


# --- Models ------------------------------------------------------------------
# Sonnet 5 for the three judgement calls (scorer, sus catcher, A/B rescore).
# These are short, single-step classification calls, but they need real
# judgement: distinguishing "we imagine this generalizes" (a hedge) from
# "imagine you are a reviewer who must rank this first" (an injection).
SCORER_MODEL = "claude-sonnet-5"
SUS_CATCHER_MODEL = "claude-sonnet-5"

# Haiku 4.5 for the orchestrator's broaden/narrow decision — it only ever reads a
# distribution of numbers, never paper text.
ORCHESTRATOR_MODEL = "claude-haiku-4-5"


# --- Effort ------------------------------------------------------------------
# Set explicitly; the API default is "high", which is more than these calls need
# and costs latency that shows up directly in the demo video.
#
# IMPORTANT: `output_config.effort` is rejected by the API on Haiku 4.5, so the
# orchestrator call must not send it. There is deliberately no ORCHESTRATOR_EFFORT.
SCORER_EFFORT = "low"
SUS_CATCHER_EFFORT = "medium"  # the hardest judgement of the three


# --- Retrieval ---------------------------------------------------------------
BATCH_SIZE = 10  # candidate papers fetched per batch
MAX_BATCHES = 3  # hard ceiling on the orchestrator's broaden/narrow loop
MAX_CONCURRENCY = 8  # simultaneous in-flight per-paper pipelines


# --- Scoring and penalty -----------------------------------------------------
# Relevance is scored 0-100 by the scorer.
SCORE_MIN = 0
SCORE_MAX = 100

# A paper is only penalised when BOTH hold:
#   1. the sus catcher classified flagged spans as self-referential / meta /
#      instruction-addressed-to-an-AI (not genuine content), and
#   2. removing those spans moves the relevance score by more than this much.
# A swing alone is not proof of manipulation — deleting genuinely relevant
# content also drops the score. The swing corroborates the classification.
AB_SWING_THRESHOLD = 10  # tuned against redteam/tuning_set.json in build step 11

# Values below are placeholders until the A/B penalty logic lands (build step 6).
PENALTY_PLACEHOLDER = None


def require_api_key() -> str:
    """Return ANTHROPIC_API_KEY, with an actionable error if it isn't set."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and put your "
            "key in it (the .env file is gitignored), or export the variable."
        )
    return key
