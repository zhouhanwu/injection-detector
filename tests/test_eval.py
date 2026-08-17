"""Grade a red-team set through the pipeline and report the numbers honestly.

    python -m tests.test_eval                     # held-out set, graded once
    python -m tests.test_eval --set tuning        # tuning set, iterate freely
    python -m tests.test_eval --repeats 1         # quick, noisier

**Every case is run more than once by default.** Single measurements of this
system visibly move: the same crude attack has measured a potency of -10, +0 and
-5 across runs, and a novelty-claim sentence was flagged in one run and not the
next. A single-pass score would carry a noise band wider than most of the effects
being measured. Flags and penalties are decided by majority across repeats and
scores by median, and the per-case disagreement rate is reported so the reader
can see how stable the underlying judgements were.

Four numbers matter, and they trade off against each other:

* **detection recall** — attacks flagged at all. Missing an attack is the failure
  the system exists to prevent.
* **penalty rate** — attacks that actually lost points. A flag that changes no
  outcome is worth less than one that does.
* **false positive rate** — clean papers flagged. Real arXiv abstracts, so this
  measures what would happen to ordinary papers.
* **false penalty rate** — clean papers that lost points. This one should be
  zero; a flag on a clean paper is a nuisance, a penalty on one is a wrong answer.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from collections import Counter
from pathlib import Path

from src import config
from src.ab_tester import run_ab_test
from src.arxiv_client import Paper
from src.sus_catcher import catch_sus

SETS = {
    "eval": Path(__file__).parent.parent / "redteam" / "eval_set.json",
    "tuning": Path(__file__).parent.parent / "redteam" / "tuning_set.json",
}

# The eval is graded against a fixed query so that relevance is held constant and
# only the detection behaviour varies between cases.
QUERY = "efficient attention mechanisms for long-context transformers"


async def grade_case(case: dict, *, client, repeats: int) -> dict:
    """Run one case `repeats` times and reduce to a single verdict."""
    paper = Paper(
        id=case["id"],
        url="(red-team case)",
        title=case["title"],
        abstract=case["abstract"],
    )

    flags: list[bool] = []
    penalties: list[bool] = []
    labels: Counter = Counter()
    potencies: list[int] = []
    adjusted: list[int] = []
    errors: list[str] = []

    for _ in range(repeats):
        try:
            sus = await catch_sus(paper, client=client)
            result = await run_ab_test(QUERY, sus, client=client)
        except Exception as exc:  # one bad run should not lose the case
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        flags.append(sus.has_flags)
        penalties.append(result.penalised)
        labels.update(sus.classification.flagged_labels)
        if result.attack_potency is not None:
            potencies.append(result.attack_potency)
        adjusted.append(result.adjusted_score)

    def majority(values: list[bool]) -> bool:
        return sum(values) * 2 > len(values) if values else False

    return {
        "id": case["id"],
        "kind": case["kind"],
        "attack_type": case.get("attack_type", "clean"),
        "expect_flagged": case["expect_flagged"],
        "flagged": majority(flags),
        "penalised": majority(penalties),
        "labels": sorted(labels),
        "unstable": len(set(flags)) > 1,  # disagreed with itself across repeats
        "potency": round(statistics.median(potencies)) if potencies else None,
        "adjusted": round(statistics.median(adjusted)) if adjusted else None,
        "runs": len(flags),
        "errors": errors,
    }


def report(outcomes: list[dict], *, name: str, repeats: int) -> None:
    attacks = [o for o in outcomes if o["kind"] == "attack"]
    clean = [o for o in outcomes if o["kind"] == "clean"]

    def pct(n: int, d: int) -> str:
        return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0 (n/a)"

    print(f"\n{'=' * 66}")
    print(f"{name.upper()} SET — {len(outcomes)} cases, {repeats} run(s) each")
    print("=" * 66)

    print(f"\nDETECTION ({len(attacks)} attacks)")
    print(f"  flagged            {pct(sum(o['flagged'] for o in attacks), len(attacks))}")
    print(f"  penalised          {pct(sum(o['penalised'] for o in attacks), len(attacks))}")

    by_type: dict[str, list[dict]] = {}
    for outcome in attacks:
        by_type.setdefault(outcome["attack_type"], []).append(outcome)
    for attack_type in sorted(by_type):
        group = by_type[attack_type]
        print(
            f"    {attack_type:<20} flagged {pct(sum(o['flagged'] for o in group), len(group)):<12}"
            f" penalised {pct(sum(o['penalised'] for o in group), len(group))}"
        )

    print(f"\nFALSE POSITIVES ({len(clean)} clean papers)")
    print(f"  flagged            {pct(sum(o['flagged'] for o in clean), len(clean))}")
    print(f"  penalised          {pct(sum(o['penalised'] for o in clean), len(clean))}")

    missed = [o for o in attacks if not o["flagged"]]
    escaped = [o for o in attacks if o["flagged"] and not o["penalised"]]
    false_pos = [o for o in clean if o["flagged"]]
    unstable = [o for o in outcomes if o["unstable"]]

    if missed:
        print(f"\nMISSED ENTIRELY ({len(missed)}) — not flagged at all:")
        for outcome in missed:
            print(f"  {outcome['id']} [{outcome['attack_type']}]")
    if escaped:
        print(f"\nFLAGGED BUT NOT PENALISED ({len(escaped)}):")
        for outcome in escaped:
            print(
                f"  {outcome['id']} [{outcome['attack_type']}] "
                f"labels={outcome['labels']} potency={outcome['potency']}"
            )
    if false_pos:
        print(f"\nFALSE POSITIVES ({len(false_pos)}):")
        for outcome in false_pos:
            print(f"  {outcome['id']} labels={outcome['labels']}")
    if unstable:
        print(f"\nUNSTABLE ACROSS REPEATS ({len(unstable)}) — flagged in some runs, not others:")
        for outcome in unstable:
            print(f"  {outcome['id']} [{outcome['attack_type']}]")

    errored = [o for o in outcomes if o["errors"]]
    if errored:
        print(f"\nERRORS ({len(errored)}):")
        for outcome in errored:
            print(f"  {outcome['id']}: {outcome['errors'][0][:110]}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Grade a red-team set.")
    parser.add_argument("--set", choices=sorted(SETS), default="eval")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENCY)
    args = parser.parse_args()

    path = SETS[args.set]
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Run: python -m redteam.generate_attacks"
        )
    cases = json.loads(path.read_text())["cases"]

    if args.set == "eval":
        print(
            "Grading the HELD-OUT set. It is meant to be run once, and the number "
            "reported honestly whatever it is.\n"
        )

    limit = asyncio.Semaphore(args.concurrency)
    client = config.make_client()

    async def guarded(case: dict) -> dict:
        async with limit:
            return await grade_case(case, client=client, repeats=args.repeats)

    try:
        outcomes = await asyncio.gather(*(guarded(case) for case in cases))
    finally:
        await client.close()

    report(outcomes, name=args.set, repeats=args.repeats)

    totals, cost = config.usage_report()
    calls = sum(entry["calls"] for entry in totals.values())
    print(f"\n{calls} model calls, approx ${cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
