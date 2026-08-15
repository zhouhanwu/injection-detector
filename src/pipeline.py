"""Per-paper pipeline: scorer -> sus catcher -> A/B tester -> report.

Isolation is per paper, not per batch: a whole batch runs concurrently with
`asyncio.gather`. Build step 7.
"""
