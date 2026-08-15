"""Structured-output schemas for the scorer and sus catcher.

Every model call in this system returns schema-constrained JSON via
`output_config.format` — no free text ever becomes the final answer, and no
assistant prefill is used. Build step 3.
"""
