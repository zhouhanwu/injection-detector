"""Isolated per-paper injection / self-reference detection.

A separate LLM call from the scorer, also isolated to a single paper. Classifies
each sentence of the abstract as content-describing vs. self-referential / meta /
instruction-addressed-to-an-AI-reader, and reports flagged spans, reasoning, and a
sus-to-content ratio computed from that reasoned classification rather than from
keyword hits. Biased toward recall. Build step 5.
"""
