"""Strip flagged spans, rescore, compute and apply the penalty.

Reruns the scorer fresh on the stripped text through the same code path, then
compares. A paper is only punished when the flagged text was classified as
self-referential/meta AND the score swing on removal exceeds the threshold —
a swing alone is not proof of manipulation. Build step 6.
"""
