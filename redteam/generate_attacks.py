"""Offline red-team attack generator.

Produces a pool of injection attempts across the spectrum: crude "ignore your
instructions", roleplay/hypothetical framing, self-referential ranking claims,
and deliberate hedge-language traps ("we imagine this generalizes to...") that a
keyword matcher would fire on but a reasoning classifier should not.

Split into a tuning set (iterate the sus catcher prompt against it) and a
held-out eval set (graded exactly once). A few hand-written attacks are mixed in
so the eval isn't only testing whether the sus catcher recognises its own
red-teamer's writing style. Build step 10.
"""
