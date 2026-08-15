"""Query -> search params -> batches -> broaden/narrow loop -> ranked list.

Never touches raw paper text; it sees aggregate numeric scores only, which puts
the component with the power to steer the search structurally out of reach of
anything injected into an abstract. The final sort is deterministic code, not a
model call. Build step 8.
"""
