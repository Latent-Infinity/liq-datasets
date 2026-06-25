"""Breakout-following label namespace.

Composes the inherited mean-reversion triple-barrier label machinery with
inverted-direction interpretation (continuation = positive, reversion =
negative, timeout = timeout). No new label-construction logic; the wrapper
is a pure sign-flip on the inherited triple-barrier outcome.

The public label contracts live in
``liq.datasets.breakout_following.labels``.
"""
