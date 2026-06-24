"""Breakout-following label namespace.

Composes the inherited mean-reversion triple-barrier label machinery with
inverted-direction interpretation (continuation = positive, reversion =
negative, timeout = timeout). No new label-construction logic; the wrapper
is a pure sign-flip on the inherited triple-barrier outcome.

This namespace is intentionally a scaffold. Concrete contracts
(``FollowLabel``, ``flip_label_for_follow``) are unavailable until their
implementation lands. Imports of those names raise ``NotImplementedError``
so contract tests track the planned surface explicitly.
"""
