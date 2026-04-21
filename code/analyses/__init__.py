"""Behavioral-measure computations used by paper figures.

Every submodule exposes at least an ``observed(dataset)`` function. Measures
that have a natural model-predicted analog also expose a ``predicted(model,
state, dataset, rng, n_samples)`` that samples synthetic recall sequences
and returns the same-shape array.
"""
