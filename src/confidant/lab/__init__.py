"""Experiments with a model of Confidant's own: local models, adapted and measured.

Nothing in the core imports this package. What it needs beyond the core (``mlx-lm``,
which runs only on Apple silicon) is behind the ``lab`` extra and is imported only when a
model is actually loaded, so the package itself imports anywhere and is tested offline
against a fake generator.
"""

from confidant.lab.backend import LocalOutputError, MlxBackend, build_local_messages, parse_scan

__all__ = ["LocalOutputError", "MlxBackend", "build_local_messages", "parse_scan"]
