"""Shared test setup.

The suite is offline by design, and this file makes that a property of the suite rather
than a habit of whoever writes the next test: any attempt to open a network connection
fails loudly. Model-dependent code is tested against recordings instead (see
``confidant.recording``).
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from confidant.recording import Recording, ReplayClient

RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(self, address, *args, **kwargs):
        raise AssertionError(
            f"A test tried to connect to {address!r}. The suite is offline; use a recording."
        )

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


@pytest.fixture
def recording():
    """Load a recording by name, e.g. ``recording("flags_pressure")``."""

    def load(name: str) -> Recording:
        return Recording.load(RECORDINGS / f"{name}.json")

    return load


@pytest.fixture
def replay_client(recording):
    """A client that answers from the named recordings, in order."""

    def build(*names: str) -> ReplayClient:
        return ReplayClient(*(recording(name) for name in names))

    return build
