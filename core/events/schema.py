"""Event envelope — the source-of-truth record shape (architecture 02 §4).

Every fact in the system is an event; everything else is a projection.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

_ENCODING = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford base32


def ulid() -> str:
    """Lexicographically sortable, time-ordered id (ULID)."""
    ts = int(time.time() * 1000)
    out = ["0"] * 26
    for i in range(9, -1, -1):
        out[i] = _ENCODING[ts & 0x1F]
        ts >>= 5
    rand = int.from_bytes(os.urandom(10), "big")
    for i in range(25, 9, -1):
        out[i] = _ENCODING[rand & 0x1F]
        rand >>= 5
    return "".join(out)


def now() -> datetime:
    return datetime.now(UTC)


class Tier(StrEnum):
    T0 = "T0"  # never leaves device
    T1 = "T1"  # transient cloud inference allowed
    T2 = "T2"  # public


class EventType(StrEnum):
    USER_MESSAGE = "conversation.message.user"
    ASSISTANT_MESSAGE = "conversation.message.assistant"
    ACTION_EXECUTED = "action.executed"
    INGEST_DOCUMENT = "ingest.document"


class Provenance(BaseModel):
    source: str = "user"  # "user" | "agent" | "connector:<name>" | "web:<url>"
    trusted: bool = True


class Event(BaseModel):
    id: str = Field(default_factory=ulid)
    v: int = 1
    type: str
    ts: datetime = Field(default_factory=now)
    device: str = "dev_desktop_1"
    tier: Tier = Tier.T1
    provenance: Provenance = Field(default_factory=Provenance)
    payload: dict[str, Any] = Field(default_factory=dict)
