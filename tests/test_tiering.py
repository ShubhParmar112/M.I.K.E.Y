"""Privacy-tier classification (sovereignty S3): plainly-private turns → Tier-0.

Precision matters both ways: a missed T0 leaks private data to the cloud; an
over-eager T0 needlessly drops a turn onto the weaker local model. These pin the
line the classifier draws.
"""

from __future__ import annotations

import pytest

from core.events.schema import Tier
from core.orchestrator.tiering import classify_tier


@pytest.mark.parametrize(
    "text",
    [
        "my banking password is hunter2",
        "what is my aadhaar number",
        "store my credit card number here",
        "my atm pin is 1234",
        "the wifi password is swordfish",
        "keep this private please",
        "don't upload this to the cloud",
        "here is my medical diagnosis report",
        "my api key is sk-live-123",
    ],
)
def test_private_turns_are_tier0(text: str) -> None:
    assert classify_tier(text) is Tier.T0


@pytest.mark.parametrize(
    "text",
    [
        "what files are in my workspace?",
        "remember my dog is named Pixel",
        "pin the tab to the top",          # 'pin' as a verb, not a secret
        "diagnose the network issue",       # tech 'diagnose', not medical
        "what is 17 times 3",
        "read notes.md and summarise it",
        "hey mikey, morning",
    ],
)
def test_benign_turns_are_tier1(text: str) -> None:
    assert classify_tier(text) is Tier.T1
