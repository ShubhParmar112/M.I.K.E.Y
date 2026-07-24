"""Privacy-tier classification (sovereignty S3 / architecture 02 §3).

Decides whether a turn is **Tier-0** — private data that must never leave the
device. The gateway already enforces T0 → local-only (S0) and the exporter already
excludes T0 from cloud training (S0); this is the missing classifier that makes
both live, so "your password never goes to the cloud" stops being a promise on
paper.

Heuristic and conservative for now (the seam a small local classifier replaces
later, like the router): T0 only on clear signals — credentials, financial or
government IDs, health records, or an explicit "keep this private". Everything
else is T1. A false T0 costs a little quality (served by the weaker local model);
a missed T0 would leak private data, so the patterns lean toward the obvious
sensitive cases rather than trying to be clever.
"""

from __future__ import annotations

import re

from core.events.schema import Tier

_T0 = re.compile(
    r"\b("
    # credentials / secrets ("pin" only when qualified — not "pin the tab")
    r"password|passwd|passphrase|otp|cvv|"
    r"pin (number|code|is)|(atm|card|debit|credit|bank|security|upi)[ -]?pin|"
    r"api[ -]?key|access[ -]?token|secret[ -]?key|private[ -]?key|credentials?|"
    # financial identifiers
    r"bank account|account number|routing number|credit card|debit card|card number|ifsc|"
    # government / personal IDs
    r"ssn|social security|aadhaar|aadhar|passport number|pan card|driver'?s licen[cs]e|"
    # health
    r"medical (record|diagnosis|report|history)|health record|prescription|blood test|"
    # explicit privacy intent
    r"keep (this|it|that) (private|secret|safe|on[ -]device|local|offline)|"
    r"(don'?t|do not) (send|share|upload|post)[^.?!]*(cloud|online|internet|server)|"
    r"confidential|off the record|between us"
    r")\b",
    re.I,
)


def classify_tier(text: str) -> Tier:
    """Tier.T0 if the turn plainly involves private data, else Tier.T1."""
    return Tier.T0 if _T0.search(text) else Tier.T1
