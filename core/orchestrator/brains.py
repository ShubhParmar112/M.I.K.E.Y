"""The brain registry + router (sovereignty S1: decompose the monolith).

Today the whole of cognition is one `gateway.complete(...)` call doing every job
at once (docs/04-intelligence-sovereignty §1). This module begins splitting that
into distinct **brains** — each a *capability profile* (data, not code: a system
prompt + a tool allowlist + a routing tier/capability), exactly as 02 §6 mandates
("roles, not agents; profiles are data"). A tiny **Router** picks which brain
handles each turn.

Still 100% cloud-backed: a brain is just a differently-scoped call through the
same Model Gateway. But the seam is now real — each brain has its own prompt,
its own tools, its own logged I/O — so later phases can (a) replace one brain at
a time with a local model and (b) train each from its own corpus.

Slice 1 ships two brains and a conservative heuristic router:

- **operator** — the full generalist (all tools). The safe default; unchanged
  behavior. Anything that might need a tool, memory, or an action lands here.
- **conversation** — persona-only, *no tools*. Greetings, sign-offs, thanks,
  small talk. Because it holds no tools it *cannot* touch memory or the executor
  — which is precisely what would have prevented the live memory_forget cascade
  on a goodbye.

The Router is a transparent heuristic here; it is the same seam a small trained
router model slots into later (§7.5) — `route()` is the only thing that changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.context.assembly import SYSTEM_PROMPT
from core.events.schema import Tier
from core.orchestrator.tools import TOOLS

CONVERSATION_PROMPT = """You are M.I.K.E.Y (Multimodal Intelligent Knowledge & Execution Engine \
for You), a personal AI companion talking with the person you work for. This is casual \
conversation — a greeting, a sign-off, a bit of small talk, or a quick opinion. You have no \
tools in this mode and you must NOT claim to remember, recall, run, read, change, or look up \
anything; just talk.

How you talk:
- Warm and natural, like a sharp human friend. A little dry wit is welcome; skip the empty \
praise and fawning.
- Use the person's name occasionally when you know it (from the memories below), not every line.
- Keep it short and real, and match their energy.
- Have a point of view; be honest over agreeable. Never invent facts to sound helpful.

If it turns out they actually want something done — a lookup, a file change, remembering or \
forgetting a fact — tell them to say so directly and you'll take care of it. Do NOT pretend \
you've already done it."""

MEMORY_PROMPT = """You are M.I.K.E.Y's memory manager — the ONLY part of the system trusted to \
change or delete long-term memory. The person has asked you to manage what M.I.K.E.Y remembers, \
and that is the whole of your job.

- To look something up: use memory_recall and answer with the source and date.
- To remember: use memory_remember to persist a durable fact the person asked you to keep. If \
this new fact updates an existing one, pass `supersedes` with the old memory's id (get it from \
memory_recall) — do NOT delete the old one separately.
- To forget: use memory_forget ONLY to delete the specific memory the person explicitly asked \
you to forget. First memory_recall to find its id and confirm it is the right one, then forget \
that ONE memory. Never forget anything the person did not name. Never delete memories to tidy up, \
deduplicate, or clean house on your own — that is never your call.

Do one memory operation at a time, deliberately. If you are unsure which memory the person means, \
ask them rather than guessing. You hold no other tools and take no other actions."""

ALL_TOOL_NAMES = frozenset(t["name"] for t in TOOLS)
# The operator keeps every tool EXCEPT memory_forget: destroying a memory is
# authority reserved for the memory brain, so the generalist can never fire it —
# the durable fix for the live memory_forget cascade.
OPERATOR_TOOLS = ALL_TOOL_NAMES - {"memory_forget"}
# The memory brain's narrow charter: read, write, and delete memory, nothing else.
MEMORY_BRAIN_TOOLS = frozenset({"memory_recall", "memory_remember", "memory_forget"})


@dataclass(frozen=True)
class Brain:
    """A capability profile: how a role speaks (prompt), what it can do (tools),
    and how it should be routed (capability + privacy tier)."""

    name: str
    system_prompt: str
    capability: str
    tier: Tier = Tier.T1
    tool_names: frozenset[str] = frozenset()

    @property
    def tools(self) -> list[dict[str, object]]:
        # Preserve TOOLS order; empty allowlist == a brain with no tools at all.
        return [t for t in TOOLS if t["name"] in self.tool_names]


OPERATOR = Brain(
    name="operator",
    system_prompt=SYSTEM_PROMPT,
    capability="general",
    tier=Tier.T1,
    tool_names=OPERATOR_TOOLS,  # everything except memory_forget
)

CONVERSATION = Brain(
    name="conversation",
    system_prompt=CONVERSATION_PROMPT,
    capability="chat",
    tier=Tier.T1,
    tool_names=frozenset(),  # no tools: cannot touch memory or the executor
)

MEMORY = Brain(
    name="memory",
    system_prompt=MEMORY_PROMPT,
    capability="memory",
    tier=Tier.T1,
    tool_names=MEMORY_BRAIN_TOOLS,  # the only brain that may forget
)

BRAINS: dict[str, Brain] = {b.name: b for b in (OPERATOR, CONVERSATION, MEMORY)}


@dataclass(frozen=True)
class Routing:
    brain: Brain
    reason: str


# Any hint of a tool/memory/action → the full operator. Kept broad on purpose:
# a false match here just uses the more-capable brain (harmless); a *miss* would
# strand a real request in the toolless brain (harmful). Safety leans to operator.
_ACTIONY = re.compile(
    r"\b("
    r"remember|forget|recall|memoi?ri[sz]e|note that|jot|save|store|"
    r"read|write|open|list|show|display|run|execute|command|"
    r"fetch|download|http|url|ingest|load|"
    r"search|look up|lookup|find|file|folder|directory|path|"
    r"code|git|repo|create|make|delete|remove|update|edit|change|fix|"
    r"summari[sz]e|check|calculate|compute"
    r")\b",
    re.I,
)

# Intent to delete/curate long-term memory → the memory brain (the only one that
# may forget). "forget"/"unremember" are unambiguous here; delete/erase/remove/etc.
# count only when clearly aimed at memory (so "delete the temp file" stays operator).
_FORGET = re.compile(
    r"\b(forget|unremember)\b"
    r"|\b(delete|erase|remove|wipe|scrub|purge|clear)\b[^.?!]*"
    r"\b(memor|note|fact|remember|told you|you said|you know|about me)\b",
    re.I,
)

# A clearly social cue (greeting, sign-off, thanks, acknowledgement). Not anchored:
# real goodbyes are messy ("yeah so that was it, will ttyl mikey").
_SOCIAL = re.compile(
    r"\b("
    r"hi|hey|hiya|hello|yo|sup|"
    r"good ?(morning|afternoon|evening)|"
    r"thanks|thank you|thankyou|thx|cheers|appreciate|"
    r"bye|goodbye|good ?night|see ya|see you|cya|ttyl|talk (to you )?later|"
    r"that('| wa)s it|we'?re done|that'?s all|catch you later|"
    r"how are you|how's it going|how are things|what'?s up|howdy|"
    r"nice|cool|awesome|great|ok|okay|got it|no worries|np|welcome|lol|haha"
    r")\b",
    re.I,
)


class Router:
    """Chooses which brain handles a turn. Heuristic today; the seam a trained
    router model replaces later, behind this same `route()` interface."""

    def route(self, user_input: str) -> Routing:
        text = user_input.strip()
        # Forgetting/curating memory is checked first (it's also "actiony"): only
        # the memory brain holds memory_forget, so this is where those turns must go.
        if _FORGET.search(text):
            return Routing(MEMORY, "memory curation — removing/correcting what's stored")
        if _ACTIONY.search(text):
            return Routing(OPERATOR, "request implies a tool, memory op, or action")
        # A question is likely an information need (often memory/facts) — default to
        # the capable brain rather than risk a toolless miss.
        if "?" in text:
            return Routing(OPERATOR, "question — may need memory or a lookup")
        if _SOCIAL.search(text):
            return Routing(CONVERSATION, "social/small-talk with no actionable request")
        return Routing(OPERATOR, "default: full-capability operator")
