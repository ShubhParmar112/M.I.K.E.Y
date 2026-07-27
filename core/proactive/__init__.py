"""Proactivity: the part that speaks first.

An assistant you have to open is a tool. One that tells you the thing you needed
to know before you thought to ask is something else — and the difference is
almost entirely restraint. A system that interrupts wrongly gets muted, and a
muted assistant is worth less than a silent one, because now you also don't trust
it. So the interesting code here is not "notice things"; it is *when not to say
them*: `discipline.py`.

The shape follows the rest of the system. A nudge is an **event**; what is
outstanding is a **projection** over the log; delivering one appends another
event. So proactivity survives a restart, can be replayed, and every "why did you
tell me that?" has an answer in the trace.
"""
