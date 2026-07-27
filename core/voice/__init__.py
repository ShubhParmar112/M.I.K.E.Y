"""Voice: hearing and speaking, on-device.

The pieces are deliberately separate — what to say (`text`), how to say it
(`synth`), how to play it (`play`), and how to hear (`listen`) — because only the
first is testable without a microphone in the room, and it is also the part that
decides whether being spoken to is pleasant or unbearable.
"""
