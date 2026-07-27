"""Reach: the parts of the machine M.I.K.E.Y is allowed to touch.

Everything up to now has run inside a sandbox — one workspace directory, an
allowlist of binaries, and a hard refusal to look anywhere else. That is the
right default and it is why the assistant has been safe to leave running. It is
also why it cannot do the things you actually want doing: your repositories are
not in the sandbox, and neither is your editor.

So this package opens specific doors, and the first module in it is the lock.
`projects.py` is the boundary: a directory is reachable only if **you** registered
it, from the CLI, by name. There is deliberately no tool for registering one — a
model that can widen its own reach has no reach limit at all, and every prompt
injection in every document it has ever read becomes a way to ask for more.

Within that boundary the usual asymmetry holds: reading (status, log, diff) is
allowed outright, anything that writes or leaves the machine (commit, push,
opening a URL) goes through an approval card with a simulation attached.
"""
