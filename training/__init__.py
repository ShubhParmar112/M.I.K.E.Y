"""Offline training tooling — deliberately OUTSIDE `core/`.

`core/` is the runtime cognitive OS; `training/` is the data flywheel that turns
its event log into datasets for the eventual local brain fleet (see
docs/04-intelligence-sovereignty.md, sovereignty phase S0). Nothing in `core/`
imports from here — this only ever reads the log after the fact.
"""
