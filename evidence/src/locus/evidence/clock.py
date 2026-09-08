"""The convention for stating time on agent-facing surfaces.

An agent reading an output has no reliable sense of the current time, so a printed date can only be judged against an anchor the output itself supplies. Two rules follow, deployment-wide:

- Every output where timing is consequential states the current moment: a `current_timestamp` field on a JSON surface, a `current timestamp: <stamp>` line on a prose one, the `started <stamp>  <address>` opening line on a run (speak.py). Streams get the anchor; a persisted artifact instead stamps the event that made it, named for what it is (`started`, `measured`, `fetched`), because "current" inside a file read later reads as the reader's now.
- Every stated instant is absolute UTC in the one spelling below — never a bare date where the fact is an instant, never a zone-less rendering, never a relative phrase ("today", "3h ago"), which un-anchors itself the moment it is read. A fact refreshed on a schedule names its refresh convention beside its timestamp.

Epoch milliseconds need no rendering — they are already absolute and unambiguous; this spelling is for wherever a human-readable instant appears.
"""

from datetime import datetime, timezone


def utc_stamp(ms: int | None = None) -> str:
    """The instant as `YYYY-MM-DDTHH:MM:SSZ` — the one spelling every locus surface prints. No argument reads the clock now; an epoch-ms argument renders that instant."""
    moment = (
        datetime.now(timezone.utc)
        if ms is None
        else datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    )
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")
