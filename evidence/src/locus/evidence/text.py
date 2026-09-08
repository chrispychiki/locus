"""The URL rendering rule the event stream and context blocks share with the projection.

A URL on an agent-facing text surface is affordance metadata, not content: it says where the visitor is or where a link points, never its tracking params. The rule is distill/text.js's describeUrl — one rule, stated on both sides of the language boundary because the projection renders in JS and the event stream in Python — and the two are pinned to each other by a contract test (test_text.py) that runs the JS implementation over the same cases, so they cannot drift apart silently. The ceiling's rationale and the cut's shape are stated there, once; the data:-URI summarization stays JS-only: a stream URL is a page address and is never a data: URI.
"""

URL_CAP = 2000


def _magnitude(n: int) -> str:
    return f"{n} chars" if n < 1024 else f"{round(n / 1024)}kB"


def _units(s: str) -> int:
    """UTF-16 code units — the unit describeUrl measures in. JS string lengths are code units, so the Python twin counts the same way or the two caps diverge on any astral character (an emoji in a path is two units there, one code point here)."""
    return sum(2 if ord(c) > 0xFFFF else 1 for c in s)


def _cut(s: str, end_units: int) -> str:
    """The longest prefix of at most end_units code units that ends on a whole character — never mid-astral, and never on a lone high surrogate (the JS side states the rule once, in text.js's safeCut)."""
    units = 0
    idx = 0
    for i, c in enumerate(s):
        w = 2 if ord(c) > 0xFFFF else 1
        if units + w > end_units:
            break
        units += w
        idx = i + 1
    if idx and 0xD800 <= ord(s[idx - 1]) <= 0xDBFF:
        idx -= 1
    return s[:idx]


def describe_url(url: str | None) -> str | None:
    if url is None or _units(url) <= URL_CAP:
        return url
    hash_at = url.find("#")
    frag = url[hash_at:] if hash_at >= 0 else ""
    base = url[:hash_at] if hash_at >= 0 else url
    q = base.find("?")
    if q >= 0:
        kept = f"{base[:q]}?… ({_magnitude(_units(base[q + 1 :]))} query){frag}"
        if _units(kept) <= URL_CAP:
            return kept
    return _cut(url, URL_CAP - 1) + "…"


def escape_value(s: str) -> str:
    """A field's value in the written form the projection gives it (text.js's escapeValue): backslash, newline, carriage return, and tab escaped, so one value has one written form. The event stream keys a repeated value on this form."""
    return (
        s.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
