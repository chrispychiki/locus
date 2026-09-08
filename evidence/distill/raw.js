// The bun-side reader of the raw plane: raw_json rows hold zlib-compressed canonical JSON
// (written by hydrate.pack_raw, whose docstring owns the storage-form rationale). Every bun
// script that needs a raw event goes through here. node:zlib, not Bun.inflateSync: the stored
// bytes are zlib-format (RFC 1950) and node:zlib is exact about that, where Bun's own
// deflate/inflate pair speaks raw deflate.
import { inflateSync } from "node:zlib";

export const parseRaw = (blob) => JSON.parse(inflateSync(blob).toString());

// How the canonical stream is read back — the bun-side twin of db.py's CANONICAL_ORDER, which owns
// the order's rationale. Every bun-side read of the stream interpolates this, never its own ORDER BY.
export const CANONICAL_ORDER = "ORDER BY timestamp, counter, id";
