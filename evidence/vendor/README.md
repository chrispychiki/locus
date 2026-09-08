# Vendored replay assets

Vendoring rule: latest published version that (a) clears the 48h supply-chain cooldown and (b) demonstrably works in our harness. Record the npm publish date when updating.

- `rrweb-replay-2.1.1.min.js/.css` — @rrweb/replay 2.1.1 UMD (published 2026-07-21, byte-identical dist to 2.1.0 — the release bumps dependencies only); local screenshot renderer harness (render.py). Verified rendering real slices.
- `rrweb-player-2.1.1.min.js/.css` — rrweb-player 2.1.1 UMD (published 2026-07-21); the embeddable replay component (replay.py). The stable releases before it (2.0.0, 2.0.1, 2.1.0) are broken — `new Player({target, props})` renders the frame div but never mounts internals (no iframe, no controller, no error; a Svelte production build resolving lifecycle APIs to stubs, fixed upstream in 2.1.1 PR #1901) — so on every update, verify the player actually mounts and plays a real slice, not just loads.
