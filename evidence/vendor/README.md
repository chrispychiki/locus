# Vendored replay assets

Vendoring rule: latest published version that (a) clears the 48h supply-chain cooldown and (b) demonstrably works in our harness. Record the npm publish date when updating. The replayer is the same rrweb release as the recorder's `@rrweb/record` and the `@rrweb/types` the constants are generated from, and `distill/mirror.js` transcribes its rebuild, so all of them are upgraded together.

- `rrweb-replay-2.1.7.min.js/.css` — @rrweb/replay 2.1.7 UMD (`dist/replay.umd.min.cjs`, `dist/style.min.css`; published 2026-10-02); local screenshot renderer harness (render.py). Verified rendering real slices.
- `rrweb-player-2.1.7.min.js/.css` — rrweb-player 2.1.7 UMD (`dist/rrweb-player.umd.min.cjs`, `dist/style.min.css`; published 2026-10-02); the embeddable replay component (replay.py). The stable releases before 2.1.1 (2.0.0, 2.0.1, 2.1.0) are broken — `new Player({target, props})` renders the frame div but never mounts internals (no iframe, no controller, no error; a Svelte production build resolving lifecycle APIs to stubs) — so on every update, verify the player actually mounts and plays a real slice, not just loads.
