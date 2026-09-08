/**
 * The cross-package contract suites — the built bundle driven in a real Chromium, and the
 * recorder↔evidence integration contract — run from inside `bun test`, so the recorder's bar is one
 * command rather than two, and a natural invocation cannot silently under-run it.
 *
 * The price is that these two suites are not self-contained. They shell out to the Python packages, so on
 * top of the recorder's own `bun install` they need `uv`, a synced `analysis/` project (whose env
 * carries the evidence package too — the integration test imports both), and Playwright's Chromium —
 * see analysis/README.md § Install. Without those, this one file fails while every other suite in the package
 * passes, and its failure is a missing Python project, not a broken recorder.
 */
import { describe, expect, test } from "bun:test";

describe("cross-package contract suites", () => {
	test("snippet autostart + recorder↔evidence integration (pytest)", () => {
		const result = Bun.spawnSync(
			[
				"uv",
				"run",
				"--project",
				"../analysis",
				"pytest",
				"-q",
				"tests/test_snippet_autostart.py",
				"tests/test_recorder_integration.py",
			],
			{ cwd: `${import.meta.dir}/..` },
		);
		if (result.exitCode !== 0) {
			console.error(result.stdout.toString());
			console.error(result.stderr.toString());
		}
		expect(result.exitCode).toBe(0);
	}, 300_000);
});
