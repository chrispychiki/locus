/**
 * The unconditional privacy guarantee, pinned at the integration point where it lives: whatever
 * rrwebRules a deployment passes start(), the options handed to rrweb.record() always carry the
 * credential-mask posture — every input type routed (maskInputOptions), the composed maskInputFn
 * in place, a rule's own maskInputFn never handed to rrweb raw. No deployment input can override
 * it. (What the composed fn masks is the credential-masking suite's.)
 */
import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";

import { installBrowserGlobals, makeWindow } from "./_support.js";

let capturedRecordOptions = null;
let recordCalls = 0;
let titleReads = 0;
let recording = false;

mock.module("@rrweb/record", () => {
	const record = (options) => {
		capturedRecordOptions = options;
		recordCalls += 1;
		recording = true;
		return () => {
			recording = false;
		};
	};
	// takeFullSnapshot is rrweb's own periodic-checkout entry, which refuses to run before recording
	// opens. The recorder never calls it directly; it is modelled here so it never throws.
	record.takeFullSnapshot = () => {
		if (!recording)
			throw new Error("please take full snapshot after start recording");
	};
	return { record };
});

mock.module("@fingerprintjs/botd", () => ({
	load: async () => ({ detect: () => ({ bot: false }) }),
}));

const { start } = await import("../src/index.js");

let teardown = () => {};

beforeEach(() => {
	capturedRecordOptions = null;
	recordCalls = 0;
	titleReads = 0;
	recording = false;
	teardown = installBrowserGlobals({
		window: makeWindow({ href: "https://client.example.com/page" }),
		document: {
			get title() {
				titleReads += 1;
				return "t";
			},
		},
	});
});

afterEach(() => teardown());

const noopSink = { send: async () => {} };
const universal = (options) => [{ pattern: "*", options }];

describe("the credential-mask posture is unconditional at start()", () => {
	test("default deployment: password masking is on", async () => {
		const r = await start({ sink: noopSink, botDetection: false });
		expect(r).not.toBeNull();
		expect(capturedRecordOptions.maskInputOptions.password).toBe(true);
		r.stop();
	});

	test("every input type routes to the composed maskInputFn", async () => {
		const r = await start({ sink: noopSink, botDetection: false });
		for (const key of [
			"text",
			"email",
			"tel",
			"number",
			"textarea",
			"password",
		]) {
			expect(capturedRecordOptions.maskInputOptions[key]).toBe(true);
		}
		expect(typeof capturedRecordOptions.maskInputFn).toBe("function");
		r.stop();
	});

	test("a rule's maskInputFn is never handed to rrweb raw", async () => {
		const leak = (text) => text;
		const r = await start({
			sink: noopSink,
			botDetection: false,
			rrwebRules: universal({ maskInputFn: leak }),
		});
		expect(capturedRecordOptions.maskInputFn).not.toBe(leak);
		expect(typeof capturedRecordOptions.maskInputFn).toBe("function");
		r.stop();
	});

	test("a deployment passing maskInputOptions.password=false cannot turn it off", async () => {
		const r = await start({
			sink: noopSink,
			botDetection: false,
			rrwebRules: universal({
				maskInputOptions: { password: false, email: true },
			}),
		});
		expect(capturedRecordOptions.maskInputOptions.password).toBe(true);
		expect(capturedRecordOptions.maskInputOptions.email).toBe(true);
		r.stop();
	});

	test("a deployment passing an empty maskInputOptions cannot drop the password key", async () => {
		const r = await start({
			sink: noopSink,
			botDetection: false,
			rrwebRules: universal({ maskInputOptions: {} }),
		});
		expect(capturedRecordOptions.maskInputOptions.password).toBe(true);
		r.stop();
	});

	test("a deployment passing no maskInputOptions at all still gets password masking", async () => {
		const r = await start({
			sink: noopSink,
			botDetection: false,
			rrwebRules: universal({ maskAllInputs: false, recordCanvas: true }),
		});
		expect(capturedRecordOptions.maskInputOptions.password).toBe(true);
		expect(capturedRecordOptions.maskAllInputs).toBeUndefined();
		r.stop();
	});

	test("a rule cannot replace the recorder's options wholesale to evade it — recordDOM stays and password stays", async () => {
		const r = await start({
			sink: noopSink,
			botDetection: false,
			rrwebRules: universal({
				maskInputOptions: { password: false },
				recordDOM: false,
			}),
		});
		expect(capturedRecordOptions.maskInputOptions.password).toBe(true);
		expect(capturedRecordOptions.recordDOM).toBe(true);
		r.stop();
	});

	test("a rule cannot override the recorder's emit or recordDOM", async () => {
		const evil = () => {
			throw new Error("a rule's emit must never be used");
		};
		const r = await start({
			sink: noopSink,
			botDetection: false,
			rrwebRules: universal({ emit: evil, recordDOM: false }),
		});
		expect(capturedRecordOptions.emit).not.toBe(evil);
		expect(typeof capturedRecordOptions.emit).toBe("function");
		expect(capturedRecordOptions.recordDOM).toBe(true);
		r.stop();
	});

	// A single universal rule resolves the same options on every route, so nothing ever restarts
	// rrweb. (Routes that resolve differently are the route-masking suite's.)
	test("record() is invoked once and an SPA route change does not re-resolve masking", async () => {
		const r = await start({
			sink: noopSink,
			botDetection: false,
			rrwebRules: universal({
				maskInputOptions: { password: true, email: true },
			}),
		});
		expect(recordCalls).toBe(1);
		const optionsAtStart = capturedRecordOptions;

		window.history.pushState(null, "", "https://client.example.com/checkout");

		expect(titleReads).toBe(1); // the route change emitted its PageLoad marker
		expect(recordCalls).toBe(1); // ...but never a fresh record()
		expect(capturedRecordOptions).toBe(optionsAtStart); // the same fixed options object, untouched
		expect(capturedRecordOptions.maskInputOptions).toMatchObject({
			password: true,
			email: true,
		});
		r.stop();
	});

	// The test above drives pushState; an implementation that re-invoked record() on popstate alone
	// would slip past it.
	test("popstate and hashchange route changes do not re-invoke record()", async () => {
		const r = await start({
			sink: noopSink,
			botDetection: false,
			rrwebRules: universal({
				maskInputOptions: { password: true, email: true },
			}),
		});
		expect(recordCalls).toBe(1);
		const optionsAtStart = capturedRecordOptions;

		window.location.href = "https://client.example.com/billing";
		window.dispatch("popstate");

		window.location.href = "https://client.example.com/billing#section";
		window.dispatch("hashchange");

		expect(titleReads).toBe(2); // both dispatches reached the route-aware PageLoad path
		expect(recordCalls).toBe(1);
		expect(capturedRecordOptions).toBe(optionsAtStart);
		expect(capturedRecordOptions.maskInputOptions).toMatchObject({
			password: true,
			email: true,
		});
		r.stop();
	});

	// The guarantee is a property of the options object handed to rrweb, never something left
	// implicit in a broader flag — and the flag itself never reaches rrweb, whose record() would
	// resolve its gate from it alone and discard the options beside it.
	test("password stays explicitly true even alongside maskAllInputs", async () => {
		const r = await start({
			sink: noopSink,
			botDetection: false,
			rrwebRules: universal({
				maskAllInputs: true,
				maskInputOptions: { email: true },
			}),
		});
		expect(capturedRecordOptions.maskInputOptions.password).toBe(true);
		expect(capturedRecordOptions.maskInputOptions.select).toBe(true);
		expect(capturedRecordOptions.maskAllInputs).toBeUndefined();
		r.stop();
	});
});
