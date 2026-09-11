/**
 * start()'s orchestration, the parts not pinned by the masking / catastrophic / route-awareness
 * suites: the host-page safety gates (local environments, bots, prerender, the error threshold) and
 * capture-correctness (PageLoad's placement and cardinality, the visibility markers, the head slice).
 *
 * Observed through the real buffer and the real chunk path — a capturing sink collects the
 * gzipped chunks and the test gunzips them. A buffer double would leak into every sibling suite,
 * since bun's mock.module is global, so the rig installs the IndexedDB fake instead.
 */
import { afterEach, beforeEach, describe, expect, mock, test } from "bun:test";
import { gunzipSync } from "fflate";

import {
	fullSnapshot,
	installBrowserGlobals,
	meta,
	settle,
} from "./_support.js";

let capturedEmit = null;
let capturedErrorHandler = null;
let capturedRecordOptions = null;
let stopRecordingCalls = 0;
let botIsBot = false;
let botDetections = null;

mock.module("@rrweb/record", () => {
	const record = (options) => {
		capturedRecordOptions = options;
		capturedEmit = options.emit;
		capturedErrorHandler = options.errorHandler;
		return () => {
			stopRecordingCalls += 1;
		};
	};
	record.takeFullSnapshot = () => {};
	return { record };
});
mock.module("@fingerprintjs/botd", () => ({
	load: async () => ({
		detect: () => ({ bot: botIsBot }),
		getDetections: () =>
			botDetections ?? {
				detectWindowSize: { bot: botIsBot },
				detectUserAgent: { bot: false },
			},
	}),
}));

const { start } = await import("../src/index.js");
const { storeRejection } = await import("../src/sink.js");
const { EventType, IncrementalSource } = await import(
	"../src/rrweb_constants.js"
);

const decode = (bytes) =>
	JSON.parse(new TextDecoder().decode(gunzipSync(bytes)));

let docListeners = {};
let teardown = () => {};

beforeEach(() => {
	capturedEmit = null;
	capturedErrorHandler = null;
	capturedRecordOptions = null;
	stopRecordingCalls = 0;
	botIsBot = false;
	botDetections = null;
	docListeners = {};
	teardown = installBrowserGlobals({
		document: {
			title: "Home",
			referrer: "https://ref.example/from",
			addEventListener: (type, fn) => {
				docListeners[type] = fn;
			},
			// Tracked, not a no-op: terminate() unregisters on document as well as window, and a
			// double that forgets removals cannot tell a torn-down recorder from a live one.
			removeEventListener: (type, fn) => {
				if (docListeners[type] === fn) delete docListeners[type];
			},
		},
	});
});
afterEach(() => teardown());

const straddle = (ts, timeOffset) => ({
	type: EventType.IncrementalSnapshot,
	data: { source: IncrementalSource.MouseMove, positions: [{ timeOffset }] },
	timestamp: ts,
});

describe("start() host-page safety gates", () => {
	test("a local environment is never recorded", async () => {
		globalThis.window.location.hostname = "localhost";
		const r = await start({
			sink: { send: async () => {} },
			botDetection: false,
		});
		expect(r).toBeNull();
		expect(capturedEmit).toBeNull();
	});

	test("a detected bot is never recorded", async () => {
		botIsBot = true;
		const r = await start({ sink: { send: async () => {} } });
		expect(r).toBeNull();
		expect(capturedEmit).toBeNull();
	});

	test("bot detection runs by default when the flag is omitted", async () => {
		botIsBot = true;
		const r = await start({ sink: { send: async () => {} } }); // no botDetection key at all
		expect(r).toBeNull();
		expect(capturedEmit).toBeNull();
	});

	test("a bot-gated page context attests itself — one capture_gated ping, then nothing", async () => {
		// The recorded corpus is definitionally only what passed the gates, so without this ping
		// "how much of my traffic is bots" has no witness anywhere.
		botIsBot = true;
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const r = await start({ sink: { send: async () => {} }, telemetry });
		expect(r).toBeNull();
		expect(capturedEmit).toBeNull();
		expect(telemetry.pings).toHaveLength(1);
		const [gated] = telemetry.pings;
		expect(gated.metric).toBe("capture_gated");
		expect(gated.reason).toBe("bot");
		expect(gated.detail).toBe("detectWindowSize");
		expect(gated.url).toBe(window.location.href);
		expect(gated.visitorId.length).toBeGreaterThan(0);
	});

	const telemetryLog = () => ({
		pings: [],
		emit(e) {
			this.pings.push(e);
		},
	});

	test.each([
		[
			"detectMimeTypesConsistent",
			"Facebook's Android in-app browser patches navigator",
		],
		[
			"detectPluginsLengthInconsistency",
			"Chromium empties navigator.plugins without its PDF viewer, and Android Chrome always does",
		],
	])("a verdict on %s alone records — %s", async (detector) => {
		botIsBot = true;
		botDetections = { [detector]: { bot: true } };
		const telemetry = telemetryLog();
		const r = await start({ sink: { send: async () => {} }, telemetry });
		expect(r).not.toBeNull();
		expect(
			telemetry.pings.filter((p) => p.metric === "capture_gated"),
		).toHaveLength(0);
		await r.stop();
	});

	test("an exempted verdict rides every birth as gate_exempt — the admitted cohort's only witness", async () => {
		botIsBot = true;
		botDetections = { detectPluginsLengthInconsistency: { bot: true } };
		const telemetry = telemetryLog();
		const r = await start({ sink: { send: async () => {} }, telemetry });
		await settle();
		const births = telemetry.pings.filter((p) => p.metric === "slice_started");
		expect(births.length).toBeGreaterThan(0);
		for (const b of births) {
			expect(b.gate_exempt).toBe("detectPluginsLengthInconsistency");
		}
		await r.stop();
	});

	test("a clean verdict and an ungated context both birth with gate_exempt empty", async () => {
		for (const options of [{}, { botDetection: false }]) {
			const telemetry = telemetryLog();
			const r = await start({
				sink: { send: async () => {} },
				telemetry,
				...options,
			});
			await settle();
			expect(
				telemetry.pings.find((p) => p.metric === "slice_started").gate_exempt,
			).toBe("");
			await r.stop();
		}
	});

	test("a verdict the deployment took ahead of start() is the gate's own — no second BotD run, and its exempted detector rides the births", async () => {
		botIsBot = true;
		botDetections = { detectWebDriver: { bot: true } };
		const telemetry = telemetryLog();
		const r = await start({
			sink: { send: async () => {} },
			telemetry,
			botVerdict: { isBot: false, detail: "detectMimeTypesConsistent" },
		});
		await settle();
		expect(r).not.toBeNull();
		expect(
			telemetry.pings.filter((p) => p.metric === "capture_gated"),
		).toHaveLength(0);
		expect(
			telemetry.pings.find((p) => p.metric === "slice_started").gate_exempt,
		).toBe("detectMimeTypesConsistent");
		await r.stop();
	});

	test("an exempt detector beside any other still gates — the exemption is the lone-detector verdict only", async () => {
		botIsBot = true;
		botDetections = {
			detectMimeTypesConsistent: { bot: true },
			detectWebDriver: { bot: true },
		};
		const telemetry = telemetryLog();
		const r = await start({ sink: { send: async () => {} }, telemetry });
		expect(r).toBeNull();
		expect(telemetry.pings).toHaveLength(1);
		expect(telemetry.pings[0].detail).toBe(
			"detectMimeTypesConsistent,detectWebDriver",
		);
	});

	test("two exempt detectors together still gate", async () => {
		botIsBot = true;
		botDetections = {
			detectMimeTypesConsistent: { bot: true },
			detectPluginsLengthInconsistency: { bot: true },
		};
		const telemetry = telemetryLog();
		const r = await start({ sink: { send: async () => {} }, telemetry });
		expect(r).toBeNull();
		expect(telemetry.pings[0].detail).toBe(
			"detectMimeTypesConsistent,detectPluginsLengthInconsistency",
		);
	});

	test("the local-environment gate stays silent — the operator's own dev machine is not their traffic", async () => {
		globalThis.window.location.hostname = "localhost";
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const r = await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
		});
		expect(r).toBeNull();
		expect(telemetry.pings).toEqual([]);
	});

	// One host per family the gate covers: loopback v4/v6, the .local mDNS suffix, and each RFC1918
	// private range.
	test.each([
		"127.0.0.1",
		"::1",
		"dev.local",
		"10.0.0.5",
		"192.168.1.10",
		"172.16.4.4",
		"172.31.255.1",
	])("a local/private host (%s) is never recorded", async (hostname) => {
		globalThis.window.location.hostname = hostname;
		const r = await start({
			sink: { send: async () => {} },
			botDetection: false,
		});
		expect(r).toBeNull();
		expect(capturedEmit).toBeNull();
	});

	// Public hosts that merely contain a local-looking string.
	test.each([
		"10.example.com",
		"localhost.tools.acme.com",
		"my127.0.0.1.example.com",
		"192.168.shop",
		"notlocalhost.com",
		"example.local.com",
	])(
		"a public host that merely looks local (%s) is recorded",
		async (hostname) => {
			globalThis.window.location.hostname = hostname;
			const r = await start({
				sink: { send: async () => {} },
				botDetection: false,
			});
			expect(r).not.toBeNull();
			expect(capturedEmit).not.toBeNull();
			await r.stop();
		},
	);

	test("recordLocalEnvironment=true overrides the local-host refusal", async () => {
		globalThis.window.location.hostname = "localhost";
		const r = await start({
			sink: { send: async () => {} },
			botDetection: false,
			recordLocalEnvironment: true,
		});
		expect(r).not.toBeNull();
		expect(capturedEmit).not.toBeNull();
	});

	test("a prerendered page records nothing until activation, then starts", async () => {
		globalThis.document.prerendering = true;
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const promise = start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
		});
		await settle();
		expect(capturedEmit).toBeNull();
		expect(telemetry.pings).toEqual([]);
		expect(globalThis.document.cookie).toBe(""); // no visitor state written pre-activation

		globalThis.document.prerendering = false; // spec order: flag flips before the event fires
		docListeners.prerenderingchange();
		const r = await promise;
		expect(r).not.toBeNull();
		expect(capturedEmit).not.toBeNull();
	});

	test("a prerender that never activates leaves no trace at all", async () => {
		globalThis.document.prerendering = true;
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		start({ sink: { send: async () => {} }, telemetry, botDetection: false });
		await settle();
		expect(capturedEmit).toBeNull();
		expect(telemetry.pings).toEqual([]);
		expect(globalThis.document.cookie).toBe("");
	});

	test("error-threshold termination emits a recorder_fault ping carrying the last error", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			errorThreshold: 2,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(meta(2000));
		capturedEmit(straddle(2001, -600)); // counting error 1
		capturedEmit(straddle(2002, -600)); // counting error 2 -> terminate
		expect(stopRecordingCalls).toBe(1);

		const faults = telemetry.pings.filter((e) => e.metric === "recorder_fault");
		expect(faults).toHaveLength(1);
		expect(faults[0].reason).toBe("terminated");
		expect(faults[0].error.length).toBeGreaterThan(0);
		expect(faults[0].error.length).toBeLessThanOrEqual(256);
		// The fault carries the page it happened on: acting on a fault starts there.
		expect(faults[0].url).toBe(window.location.href);
	});

	test("a throw inside capture lands on the recorder's own fault plane, is never swallowed, and never ends the recording", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			// One: a capture throw is partial capture loss, not the whole-recorder failure the stop
			// threshold ends, so even the very first one must not spend against it.
			errorThreshold: 1,
			intervalMs: 1_000_000,
		});

		// rrweb suppresses the error only on `true`; anything else rethrows to the page's own
		// global error reporting, which is where an uncaught error belongs.
		expect(capturedErrorHandler(new Error("observer blew up"))).not.toBe(true);
		const faults = telemetry.pings.filter((e) => e.metric === "recorder_fault");
		expect(faults.map((f) => f.reason)).toEqual(["capture_failed"]);
		expect(faults[0].error).toContain("observer blew up");

		// A throwing observer throws for as long as the page churns: the recording continues on
		// the observers that still work, the recorder keeps its first reading, and every throw
		// reaches the page.
		for (let i = 0; i < 20; i++) {
			expect(capturedErrorHandler(new Error("again"))).not.toBe(true);
		}
		expect(stopRecordingCalls).toBe(0);
		expect(
			telemetry.pings.filter((e) => e.metric === "recorder_fault"),
		).toHaveLength(1);
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		expect(telemetry.pings.some((e) => e.metric === "page_load")).toBe(true);
	});

	test("a run of counting errors terminates the recorder rather than harming the page unboundedly", async () => {
		// An out-of-bounds move straddle (stream.js) is the counting error driven here.
		await start({
			sink: { send: async () => {} },
			botDetection: false,
			errorThreshold: 2,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(meta(2000)); // prior slice ends at 1000, current slice starts at 2000
		capturedEmit(straddle(2001, -600)); // true start 1401: out of bounds -> counting error 1
		expect(stopRecordingCalls).toBe(0);
		capturedEmit(straddle(2002, -600)); // true start 1402: counting error 2 -> terminate
		expect(stopRecordingCalls).toBe(1);
	});
});

describe("start() capture-correctness", () => {
	test("PageLoad fires exactly once, on the first FullSnapshot, carrying url/title/referrer", async () => {
		const chunks = [];
		const r = await start({
			sink: {
				send: async (b) => {
					chunks.push(b);
				},
			},
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		capturedEmit(fullSnapshot(2001)); // a second snapshot must not re-fire PageLoad
		await settle();
		await r.flush();

		const events = chunks.flatMap((b) => decode(b).events);
		const pageLoads = events.filter((e) => e.type === EventType.PageLoad);
		expect(pageLoads).toHaveLength(1);
		expect(pageLoads[0].data).toEqual({
			url: "https://client.example.com/p",
			title: "Home",
			referrer: "https://ref.example/from",
		});
	});

	test("the first FullSnapshot emits a page_load ping on the rates plane, in the birth slice", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		capturedEmit(fullSnapshot(2001)); // a second snapshot must not re-fire the ping
		await settle();

		const born = telemetry.pings.find((e) => e.metric === "slice_started");
		const loads = telemetry.pings.filter((e) => e.metric === "page_load");
		expect(loads).toHaveLength(1);
		expect(loads[0]).toMatchObject({
			visitorId: born.visitorId,
			sliceId: born.sliceId,
			url: "https://client.example.com/p",
			referrer: "https://ref.example/from",
			spaRoute: false,
		});
	});

	test("no ping carries a query string — the recording keeps the arrival whole, the telemetry plane gets the address", async () => {
		// A query string is where an email, a reset token, or an ad network's click ids ride, and the
		// telemetry plane is unmasked by construction, so every url on it is an address.
		globalThis.window.location.href =
			"https://client.example.com/checkout?email=a%40b.com&gclid=xyz";
		globalThis.document.referrer = "https://ref.example/from?token=secret";
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const chunks = [];
		const r = await start({
			sink: {
				send: async (b) => {
					chunks.push(b);
				},
			},
			telemetry,
			botDetection: false,
			errorThreshold: 1,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		await r.flush();
		capturedEmit(straddle(1002, -600)); // a counting error -> a terminated fault

		const born = telemetry.pings.find((e) => e.metric === "slice_started");
		const load = telemetry.pings.find((e) => e.metric === "page_load");
		const fault = telemetry.pings.find((e) => e.metric === "recorder_fault");
		expect(born.url).toBe("https://client.example.com/checkout");
		expect(load.url).toBe("https://client.example.com/checkout");
		expect(load.referrer).toBe("https://ref.example/from");
		expect(fault.url).toBe("https://client.example.com/checkout");
		for (const ping of telemetry.pings) {
			expect(JSON.stringify(ping)).not.toContain("secret");
			expect(JSON.stringify(ping)).not.toContain("gclid");
		}

		const recorded = chunks
			.flatMap((b) => decode(b).events)
			.find((e) => e.type === EventType.PageLoad);
		expect(recorded.data.url).toBe(
			"https://client.example.com/checkout?email=a%40b.com&gclid=xyz",
		);
		expect(recorded.data.referrer).toBe(
			"https://ref.example/from?token=secret",
		);
	});

	test("a bot-gated page context's ping carries the address, not the query", async () => {
		botIsBot = true;
		globalThis.window.location.href = "https://client.example.com/l?gclid=xyz";
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({ sink: { send: async () => {} }, telemetry });
		const [gated] = telemetry.pings;
		expect(gated.metric).toBe("capture_gated");
		expect(gated.url).toBe("https://client.example.com/l");
	});

	// A visitor whose id cannot reach their next page records normally, so nothing but the birth
	// distinguishes those page contexts.
	test("every birth says how the visitor's identity was come by", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		await settle();

		const births = telemetry.pings.filter((e) => e.metric === "slice_started");
		expect(births.length).toBeGreaterThan(0);
		for (const born of births) expect(born.visitor_source).toBe("written");
	});

	test("a deployment supplying its own id can say how it came by it", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
			visitorId: "Supplied_Id-01",
			visitorSource: "server_session",
		});
		await settle();
		expect(
			telemetry.pings.find((e) => e.metric === "slice_started"),
		).toMatchObject({
			visitorId: "Supplied_Id-01",
			visitor_source: "server_session",
		});
	});

	test("a supplied id with no stated source leaves the question open", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
			visitorId: "Supplied_Id-01",
		});
		await settle();
		expect(
			telemetry.pings.find((e) => e.metric === "slice_started").visitor_source,
		).toBe("undeclared");
	});

	test("a poisoned visitor cookie never reaches the chunk key", async () => {
		globalThis.document.cookie = "locusVisitorId=bad id/with spaces";
		const descriptors = [];
		const r = await start({
			sink: {
				send: async (_b, d) => {
					descriptors.push(d);
				},
			},
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		await r.flush();

		expect(descriptors.length).toBeGreaterThan(0);
		for (const d of descriptors) {
			expect(d.visitorId).toMatch(/^[A-Za-z0-9_-]{8,64}$/);
			expect(d.visitorId).not.toBe("bad id/with spaces");
		}
	});

	test("on becoming visible again, a PageVisible marker is buffered with the url", async () => {
		const chunks = [];
		const r = await start({
			sink: {
				send: async (b) => {
					chunks.push(b);
				},
			},
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000)); // open a slice so markers are stampable
		globalThis.document.visibilityState = "hidden";
		docListeners.visibilitychange();
		globalThis.document.visibilityState = "visible";
		docListeners.visibilitychange();
		await settle();
		await r.flush();

		const events = chunks.flatMap((b) => decode(b).events);
		const visible = events.filter((e) => e.type === EventType.PageVisible);
		expect(visible).toHaveLength(1);
		expect(visible[0].data.url).toBe("https://client.example.com/p");
	});

	test("a bounce-before-DOM-ready page still beacons its PageHidden arrival facts", async () => {
		const beaconChunks = [];
		await start({
			sink: {
				send: async () => {},
				beacon: (b) => {
					beaconChunks.push(b);
				},
			},
			botDetection: false,
			intervalMs: 1_000_000,
		});
		// No meta()/fullSnapshot() — DOM never readied, so rrweb emitted no Meta+FullSnapshot.
		globalThis.document.visibilityState = "hidden";
		docListeners.visibilitychange();

		const beaconEvents = beaconChunks.flatMap((b) => decode(b).events);
		const hidden = beaconEvents.filter((e) => e.type === EventType.PageHidden);
		expect(hidden).toHaveLength(1);
		expect(hidden[0].data.url).toBe("https://client.example.com/p");
	});

	test("the birth ping fires at start(), before rrweb has emitted anything", async () => {
		const pings = [];
		await start({
			sink: { send: async () => {} },
			telemetry: { emit: (e) => pings.push(e) },
			botDetection: false,
			intervalMs: 1_000_000,
		});

		const births = pings.filter((p) => p.metric === "slice_started");
		expect(births).toHaveLength(1);
		expect(births[0].first_slice).toBe(true);
		expect(births[0].sliceId).toMatch(/^\d{14}-[a-z0-9]{4}$/);
	});

	// The two contexts share the visitor's IndexedDB, which is the thing that could carry a slice id
	// across.
	test("a second page context stamps its own head slice, never the previous context's", async () => {
		const descriptors = [];
		const sink = {
			send: async (_b, d) => {
				descriptors.push(d);
			},
		};

		await start({ sink, botDetection: false, intervalMs: 1_000_000 });
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		const first = await start({
			sink,
			botDetection: false,
			intervalMs: 1_000_000,
		});
		// The second context's rrweb has not emitted yet; this marker beats its record-start Meta.
		globalThis.document.visibilityState = "hidden";
		docListeners.visibilitychange();
		await first.flush();

		const slices = new Set(descriptors.map((d) => d.sliceId));
		expect(slices.size).toBe(2);
	});

	// The override reaches the chunk key as given, even outside the recorder's own cookie charset —
	// the asymmetry with the self-healing cookie is deliberate (index.js).
	test("a deliberate visitorId override is honored verbatim, not reformatted", async () => {
		const custom = "MyClient.User.42";
		const descriptors = [];
		const r = await start({
			sink: {
				send: async (_b, d) => {
					descriptors.push(d);
				},
			},
			visitorId: custom,
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		await r.flush();

		expect(descriptors.length).toBeGreaterThan(0);
		for (const d of descriptors) {
			expect(d.visitorId).toBe(custom);
		}
	});

	test("PageLoad lands in the page-load slice, immediately behind the covering snapshot", async () => {
		const chunks = [];
		const r = await start({
			sink: {
				send: async (b) => {
					chunks.push(b);
				},
			},
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		await r.flush();

		const decoded = chunks.map(decode);
		const snapshotChunk = decoded.find((c) =>
			c.events.some((e) => e.type === EventType.FullSnapshot),
		);
		expect(snapshotChunk).toBeDefined();
		const types = snapshotChunk.events.map((e) => e.type);
		const snapIdx = types.indexOf(EventType.FullSnapshot);
		const loadIdx = types.indexOf(EventType.PageLoad);
		// Same slice (one chunk per slice), and PageLoad is the very next event after the snapshot.
		expect(loadIdx).toBe(snapIdx + 1);
	});

	test("a buffer that fails twice mid-life degrades to memory and keeps shipping instead of terminating", async () => {
		// A factory whose connections always refuse transactions: the first write dies, heals once
		// (the reopen succeeds), dies again — out of allowance.
		const db = {
			transaction: () => {
				const error = new Error("The database connection is closing.");
				error.name = "InvalidStateError";
				throw error;
			},
			close: () => {},
		};
		globalThis.indexedDB = {
			open: () => {
				const request = {};
				queueMicrotask(() => {
					request.result = db;
					request.onsuccess?.();
				});
				return request;
			},
		};
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const chunks = [];
		const r = await start({
			sink: {
				send: async (b) => {
					chunks.push(b);
				},
			},
			telemetry,
			botDetection: false,
			errorThreshold: 2,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		await r.flush();

		expect(stopRecordingCalls).toBe(0);
		const reasons = telemetry.pings
			.filter((e) => e.metric === "recorder_fault")
			.map((e) => e.reason);
		expect(reasons).toContain("buffer_not_durable");
		expect(reasons).not.toContain("terminated");
		const events = chunks.flatMap((b) => decode(b).events);
		expect(events.some((e) => e.type === EventType.FullSnapshot)).toBe(true);
	});

	test("pagehide closes the buffer and pageshow reopens it — the refused write is reported, nothing terminates", async () => {
		const chunks = [];
		const r = await start({
			sink: {
				send: async (b) => {
					chunks.push(b);
				},
			},
			botDetection: false,
			errorThreshold: 2,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		await settle(); // committed before the teardown
		globalThis.window.dispatch("pagehide");
		capturedEmit(fullSnapshot(1001)); // refused by the closed buffer
		await settle();
		globalThis.window.dispatch("pageshow");
		capturedEmit(fullSnapshot(2001)); // capture resumed
		await settle();
		await r.flush();

		expect(stopRecordingCalls).toBe(0); // refused teardown writes never count
		const decoded = chunks.map(decode);
		const timestamps = decoded.flatMap((c) => c.events).map((e) => e.timestamp);
		expect(timestamps).toContain(1000);
		expect(timestamps).toContain(2001);
		expect(timestamps).not.toContain(1001); // lost to the closed buffer...
		expect(decoded.flatMap((c) => c.errors)) // ...and named, never silent
			.toContainEqual(expect.stringContaining("buffer closed during append"));
	});

	test("a write killed by teardown before pagehide runs is not reported as a broken device", async () => {
		// The ordering the fleet actually delivers: the browser force-closes the connection at
		// navigation commit and the write's rejection reaches the recorder BEFORE the page's own
		// pagehide listener runs, so buffer.close() has not happened and nothing is a
		// BufferClosedError. The recorder cannot tell this from a device whose storage is broken —
		// both look like an operation that failed, then a reopen that failed — so it must not name
		// one of them.
		let open = () => {
			const request = {};
			queueMicrotask(() => {
				request.result = {
					transaction: () => {
						const error = new Error("The database connection is closing.");
						error.name = "InvalidStateError";
						throw error;
					},
					close: () => {},
				};
				request.onsuccess?.();
			});
			return request;
		};
		globalThis.indexedDB = { open: (...args) => open(...args) };
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			errorThreshold: 2,
			intervalMs: 1_000_000,
		});
		// The page is dying: no new connection is coming.
		open = () => {
			const request = {};
			queueMicrotask(() => {
				request.error = new Error("dying context");
				request.onerror?.();
			});
			return request;
		};
		capturedEmit(meta(1000));
		await settle(); // the rejection lands first...
		globalThis.window.dispatch("pagehide"); // ...and only then does teardown run
		await settle();

		const reasons = telemetry.pings
			.filter((e) => e.metric === "recorder_fault")
			.map((e) => e.reason);
		expect(reasons).not.toContain("buffer_unavailable");
		expect(reasons).not.toContain("terminated");
		expect(stopRecordingCalls).toBe(0);
	});

	test("stop unregisters the teardown listeners along with everything else", async () => {
		const r = await start({
			sink: { send: async () => {} },
			botDetection: false,
			intervalMs: 1_000_000,
		});
		expect(globalThis.window.listenerCount("pagehide")).toBe(1);
		expect(globalThis.window.listenerCount("pageshow")).toBe(1);
		// The document side too: a stopped recorder that still answers visibilitychange, freeze, or
		// resume goes on recording markers and shooting beacons out of a torn-down context.
		for (const type of ["visibilitychange", "freeze", "resume"]) {
			expect(docListeners[type]).toBeDefined();
		}
		r.stop();
		expect(globalThis.window.listenerCount("pagehide")).toBe(0);
		expect(globalThis.window.listenerCount("pageshow")).toBe(0);
		for (const type of ["visibilitychange", "freeze", "resume"]) {
			expect(docListeners[type]).toBeUndefined();
		}
	});

	test("a null rejection out of the sink reaches telemetry as an honest description, not the string 'null'", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const r = await start({
			sink: { send: async () => Promise.reject(null) },
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		await r.flush();

		const fault = telemetry.pings.find(
			(e) => e.metric === "recorder_fault" && e.reason === "drain_failed",
		);
		expect(fault).toBeDefined();
		expect(fault.error).toBe("null was thrown (no error object)");
	});

	test("the channel-assessment probe fires before the birth, naming the head slice in the store's shape", async () => {
		const calls = [];
		const telemetry = {
			emit: (e) => calls.push(e),
			probe: (e) => calls.push(e),
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
		});

		const probeIdx = calls.findIndex((e) => e.metric === "transport_probe");
		const birthIdx = calls.findIndex((e) => e.metric === "slice_started");
		expect(probeIdx).toBeGreaterThanOrEqual(0);
		// The carrier reads the standing transport for every later shot off this ping's fate, so
		// it must precede the birth.
		expect(probeIdx).toBeLessThan(birthIdx);
		const probe = calls[probeIdx];
		expect(probe.sliceId).toBe(calls[birthIdx].sliceId);
		// The store gates the ping by shape (worker.js): a drifted probe bounces and the
		// blocked-beacon share goes blind.
		expect(probe.sliceId).toMatch(/^\d{14}-[a-z0-9]{4}$/);
		expect(probe.visitorId.length).toBeGreaterThan(0);
		expect(typeof probe.recorderVersion).toBe("string");
	});

	// The delivered-count and the cap trim eat the same end of the queue, so a delivery that forgets
	// its whole snapshot forgets the records the trim already took plus that many of the newest —
	// testimony that never shipped, dropped as if it had.
	test("a delivery forgets only the records it shipped, never ones the cap trimmed under it", async () => {
		let release;
		const gate = new Promise((r) => {
			release = r;
		});
		let reached;
		const arrived = new Promise((r) => {
			reached = r;
		});
		const chunks = [];
		const r = await start({
			sink: {
				send: async (b) => {
					chunks.push(b);
					reached();
					await gate;
				},
			},
			botDetection: false,
			errorThreshold: 1000,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(meta(2000)); // prior slice [.., 1000], current opens at 2000
		// Exactly the cap: 100 out-of-bounds straddles, true starts 1401..1500.
		for (let i = 0; i < 100; i += 1) capturedEmit(straddle(2001 + i, -600));
		await settle();

		const shipping = r.flush();
		await arrived; // the carrier chunk is at the sink, holding all 100

		// Mid-flight: 20 more, true starts 1501..1520. The queue is at its cap, so admitting them
		// evicts 20 of the records already riding the in-flight chunk.
		for (let i = 0; i < 20; i += 1) capturedEmit(straddle(4001 + i, -2500));
		release();
		await shipping;
		await settle();
		await r.flush();

		const shipped = chunks.map(decode).flatMap((c) => c.errors);
		for (let n = 1501; n <= 1520; n += 1) {
			expect(shipped.some((e) => e.includes(`true start ${n}`))).toBe(true);
		}
		expect(new Set(shipped).size).toBe(shipped.length); // and none shipped twice
	});

	test("the error queue caps at the newest records — an undeliverable context never grows it unboundedly", async () => {
		const chunks = [];
		const r = await start({
			sink: {
				send: async (b) => {
					chunks.push(b);
				},
			},
			botDetection: false,
			errorThreshold: 1000,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(meta(2000));
		// 120 out-of-bounds straddles, each described with its own true start: 1401..1520.
		for (let i = 0; i < 120; i += 1) {
			capturedEmit(straddle(2001 + i, -600));
		}
		await settle();
		await r.flush();

		const errors = chunks.map(decode).flatMap((c) => c.errors);
		expect(errors).toHaveLength(100);
		// The newest survive, oldest dropped — testimony of what was failing when it last failed.
		expect(errors[0]).toContain("true start 1421");
		expect(errors.at(-1)).toContain("true start 1520");
	});

	test("backlog eviction faults by name — capture displacing undelivered records is never silent", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
			maxBacklogBytes: 1,
		});
		// The ring enforces against the durable byte counter, so the first append must land and the
		// counter seed settle before a later append has anything to evict.
		capturedEmit(meta(1000));
		await settle();
		capturedEmit(fullSnapshot(1001));
		await settle();

		const fault = telemetry.pings.find(
			(e) => e.metric === "recorder_fault" && e.reason === "backlog_evicted",
		);
		expect(fault).toBeDefined();
		expect(fault.error).toContain("evicted");
	});

	test("a write the buffer itself cannot take faults by name — buffer_write_failed, never silence", async () => {
		// An event no serialization can hold (a BigInt smuggled into event data by a hostile or
		// broken page API) makes the append itself impossible: not a teardown, not a fallen
		// backend, so it must reach the fault plane by its own name.
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit({
			type: EventType.IncrementalSnapshot,
			data: { source: IncrementalSource.Mutation, poisoned: 1n },
			timestamp: 1001,
		});
		await settle();

		const reasons = telemetry.pings
			.filter((e) => e.metric === "recorder_fault")
			.map((e) => e.reason);
		expect(reasons).toContain("buffer_write_failed");
	});

	test("a poison batch's drop faults by name — permanent data loss is never silent", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		const r = await start({
			sink: {
				send: async () => {
					throw storeRejection("400 malformed chunk");
				},
			},
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		capturedEmit(fullSnapshot(1001));
		await settle();
		// One store refusal per flush; the default attempt cap is 5 (uploader.js).
		for (let i = 0; i < 5; i += 1) await r.flush();

		const reasons = telemetry.pings
			.filter((e) => e.metric === "recorder_fault")
			.map((e) => e.reason);
		expect(reasons).toContain("poison_dropped");
	});

	test("on hidden, a PageHidden marker is both buffered and beaconed", async () => {
		const beaconChunks = [];
		await start({
			sink: {
				send: async () => {},
				beacon: (b) => {
					beaconChunks.push(b);
				},
			},
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000)); // open a slice so the marker is stampable
		globalThis.document.visibilityState = "hidden";
		docListeners.visibilitychange();

		const beaconEvents = beaconChunks.flatMap((b) => decode(b).events);
		const hidden = beaconEvents.filter((e) => e.type === EventType.PageHidden);
		expect(beaconChunks).toHaveLength(1);
		expect(hidden).toHaveLength(1);
		expect(hidden[0].data.url).toBe("https://client.example.com/p");

		// Hidden is the last moment a page can reliably observe, which is the whole reason the shot
		// exists — becoming visible again is not that moment, and a beacon there spends the
		// visitor's keepalive budget on a chunk the ordinary drain already owns.
		globalThis.document.visibilityState = "visible";
		docListeners.visibilitychange();
		expect(beaconChunks).toHaveLength(1);
	});

	// Two rrweb options are set explicitly and neither is negotiable-by-accident: recordDOM, without
	// which takeFullSnapshot returns early and no snapshot is ever captured (pinned by the masking
	// suite), and the periodic checkout cadence, which is what keeps a stream self-healing — a
	// missing one means one FullSnapshot per page context, ever.
	test("the record() options carry the checkout cadence, and the cadence defaults are a rule's to move", async () => {
		const r = await start({
			sink: { send: async () => {} },
			botDetection: false,
			intervalMs: 1_000_000,
		});
		expect(capturedRecordOptions.checkoutEveryNms).toBe(1_800_000);
		expect(capturedRecordOptions.userTriggeredOnInput).toBe(true);
		r.stop();

		const tuned = await start({
			sink: { send: async () => {} },
			botDetection: false,
			intervalMs: 1_000_000,
			rrwebRules: [
				{
					pattern: "*",
					options: { checkoutEveryNms: 60_000, userTriggeredOnInput: false },
				},
			],
		});
		expect(capturedRecordOptions.checkoutEveryNms).toBe(60_000);
		expect(capturedRecordOptions.userTriggeredOnInput).toBe(false);
		tuned.stop();
	});

	// A broken buffer fails on every write, a throwing observer throws on every event: the fault
	// names the failure, not its rate, and a channel that repeated it would cost the visitor a shot
	// per event.
	test("a fault goes out once per reason per page context, however often the failure repeats", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			errorThreshold: 1_000,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000));
		// A BigInt in event data makes the append itself impossible, and a page API that smuggles
		// one in does it on every event it touches — the repeating failure the dedupe is for.
		for (let i = 0; i < 5; i++) {
			capturedEmit({
				type: EventType.IncrementalSnapshot,
				data: { source: IncrementalSource.Mutation, poisoned: 1n },
				timestamp: 1001 + i,
			});
		}
		await settle();

		const faults = telemetry.pings.filter((e) => e.metric === "recorder_fault");
		expect(faults.map((f) => f.reason)).toEqual(["buffer_write_failed"]);
	});

	test("only the context's own first slice claims the birth — later slices say so", async () => {
		const telemetry = {
			pings: [],
			emit(e) {
				this.pings.push(e);
			},
		};
		await start({
			sink: { send: async () => {} },
			telemetry,
			botDetection: false,
			intervalMs: 1_000_000,
		});
		capturedEmit(meta(1000)); // fills the head slice
		capturedEmit(meta(2000)); // a checkout: opens a second slice
		capturedEmit(meta(3000)); // and a third

		const births = telemetry.pings.filter((e) => e.metric === "slice_started");
		expect(births).toHaveLength(3);
		expect(births.map((b) => b.first_slice)).toEqual([true, false, false]);
	});
});
