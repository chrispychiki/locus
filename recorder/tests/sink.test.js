import { describe, expect, jest, test } from "bun:test";

import {
	faultPing,
	gatedPing,
	httpSink,
	httpTelemetry,
	isStoreRejection,
	MAX_BEACON_BYTES,
	storeRejection,
	storeSink,
	transportCarrier,
} from "../src/sink.js";
import { RECORDER_VERSION } from "../src/version.js";
import { UNIDENTIFIED_VISITOR } from "../src/visitor.js";

/**
 * One fake page's Resource Timing: installs a PerformanceObserver double on the globals;
 * land(...entries) delivers entries to every live observer as the real one would. Entries are
 * plain {name, initiatorType, transferSize}.
 */
function fakeTiming() {
	const observers = [];
	const realObserver = globalThis.PerformanceObserver;
	globalThis.PerformanceObserver = class {
		constructor(cb) {
			this.cb = cb;
		}
		observe() {
			observers.push(this);
		}
		disconnect() {
			const i = observers.indexOf(this);
			if (i >= 0) observers.splice(i, 1);
		}
	};
	return {
		land: (...entries) => {
			for (const o of observers.slice()) o.cb({ getEntries: () => entries });
		},
		/** How many observers are still watching the page — an assessment that never disconnects leaves one here for the document's life. */
		live: () => observers.length,
		restore: () => {
			if (realObserver === undefined) delete globalThis.PerformanceObserver;
			else globalThis.PerformanceObserver = realObserver;
		},
	};
}

const capturing =
	(arr, result = true) =>
	(url, bodyOrOpts) => {
		arr.push([url, bodyOrOpts]);
		return result;
	};

describe("transportCarrier", () => {
	test("shots ride keepalive fetch until any verdict lands", () => {
		const fetches = [];
		const carrier = transportCarrier({
			beaconFn: () => {
				throw new Error("no shot may reach the beacon pre-verdict");
			},
			fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
		});
		carrier.shoot("https://store.test/t", "body");
		expect(fetches.length).toBe(1);
		const [target, opts] = fetches[0];
		expect(target).toBe("https://store.test/t");
		expect(opts.method).toBe("POST");
		expect(opts.keepalive).toBe(true);
		expect(opts.headers).toBeUndefined();
	});

	test("a delivered probe — nonzero transferSize on its entry — verdicts the beacon channel", () => {
		const timing = fakeTiming();
		try {
			const beacons = [],
				fetches = [];
			const carrier = transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			});
			carrier.assess("https://store.test/t", "probe");
			expect(beacons.length).toBe(1);
			expect(beacons[0]).toEqual(["https://store.test/t", "probe"]);

			timing.land({
				name: "https://store.test/t",
				initiatorType: "beacon",
				transferSize: 300,
			});
			carrier.shoot("https://store.test/t", "ping");
			expect(beacons.length).toBe(2);
			expect(fetches.length).toBe(0);
		} finally {
			timing.restore();
		}
	});

	test("a provably blocked probe — zero transferSize while a fetch entry on the same URL reads nonzero — verdicts keepalive fetch", async () => {
		// Only an entry on the probed URL itself can prove the field: a nonzero size elsewhere on the
		// page proves nothing, because WebKit fills transferSize same-origin while reading zero for
		// every cross-origin entry.
		const timing = fakeTiming();
		try {
			const beacons = [],
				fetches = [];
			const carrier = transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			});
			carrier.assess("https://store.test/t", "probe");
			timing.land(
				{
					name: "https://store.test/t",
					initiatorType: "beacon",
					transferSize: 0,
				},
				{
					name: "https://store.test/t",
					initiatorType: "fetch",
					transferSize: 300,
				},
			);

			carrier.shoot("https://store.test/t", "ping");
			expect(beacons.length).toBe(1); // the probe only — the block was read off its entry
			expect(fetches.length).toBe(1);
			// The verdict is settled, not merely the pre-verdict default still standing: the proof
			// closed the assessment, so the later evidence that would otherwise read "field dead"
			// — a zero-size entry for a fetch whose promise resolved — cannot reopen it.
			expect(timing.live()).toBe(0);
			await Promise.resolve();
			timing.land({
				name: "https://store.test/t",
				initiatorType: "fetch",
				transferSize: 0,
			});
			carrier.shoot("https://store.test/t", "ping2");
			expect(beacons.length).toBe(1);
			expect(fetches.length).toBe(2);
		} finally {
			timing.restore();
		}
	});

	test("the assessment stops watching at its deadline — no observer outlives the window on the host page", () => {
		const timing = fakeTiming();
		jest.useFakeTimers();
		try {
			const beacons = [],
				fetches = [];
			const carrier = transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			});
			carrier.assess("https://store.test/t", "probe");
			expect(timing.live()).toBe(1);
			jest.advanceTimersByTime(6 * 60_000);
			expect(timing.live()).toBe(0);
			// Past the deadline the verdict simply stays unproven; shots keep the blind-safe channel.
			carrier.shoot("https://store.test/t", "ping");
			expect(fetches.length).toBe(1);
			expect(beacons.length).toBe(1);
		} finally {
			jest.useRealTimers();
			timing.restore();
		}
	});

	test("a zero probe alone verdicts nothing — a nonzero entry elsewhere on the page proves no field, and shots stay on the blind-safe fetch", () => {
		const timing = fakeTiming();
		try {
			const beacons = [],
				fetches = [];
			const carrier = transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			});
			carrier.assess("https://store.test/t", "probe");
			timing.land(
				{
					name: "https://site/app.js",
					initiatorType: "script",
					transferSize: 4096,
				},
				{
					name: "https://store.test/t",
					initiatorType: "other",
					transferSize: 0,
				},
			);

			carrier.shoot("https://store.test/t", "ping");
			expect(beacons.length).toBe(1); // the probe only — an unresolved zero never rides beacon
			expect(fetches.length).toBe(1);
		} finally {
			timing.restore();
		}
	});

	test("a zero probe resolves on the witness's late nonzero entry — the block was real, the channel is fetch, and no shot rode beacon in the gap", () => {
		const timing = fakeTiming();
		try {
			const beacons = [],
				fetches = [];
			const carrier = transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			});
			carrier.assess("https://store.test/t", "probe");
			timing.land({
				name: "https://store.test/t",
				initiatorType: "other",
				transferSize: 0,
			});

			carrier.shoot("https://store.test/t", "ping"); // in the gap: still fetch
			timing.land({
				name: "https://store.test/t",
				initiatorType: "fetch",
				transferSize: 300,
			});
			carrier.shoot("https://store.test/t", "ping");
			expect(beacons.length).toBe(1); // the probe only
			expect(fetches.length).toBe(2);
		} finally {
			timing.restore();
		}
	});

	test("a zero probe beside a zero-size fetch entry that provably delivered verdicts the beacon — the field is dead", async () => {
		// WebKit's shape: transferSize reads zero on every cross-origin entry, TAO or not, so a
		// fetch whose promise resolved while its entry reads zero is the proof the field is dead —
		// and the engine with that shape is the one that needs the beacon.
		const timing = fakeTiming();
		try {
			const beacons = [],
				fetches = [];
			const carrier = transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			});
			carrier.assess("https://store.test/t", "probe");
			carrier.shoot("https://store.test/t", "ping"); // the witness, on fetch
			await new Promise((r) => setTimeout(r, 0)); // its promise resolves: delivery proven
			timing.land(
				{
					name: "https://store.test/t",
					initiatorType: "other",
					transferSize: 0,
				},
				{
					name: "https://store.test/t",
					initiatorType: "fetch",
					transferSize: 0,
				},
			);

			carrier.shoot("https://store.test/t", "ping");
			expect(beacons.length).toBe(2); // the probe + the post-verdict shot
			expect(fetches.length).toBe(1); // the witness only
		} finally {
			timing.restore();
		}
	});

	test("the field-dead verdict lands whichever arrives last, the zero entries or the delivery proof", async () => {
		const timing = fakeTiming();
		try {
			const beacons = [],
				fetches = [];
			let resolveFetch;
			const carrier = transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: (u, o) => {
					fetches.push([u, o]);
					return new Promise((r) => {
						resolveFetch = r;
					});
				},
			});
			carrier.assess("https://store.test/t", "probe");
			carrier.shoot("https://store.test/t", "ping");
			timing.land(
				{
					name: "https://store.test/t",
					initiatorType: "other",
					transferSize: 0,
				},
				{
					name: "https://store.test/t",
					initiatorType: "fetch",
					transferSize: 0,
				},
			);

			carrier.shoot("https://store.test/t", "ping"); // zero entries alone prove nothing yet
			expect(beacons.length).toBe(1);
			expect(fetches.length).toBe(2);

			resolveFetch({ ok: true });
			await new Promise((r) => setTimeout(r, 0));
			carrier.shoot("https://store.test/t", "ping");
			expect(beacons.length).toBe(2);
		} finally {
			timing.restore();
		}
	});

	test("without Resource Timing the probe verdicts the beacon — the default where nothing is known", () => {
		const realObserver = globalThis.PerformanceObserver;
		delete globalThis.PerformanceObserver;
		try {
			const beacons = [],
				fetches = [];
			const carrier = transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			});
			carrier.assess("https://store.test/t", "probe");
			carrier.shoot("https://store.test/t", "ping");
			expect(beacons.length).toBe(2);
			expect(fetches.length).toBe(0);
		} finally {
			globalThis.PerformanceObserver = realObserver;
		}
	});

	test("without sendBeacon no probe fires and everything rides keepalive fetch", () => {
		const fetches = [];
		const carrier = transportCarrier({
			beaconFn: null,
			fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
		});
		carrier.assess("https://store.test/t", "probe");
		carrier.shoot("https://store.test/t", "ping");
		expect(fetches.length).toBe(1);
		expect(fetches[0][1].body).toBe("ping");
	});

	test("the assessment fires at most once per carrier", () => {
		const beacons = [];
		const carrier = transportCarrier({
			beaconFn: capturing(beacons),
			fetchFn: () => Promise.resolve({ ok: true }),
		});
		carrier.assess("https://store.test/t", "probe");
		carrier.assess("https://store.test/t", "probe");
		expect(beacons.length).toBe(1);
	});

	test("on the beacon channel a refused shot — false or a throw — falls back to a keepalive fetch, that shot only", () => {
		const timing = fakeTiming();
		try {
			const fetches = [];
			let beaconResult = true;
			const carrier = transportCarrier({
				beaconFn: () => {
					if (beaconResult === "throw") throw new Error("down");
					return beaconResult;
				},
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			});
			carrier.assess("https://store.test/t", "probe");
			timing.land({
				name: "https://store.test/t",
				initiatorType: "beacon",
				transferSize: 300,
			});

			beaconResult = false;
			carrier.shoot("https://store.test/t", "ping"); // returned false
			beaconResult = "throw";
			carrier.shoot("https://store.test/t", "ping"); // threw
			expect(fetches.length).toBe(2);
			expect(fetches.every(([, o]) => o.keepalive === true)).toBe(true);
		} finally {
			timing.restore();
		}
	});

	test("a rejected fetch shot is swallowed, never an unhandled rejection", async () => {
		const carrier = transportCarrier({
			beaconFn: null,
			fetchFn: () => Promise.reject(new TypeError("Failed to fetch")),
		});
		carrier.shoot("https://store", "ping");
		await new Promise((r) => setTimeout(r, 0));
	});
});

describe("the refusal mark", () => {
	test("rides the error's name, so a sink bundled apart from the uploader is read the same", () => {
		const minted = storeRejection("413 chunk too large");
		expect(isStoreRejection(minted)).toBe(true);
		expect(String(minted)).toContain("413 chunk too large");
		// All a separately bundled sink can share is the name — no class identity crosses.
		expect(
			isStoreRejection(Object.assign(new Error("no"), { name: minted.name })),
		).toBe(true);
		// Anything unmarked — an injected sink that classifies nothing included — is the environment.
		expect(isStoreRejection(new Error("no"))).toBe(false);
		expect(isStoreRejection(null)).toBe(false);
	});
});

describe("httpSink.send", () => {
	test("posts the raw bytes to the resolved url, with no author-set header", async () => {
		// A header here, or a method outside the safelist, would force a preflight (sink.js).
		const calls = [];
		const sink = httpSink({
			url: (d) => `https://store/chunks/${d.chunkKey}`,
			fetchFn: (u, o) => {
				calls.push([u, o]);
				return Promise.resolve({ ok: true });
			},
		});
		const bytes = new Uint8Array([0x1f, 0x8b, 3]);
		await sink.send(bytes, { chunkKey: "c1" });

		expect(calls[0][0]).toBe("https://store/chunks/c1");
		expect(calls[0][1].method).toBe("POST");
		expect(calls[0][1].headers).toBeUndefined();
		expect(calls[0][1].body).toBe(bytes);
	});

	test("rejects (so the uploader retries) when the response is not ok", async () => {
		const sink = httpSink({
			url: () => "https://store/x",
			fetchFn: () => Promise.resolve({ ok: false, status: 403 }),
		});
		await expect(sink.send(new Uint8Array([1]), {})).rejects.toThrow("403");
	});

	test("a stamped 4xx is the store refusing these bytes, and the rejection says so", async () => {
		for (const status of [400, 403, 404, 413, 422]) {
			const sink = httpSink({
				url: () => "https://store/x",
				fetchFn: () =>
					Promise.resolve({
						ok: false,
						status,
						headers: { get: (name) => (name === "locus-store" ? "1" : null) },
					}),
			});
			const error = await sink.send(new Uint8Array([1]), {}).catch((e) => e);
			expect(isStoreRejection(error)).toBe(true);
		}
	});

	test("an unstamped 4xx is a middlebox answering in the store's place — retried, never blamed on the batch", async () => {
		// A filtering proxy's 403 and a captive portal's 404 wear the store's own codes; only the
		// store's stamp makes a status its verdict. The batch waits in the backlog and delivers
		// when the environment changes.
		for (const status of [400, 403, 404, 413, 422]) {
			const sink = httpSink({
				url: () => "https://store/x",
				fetchFn: () =>
					Promise.resolve({ ok: false, status, headers: { get: () => null } }),
			});
			const error = await sink.send(new Uint8Array([1]), {}).catch((e) => e);
			expect(String(error)).toContain(String(status));
			expect(isStoreRejection(error)).toBe(false);
		}
	});

	test("a response with no readable headers at all classifies as the environment", async () => {
		const sink = httpSink({
			url: () => "https://store/x",
			fetchFn: () => Promise.resolve({ ok: false, status: 400 }),
		});
		const error = await sink.send(new Uint8Array([1]), {}).catch((e) => e);
		expect(isStoreRejection(error)).toBe(false);
	});

	test("a 5xx, a 429, and a 408 are the store's own moment — the rejection blames nothing on the batch", async () => {
		// A worker outage, a rate limit, and a proxy's request timeout all answer without ever
		// judging the payload; counting them would turn an outage into a dropped recording.
		for (const status of [408, 429, 500, 502, 503]) {
			const sink = httpSink({
				url: () => "https://store/x",
				fetchFn: () => Promise.resolve({ ok: false, status }),
			});
			const error = await sink.send(new Uint8Array([1]), {}).catch((e) => e);
			expect(String(error)).toContain(String(status));
			expect(isStoreRejection(error)).toBe(false);
		}
	});

	test("a send that gets no response at all — the offline case — blames nothing on the batch", async () => {
		const sink = httpSink({
			url: () => "https://store/x",
			fetchFn: () => Promise.reject(new TypeError("Failed to fetch")),
		});
		const error = await sink.send(new Uint8Array([1]), {}).catch((e) => e);
		expect(isStoreRejection(error)).toBe(false);
	});

	test("aborts a hung upload after timeoutMs so it cannot wedge the single-flight uploader, and the abort blames nothing on the batch", async () => {
		const sink = httpSink({
			url: () => "https://store/x",
			timeoutMs: 10,
			fetchFn: (_u, o) =>
				new Promise((_, reject) => {
					o.signal.addEventListener("abort", () =>
						reject(new Error("aborted")),
					);
				}),
		});
		const error = await sink.send(new Uint8Array([1]), {}).catch((e) => e);
		expect(error).toBeInstanceOf(Error);
		expect(isStoreRejection(error)).toBe(false);
	});

	test("a resume event re-arms the abort deadline — an upload suspended by a tab freeze survives the thaw", async () => {
		// Deadline 150ms, fetch resolves at 220ms, resume at ~100ms pushes the deadline to ~250ms,
		// so the upload lands only if the re-arm happened.
		const listeners = [];
		globalThis.document = {
			addEventListener: (type, fn) => listeners.push([type, fn]),
			removeEventListener: () => {},
		};
		try {
			const sink = httpSink({
				url: () => "https://store/x",
				timeoutMs: 150,
				fetchFn: (_u, o) =>
					new Promise((resolve, reject) => {
						o.signal.addEventListener("abort", () =>
							reject(new Error("aborted by stale deadline")),
						);
						setTimeout(() => resolve({ ok: true }), 220);
					}),
			});
			const inflight = sink.send(new Uint8Array([1]), {});
			await new Promise((r) => setTimeout(r, 100));
			for (const [type, fn] of listeners) if (type === "resume") fn();
			await expect(inflight).resolves.toBeUndefined();
		} finally {
			delete globalThis.document;
		}
	});
});

describe("httpSink.beacon", () => {
	test("refuses an over-cap shot and hands an under-cap one to the carrier as the raw bytes", () => {
		// The bytes go unwrapped: a Blob would carry a Content-Type, and no type sendBeacon may
		// declare can describe gzip, so declaring one drops the shot.
		const timing = fakeTiming();
		try {
			const sent = [];
			const carrier = transportCarrier({
				beaconFn: capturing(sent),
				fetchFn: () => Promise.resolve({ ok: true }),
			});
			carrier.assess("https://store/beacon", "probe");
			timing.land({
				name: "https://store/beacon",
				initiatorType: "beacon",
				transferSize: 300,
			});
			const sink = httpSink({ url: () => "https://store/beacon", carrier });

			expect(sink.beacon(new Uint8Array(MAX_BEACON_BYTES + 1), {})).toBe(false);
			expect(sent).toHaveLength(1); // the probe only

			const bytes = new Uint8Array(16);
			expect(sink.beacon(bytes, {})).toBe(true);
			expect(sent[1][0]).toBe("https://store/beacon");
			expect(sent[1][1]).toBe(bytes);
		} finally {
			timing.restore();
		}
	});

	test("a beacon body of exactly MAX_BEACON_BYTES is allowed — the cap is the Fetch keepalive ceiling, not one below it", () => {
		// The keepalive limit is strictly greater-than: a body of exactly the cap is still sendable.
		const fetches = [];
		const sink = httpSink({
			url: () => "https://store/beacon",
			carrier: transportCarrier({
				beaconFn: null,
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			}),
		});
		expect(sink.beacon(new Uint8Array(MAX_BEACON_BYTES), {})).toBe(true);
		expect(fetches).toHaveLength(1);
	});

	test("chunk shots ride the shared carrier's verdict — a blocked-beacon environment sends them over keepalive fetch", () => {
		// The verdict is a fact about the environment, not about the telemetry channel: the one
		// carrier the deployment shares (global.js) routes the hidden-marker chunk shots by the same
		// assessment the ping probe produced.
		const timing = fakeTiming();
		try {
			const beacons = [],
				fetches = [];
			const carrier = transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			});
			const telemetry = httpTelemetry({
				url: () => "https://store/telemetry/v1",
				carrier,
			});
			const sink = httpSink({
				url: (d) => `https://store/chunks/${d.chunkKey}`,
				carrier,
			});

			telemetry.probe({ metric: "transport_probe", visitorId: "v1" });
			timing.land(
				{
					name: "https://store/telemetry/v1",
					initiatorType: "beacon",
					transferSize: 0,
				},
				{
					name: "https://store/telemetry/v1",
					initiatorType: "fetch",
					transferSize: 300,
				},
			);

			const bytes = new Uint8Array([0x1f, 0x8b]);
			expect(sink.beacon(bytes, { chunkKey: "c1" })).toBe(true);
			expect(beacons).toHaveLength(1); // the probe only
			expect(fetches).toHaveLength(1);
			expect(fetches[0][0]).toBe("https://store/chunks/c1");
			expect(fetches[0][1].body).toBe(bytes);
			expect(fetches[0][1].keepalive).toBe(true);
		} finally {
			timing.restore();
		}
	});
});

describe("httpTelemetry", () => {
	test("emits ride the carrier as a plain JSON string with the queue-moment stamp", () => {
		// A string body is sent as text/plain, which is safelisted, so the ping stays a simple
		// request and never preflights — on either transport.
		const fetches = [];
		const sink = httpTelemetry({
			url: (e) => `https://store/telemetry/snip/${e.visitorId}`,
			carrier: transportCarrier({
				beaconFn: () => {
					throw new Error("emit must not reach the beacon pre-verdict");
				},
				fetchFn: (u, o) => {
					fetches.push([u, o]);
					return Promise.resolve({ ok: true });
				},
			}),
		});

		const before = Date.now();
		sink.emit({
			metric: "slice_started",
			visitorId: "v1",
			sliceId: "s1",
			first_slice: true,
		});

		expect(fetches.length).toBe(1);
		const [target, opts] = fetches[0];
		expect(target).toBe("https://store/telemetry/snip/v1");
		expect(opts.method).toBe("POST");
		expect(opts.keepalive).toBe(true);
		expect(opts.headers).toBeUndefined();
		expect(typeof opts.body).toBe("string");
		const body = JSON.parse(opts.body);
		// The queue-moment stamp: a ping can dispatch long after the moment it reports, so the
		// client's clock rides in the body rather than being inferred at receive.
		expect(body.ts).toBeGreaterThanOrEqual(before);
		expect(body.ts).toBeLessThanOrEqual(Date.now());
		expect(body).toEqual({
			ts: body.ts,
			metric: "slice_started",
			visitorId: "v1",
			sliceId: "s1",
			first_slice: true,
		});
	});

	test("probe hands the carrier its assessment shot as a stamped ping", () => {
		const beacons = [];
		const sink = httpTelemetry({
			url: () => "https://store.test/t",
			carrier: transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: () => Promise.resolve({ ok: true }),
			}),
		});
		sink.probe({ metric: "transport_probe", visitorId: "v1" });
		expect(beacons.length).toBe(1);
		expect(JSON.parse(beacons[0][1]).metric).toBe("transport_probe");
	});

	test("never throws when the transport fails", () => {
		const sink = httpTelemetry({
			url: () => "https://store",
			carrier: transportCarrier({
				beaconFn: () => {
					throw new Error("down");
				},
				fetchFn: () => {
					throw new Error("down too");
				},
			}),
		});
		expect(() => sink.probe({ metric: "transport_probe" })).not.toThrow();
		expect(() => sink.emit({ metric: "slice_started" })).not.toThrow();
	});

	test("the channel's door strips url and referrer to addresses, whatever composed the payload", () => {
		// A hand-authored payload with query'd urls — the door is the last gate before the wire.
		const fetches = [];
		const sink = httpTelemetry({
			url: () => "https://store/t",
			carrier: transportCarrier({
				beaconFn: null,
				fetchFn: (u, o) => {
					fetches.push([u, o]);
					return Promise.resolve({ ok: true });
				},
			}),
		});
		sink.emit({
			metric: "page_load",
			visitorId: "v1",
			url: "https://site.example/checkout?email=a%40b.com&gclid=xyz",
			referrer: "https://ref.example/from?token=secret",
			spaRoute: "0",
		});
		const body = JSON.parse(fetches[0][1].body);
		expect(body.url).toBe("https://site.example/checkout");
		expect(body.referrer).toBe("https://ref.example/from");
		expect(body.visitorId).toBe("v1");
		expect(fetches[0][1].body).not.toContain("secret");
		expect(fetches[0][1].body).not.toContain("gclid");
	});
});

describe("the ping composers", () => {
	// The shapes with more than one author: a facade emits these too, so the wire invariants —
	// metric name, sentinel identity, error truncation, the page as an address — live in the
	// composer, and these tests pin them there.
	const withWindow = (href, fn) => {
		const prior = globalThis.window;
		globalThis.window = { location: { href } };
		try {
			return fn();
		} finally {
			if (prior === undefined) delete globalThis.window;
			else globalThis.window = prior;
		}
	};

	test("faultPing composes the worker's fault shape, page named by address", () => {
		const ping = withWindow("https://site.example/reset?token=secret", () =>
			faultPing("config_unresolved", new Error("boom"), { visitorId: "v1" }),
		);
		expect(ping).toEqual({
			metric: "recorder_fault",
			visitorId: "v1",
			recorderVersion: RECORDER_VERSION,
			sliceId: null,
			reason: "config_unresolved",
			error: "Error: boom",
			url: "https://site.example/reset",
		});
	});

	test("faultPing keys to the sentinel when the device could not be named, and describes a string as itself", () => {
		const ping = withWindow("https://site.example/", () =>
			faultPing(
				"config_unresolved",
				"config answered without a numeric dialPercent",
				{},
			),
		);
		expect(ping.visitorId).toBe(UNIDENTIFIED_VISITOR);
		expect(ping.error).toBe("config answered without a numeric dialPercent");
	});

	test("faultPing truncates the error line and survives a window less than a browser's", () => {
		const prior = globalThis.window;
		globalThis.window = {};
		try {
			const ping = faultPing("start_failed", new Error("x".repeat(400)), {
				visitorId: "v1",
			});
			expect(ping.error.length).toBe(256);
			expect(ping.url).toBe("");
		} finally {
			if (prior === undefined) delete globalThis.window;
			else globalThis.window = prior;
		}
	});

	test("faultPing honors an overridden provenance and a slice where the caller holds one", () => {
		const ping = withWindow("https://site.example/", () =>
			faultPing("terminated", new Error("boom"), {
				visitorId: "v1",
				sliceId: "s1",
				recorderVersion: "custom/9.9.9",
			}),
		);
		expect(ping.sliceId).toBe("s1");
		expect(ping.recorderVersion).toBe("custom/9.9.9");
	});

	test("gatedPing composes the gate attestation, page named by address", () => {
		const ping = withWindow("https://site.example/l?gclid=xyz", () =>
			gatedPing("bot", { visitorId: "v1" }),
		);
		expect(ping).toEqual({
			metric: "capture_gated",
			visitorId: "v1",
			recorderVersion: RECORDER_VERSION,
			reason: "bot",
			detail: "",
			url: "https://site.example/l",
		});
	});

	test("gatedPing carries the gate's own evidence for the verdict", () => {
		const ping = withWindow("https://site.example/l", () =>
			gatedPing("bot", { visitorId: "v1", detail: "detectWindowSize" }),
		);
		expect(ping.detail).toBe("detectWindowSize");
	});
});

describe("storeSink", () => {
	// The one home of the emit-side path shapes store/src/keys.js gates. These URLs are the
	// wire contract: a drift here is a green recorder against a worker that rejects every upload.
	test("closes the store's chunk and telemetry paths over origin and snippet id", async () => {
		const chunkCalls = [];
		const teleCalls = [];
		const { sink, telemetry } = storeSink({
			origin: "https://store.test",
			snippetId: "abc123",
			fetchFn: (u, o) => {
				chunkCalls.push([u, o]);
				return Promise.resolve({ ok: true });
			},
			carrier: transportCarrier({
				beaconFn: () => {
					throw new Error("telemetry must ride fetch pre-verdict");
				},
				fetchFn: (u, o) => {
					teleCalls.push([u, o]);
					return Promise.resolve({ ok: true });
				},
			}),
		});

		await sink.send(new Uint8Array([1]), {
			visitorId: "visitor-1",
			sliceId: "1749600000000-ab12",
			chunkKey: "0001749600000000001",
		});
		telemetry.emit({ metric: "slice_started", visitorId: "visitor-1" });

		expect(chunkCalls[0][0]).toBe(
			"https://store.test/chunks/abc123/visitor-1/1749600000000-ab12/0001749600000000001",
		);
		expect(teleCalls[0][0]).toBe(
			"https://store.test/telemetry/abc123/visitor-1",
		);
	});

	test("chunk and telemetry share one carrier — a blocked-beacon verdict routes both", () => {
		const timing = fakeTiming();
		try {
			const beacons = [],
				fetches = [];
			const carrier = transportCarrier({
				beaconFn: capturing(beacons),
				fetchFn: capturing(fetches, Promise.resolve({ ok: true })),
			});
			const { sink, telemetry } = storeSink({
				origin: "https://store.test",
				snippetId: "snip",
				carrier,
			});

			telemetry.probe({ metric: "transport_probe", visitorId: "v1" });
			timing.land(
				{
					name: "https://store.test/telemetry/snip/v1",
					initiatorType: "beacon",
					transferSize: 0,
				},
				{
					name: "https://store.test/telemetry/snip/v1",
					initiatorType: "fetch",
					transferSize: 300,
				},
			);

			const bytes = new Uint8Array([0x1f, 0x8b]);
			expect(
				sink.beacon(bytes, {
					visitorId: "v1",
					sliceId: "s1",
					chunkKey: "c1",
				}),
			).toBe(true);
			expect(beacons).toHaveLength(1); // the probe only
			expect(fetches).toHaveLength(1);
			expect(fetches[0][0]).toBe("https://store.test/chunks/snip/v1/s1/c1");
			expect(fetches[0][1].body).toBe(bytes);
		} finally {
			timing.restore();
		}
	});
});
