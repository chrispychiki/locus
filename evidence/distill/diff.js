// The projection delta carried on the last event of a mutation run: the lines the page lost and the lines
// it gained, against the prior projection. Nothing else — the page itself is in the stream, projected in
// full at each snapshot, and that is what a delta is read against. So a diff never re-prints page content
// that already stands in the event stream, and never claims a position it cannot make good on: no line numbers, no
// surrounding context, no hunk headers. Literal throughout — never a claim about what meaningfully changed.
//
// The delta rule — unchanged content past a small anchor never re-prints — recurses one level into a
// changed line pair. A projection line carries an element's whole value, so a huge line (a textarea
// holding a pasted document) would otherwise re-print wholesale on its smallest change, once per
// mutation run. Within a replacement pair every stretch the two lines share beyond anchor scale is
// elided to an anchor either side of it, the elision marked "…" — a line changed in two places (an
// attribute and the tail of its value) keeps both changes and neither copy of what sits between. A
// stretch shorter than the anchors never elides, so a line of scattered small changes prints whole. Shared stretches
// are found by indexing one line's anchor-length windows and walking the other in order, extending each
// match as far as it runs — linear in the lines, and a repeated window resolves to its next occurrence
// past the last match, so repetitive content (a table row's identical cells) still aligns. A mismatch
// costs only less elision, never a wrong claim: both sides stay literal text of their own line.
import { safeCut } from "./text.js";

const MAX_DP_LINES = 2000;
// Just enough adjacent context to place the changed span against the projection the reader already
// holds; elision below anchor scale would save nothing.
const ANCHOR_CHARS = 40;
// A stretch shorter than the two anchors that would frame it saves nothing by eliding.
const RUN_MIN = 2 * ANCHOR_CHARS;

// The stretches oldLine and newLine share, each at least RUN_MIN long, as [oldStart, newStart, length]
// in order along both lines.
function sharedRuns(oldLine, newLine) {
	if (oldLine.length < RUN_MIN || newLine.length < RUN_MIN) return [];
	const windows = new Map();
	for (let i = 0; i + RUN_MIN <= oldLine.length; i++) {
		const key = oldLine.slice(i, i + RUN_MIN);
		const at = windows.get(key);
		if (at) at.push(i);
		else windows.set(key, [i]);
	}
	const runs = [];
	let j = 0;
	let oldNext = 0;
	while (j + RUN_MIN <= newLine.length) {
		const at = windows.get(newLine.slice(j, j + RUN_MIN));
		if (!at) {
			j++;
			continue;
		}
		let lo = 0,
			hi = at.length;
		while (lo < hi) {
			const mid = (lo + hi) >> 1;
			if (at[mid] < oldNext) lo = mid + 1;
			else hi = mid;
		}
		if (lo === at.length) {
			j++;
			continue;
		}
		const i = at[lo];
		let n = RUN_MIN;
		while (
			i + n < oldLine.length &&
			j + n < newLine.length &&
			oldLine[i + n] === newLine[j + n]
		)
			n++;
		runs.push([i, j, n]);
		oldNext = i + n;
		j += n;
	}
	return runs;
}

// A line with its shared stretches elided: a stretch at the line's start keeps only its trailing
// anchor, one at its end only its leading anchor, one inside an anchor on each side. Returns null
// when nothing would elide.
function elide(line, runs, startOf) {
	let out = "";
	let pos = 0;
	let elided = false;
	for (const run of runs) {
		const start = startOf(run);
		const end = start + run[2];
		const keepHead = start > 0 ? ANCHOR_CHARS : 0;
		const keepTail = end < line.length ? ANCHOR_CHARS : 0;
		if (run[2] <= keepHead + keepTail) continue;
		out += line.slice(pos, safeCut(line, start + keepHead));
		out += "…";
		pos = safeCut(line, end - keepTail);
		elided = true;
	}
	if (!elided) return null;
	return out + line.slice(pos);
}

function trimPair(oldLine, newLine) {
	const runs = sharedRuns(oldLine, newLine);
	if (!runs.length) return null;
	const trimmedOld = elide(oldLine, runs, (run) => run[0]);
	const trimmedNew = elide(newLine, runs, (run) => run[1]);
	if (trimmedOld === null && trimmedNew === null) return null;
	return [trimmedOld ?? oldLine, trimmedNew ?? newLine];
}

// Zip each removal run with the addition run that follows it and trim every pair; unpaired extras
// and pairs sharing nothing print whole. Positional pairing can mispair a reordered region, and the
// cost of that is only a whole-line print — never a wrong claim, since both forms are literal.
function trimReplacements(ops) {
	const out = [];
	let i = 0;
	while (i < ops.length) {
		if (ops[i][0] !== "-") {
			out.push(ops[i++]);
			continue;
		}
		const minus = [];
		while (i < ops.length && ops[i][0] === "-") minus.push(ops[i++][1]);
		const plus = [];
		while (i < ops.length && ops[i][0] === "+") plus.push(ops[i++][1]);
		const pairs = Math.min(minus.length, plus.length);
		for (let k = 0; k < pairs; k++) {
			const trimmed = trimPair(minus[k], plus[k]);
			if (trimmed) {
				out.push(["-", trimmed[0]], ["+", trimmed[1]]);
			} else {
				out.push(["-", minus[k]], ["+", plus[k]]);
			}
		}
		for (let k = pairs; k < minus.length; k++) out.push(["-", minus[k]]);
		for (let k = pairs; k < plus.length; k++) out.push(["+", plus[k]]);
	}
	return out;
}

export function diffLines(oldText, newText) {
	if (oldText === newText) return "";
	const a = oldText.split("\n");
	const b = newText.split("\n");

	let pre = 0;
	while (pre < a.length && pre < b.length && a[pre] === b[pre]) pre++;
	let suf = 0;
	while (
		suf < a.length - pre &&
		suf < b.length - pre &&
		a[a.length - 1 - suf] === b[b.length - 1 - suf]
	)
		suf++;

	const aMid = a.slice(pre, a.length - suf);
	const bMid = b.slice(pre, b.length - suf);

	// Past the DP bound the region is replaced wholesale — coarser, still literal.
	const ops = trimReplacements(
		aMid.length > MAX_DP_LINES || bMid.length > MAX_DP_LINES
			? [...aMid.map((l) => ["-", l]), ...bMid.map((l) => ["+", l])]
			: lcsOps(aMid, bMid),
	);
	if (!ops.length) return "";

	return ops.map(([op, line]) => `${op} ${line}`).join("\n");
}

function lcsOps(a, b) {
	const n = a.length,
		m = b.length;
	const dp = new Uint32Array((n + 1) * (m + 1));
	for (let i = n - 1; i >= 0; i--) {
		for (let j = m - 1; j >= 0; j--) {
			dp[i * (m + 1) + j] =
				a[i] === b[j]
					? dp[(i + 1) * (m + 1) + j + 1] + 1
					: Math.max(dp[(i + 1) * (m + 1) + j], dp[i * (m + 1) + j + 1]);
		}
	}
	const ops = [];
	let i = 0,
		j = 0;
	while (i < n && j < m) {
		if (a[i] === b[j]) {
			i++;
			j++;
		} else if (dp[(i + 1) * (m + 1) + j] >= dp[i * (m + 1) + j + 1])
			ops.push(["-", a[i++]]);
		else ops.push(["+", b[j++]]);
	}
	while (i < n) ops.push(["-", a[i++]]);
	while (j < m) ops.push(["+", b[j++]]);
	return ops;
}
