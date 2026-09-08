import { expect, test } from "bun:test";
import { diffLines } from "./diff.js";

// The page is in the stream already — projected in full at each snapshot. A delta that re-printed the
// lines around a change would be spending the stream's budget on content the reader is already holding.
const carriesOnlyChangedLines = (diff) =>
	diff
		.split("\n")
		.every((line) => line.startsWith("- ") || line.startsWith("+ "));

test("identical text yields empty diff", () => {
	expect(diffLines("a\nb\nc", "a\nb\nc")).toBe("");
});

test("a changed line is the removal and the addition, and nothing else", () => {
	const diff = diffLines(
		"# Title\nold line\nfooter",
		"# Title\nnew line\nfooter",
	);
	expect(diff).toBe("- old line\n+ new line");
	expect(carriesOnlyChangedLines(diff)).toBe(true);
});

test("pure insertion", () => {
	expect(diffLines("a\nc", "a\nb\nc")).toBe("+ b");
});

test("pure deletion", () => {
	expect(diffLines("a\nb\nc", "a\nc")).toBe("- b");
});

test("distant changes carry no position, only the lines themselves", () => {
	const old = Array.from({ length: 40 }, (_, i) => `line ${i}`);
	const now = [...old];
	now[2] = "changed early";
	now[35] = "changed late";
	const diff = diffLines(old.join("\n"), now.join("\n"));
	expect(diff).toBe("- line 2\n+ changed early\n- line 35\n+ changed late");
	expect(carriesOnlyChangedLines(diff)).toBe(true);
});

test("interleaved edits stay literal — no summarizing", () => {
	const oldText = ["one", "two", "three", "four"].join("\n");
	const newText = ["one", "deux", "three", "quatre"].join("\n");
	const diff = diffLines(oldText, newText);
	expect(diff).toContain("- two");
	expect(diff).toContain("+ deux");
	expect(diff).toContain("- four");
	expect(diff).toContain("+ quatre");
	expect(diff).not.toContain("three");
	expect(carriesOnlyChangedLines(diff)).toBe(true);
});

// Anchor elision inside a changed line pair — the delta rule diff.js states.

const loneSurrogate =
	/(?:^|[^\ud800-\udbff])[\udc00-\udfff]|[\ud800-\udbff](?:$|[^\udc00-\udfff])/;

test("a small edit inside a huge line ships the changed span with anchors, not the line", () => {
	const pad = "lorem ipsum dolor sit amet ".repeat(3000);
	const diff = diffLines(
		`head\nvalue="${pad}START rest of doc"\nfoot`,
		`head\nvalue="${pad}FINISH rest of doc"\nfoot`,
	);
	expect(diff.length).toBeLessThan(400);
	expect(diff).toContain("START");
	expect(diff).toContain("FINISH");
	expect(diff).toContain("…");
	const lines = diff.split("\n");
	expect(lines.length).toBe(2);
	expect(lines[0].startsWith("- ")).toBe(true);
	expect(lines[1].startsWith("+ ")).toBe(true);
});

test("an append to the end of a huge line ships only the tail", () => {
	const pad = "x".repeat(50000);
	const diff = diffLines(`a\n${pad}`, `a\n${pad}MORE`);
	expect(diff.length).toBeLessThan(300);
	expect(diff).toContain("MORE");
	expect(diff).toContain("…");
});

test("a line changed in two places elides every shared stretch, keeping both changes", () => {
	const value = "lorem ipsum dolor sit amet ".repeat(1000);
	const diff = diffLines(
		`h\n[textarea placeholder="Es" value="${value}TAIL ONE"]\nf`,
		`h\n[textarea placeholder="Esc" value="${value}TAIL TWO"]\nf`,
	);
	expect(diff.length).toBeLessThan(400);
	expect(diff).toContain('placeholder="Es" value="lorem');
	expect(diff).toContain('placeholder="Esc" value="lorem');
	expect(diff).toContain("TAIL ONE");
	expect(diff).toContain("TAIL TWO");
	expect(diff.split("…").length - 1).toBe(2);
});

test("a shared stretch inside both lines keeps an anchor on each side of the elision", () => {
	const shared = "z".repeat(500);
	const diff = diffLines(`AAA${shared}BBB`, `CCC${shared}DDD`);
	expect(diff).toBe(
		`- AAA${"z".repeat(40)}…${"z".repeat(40)}BBB\n+ CCC${"z".repeat(40)}…${"z".repeat(40)}DDD`,
	);
});

test("scattered small changes print the line whole — no stretch reaches anchor scale", () => {
	const cells = Array.from({ length: 30 }, (_, i) => `cell ${i} value 100`);
	const oldLine = cells.join(" | ");
	const newLine = cells.map((c, i) => (i % 3 ? c : `${c}1`)).join(" | ");
	expect(diffLines(oldLine, newLine)).toBe(`- ${oldLine}\n+ ${newLine}`);
});

test("repeated content aligns to its next occurrence, not its first", () => {
	const row = "| Zosia Jane | [course](https://x.test/c) | delete |";
	const rows = (n) => Array.from({ length: n }, () => row).join(" ");
	const diff = diffLines(
		`${rows(4)} OLD MIDDLE ${rows(4)}`,
		`${rows(4)} NEW MIDDLE ${rows(4)}`,
	);
	const head = `${rows(4)} `.slice(-40);
	const tail = ` MIDDLE ${rows(4)}`.slice(0, 40);
	expect(diff).toBe(`- …${head}OLD${tail}…\n+ …${head}NEW${tail}…`);
});

test("short changed lines print whole — the trimmed form never fires below anchor scale", () => {
	expect(diffLines("# T\nold line\nf", "# T\nnew line\nf")).toBe(
		"- old line\n+ new line",
	);
});

test("pairing is positional within a replacement run; unpaired extras print whole", () => {
	const pad = "y".repeat(30000);
	const diff = diffLines(
		`${pad}AA${pad}\nsecond old`,
		`${pad}BB${pad}\nsecond new\nthird added`,
	);
	expect(diff.length).toBeLessThan(1000);
	expect(diff).toContain("AA");
	expect(diff).toContain("BB");
	expect(diff).toContain("- second old");
	expect(diff).toContain("+ second new");
	expect(diff).toContain("+ third added");
});

test("a wholesale replacement of a huge line states both values whole — the change IS the content", () => {
	const oldVal = "a".repeat(30000);
	const newVal = "b".repeat(30000);
	const diff = diffLines(`x\n${oldVal}\ny`, `x\n${newVal}\ny`);
	expect(diff).toContain(oldVal);
	expect(diff).toContain(newVal);
});

test("the elision boundary never splits a surrogate pair", () => {
	const pad = "\u{1F642}".repeat(20000);
	const diff = diffLines(`${pad}old tail`, `${pad}new tail`);
	expect(diff).toContain("…");
	for (const line of diff.split("\n")) {
		expect(loneSurrogate.test(line)).toBe(false);
	}
});

test("oversized inputs fall back to coarse replace, still literal", () => {
	const oldText = Array.from({ length: 2500 }, (_, i) => `old ${i}`).join("\n");
	const newText = Array.from({ length: 2500 }, (_, i) => `new ${i}`).join("\n");
	const diff = diffLines(oldText, newText);
	expect(diff).toContain("- old 0");
	expect(diff).toContain("+ new 2499");
	expect(carriesOnlyChangedLines(diff)).toBe(true);
});
