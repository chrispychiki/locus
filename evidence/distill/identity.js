// Each JavaScript file named on the command line as Bun's transpiler prints it — the same program without its comments or formatting — for the derivation identity (derive.py).
// By default the transpiler drops branches it judges dead and substitutes the reading process's NODE_ENV, so an edit inside a dropped branch would leave the identity unchanged, and the identity would vary with the caller's environment. Both are turned off.
//
//   bun distill/identity.js <file>...
//   stdout: {"<file>": "<transpiled code>", ...}
import { readFileSync } from "node:fs";

const transpiler = new Bun.Transpiler({
	loader: "js",
	deadCodeElimination: false,
	define: { "process.env.NODE_ENV": "process.env.NODE_ENV" },
});
const code = {};
for (const file of process.argv.slice(2)) {
	try {
		code[file] = transpiler.transformSync(readFileSync(file, "utf8"));
	} catch (error) {
		const at = error.position
			? `:${error.position.line}:${error.position.column}`
			: "";
		console.error(`${file}${at}: ${error.message}`);
		process.exit(1);
	}
}
console.log(JSON.stringify(code));
