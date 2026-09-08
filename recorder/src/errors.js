/**
 * One line describing anything thrown or rejected, for the reports that ride the chunks' errors channel and the telemetry fault plane. IndexedDB surfaces null where an error object is expected (an aborted transaction's `error` is null, and a force-closed connection can fail requests with nothing at all), and a host page can throw any value — so the line is built from name and message where they exist and names what was thrown where they do not. String() alone would render a null rejection as "null".
 */
export function describeError(error) {
	if (error === null || error === undefined) {
		return `${error === null ? "null" : "undefined"} was thrown (no error object)`;
	}
	if (typeof error === "string") {
		return error.trim() === "" ? "an empty string was thrown" : error;
	}
	const name =
		typeof error.name === "string" && error.name !== "" ? error.name : null;
	const message =
		typeof error.message === "string" && error.message !== ""
			? error.message
			: null;
	if (name !== null && message !== null) return `${name}: ${message}`;
	if (name !== null || message !== null) return name ?? message;
	try {
		const json = JSON.stringify(error);
		if (typeof json === "string" && json !== "{}")
			return `non-error thrown: ${json}`;
	} catch {
		/* circular or hostile toJSON; the String below still answers */
	}
	return `non-error thrown: ${String(error)}`;
}
