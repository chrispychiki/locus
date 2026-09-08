/**
 * The snippet id — a site's id within the deployment, and the first segment of every chunk and telemetry key.
 * It must match the store worker's path gate (store/src/keys.js SNIPPET_ID); an id the gate
 * refuses makes every upload 404 and captures nothing, silently, so the autostart facade
 * validates it before start().
 */
export const SNIPPET_ID_PATTERN = /^[a-z0-9]{3,64}$/;

export function isValidSnippetId(id) {
	return typeof id === "string" && SNIPPET_ID_PATTERN.test(id);
}
