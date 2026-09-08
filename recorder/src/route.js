/**
 * SPA route-awareness. rrweb emits no Meta/FullSnapshot on a client-side route change, so an SPA
 * navigation is otherwise indistinguishable from an in-page mutation burst. watchRoutes hooks every
 * channel a route change can arrive through — pushState, replaceState, popstate, hashchange — and
 * hands each real change to the caller's onRoute. The route is keyed by its identity (pageAddress:
 * origin + path + fragment, every query dropped — a hash-router's own included), so a query-only
 * rewrite (analytics, state sync) is not a navigation. The history methods belong to the host page,
 * so the wrapper calls the original first and never throws into the host's navigation; the returned
 * restore undoes the patch.
 */
import { pageAddress } from "./address.js";

/**
 * @param {object} hooks
 * @param {() => boolean} hooks.isActive       route changes are ignored once this reads false
 * @param {(href: string) => void} hooks.onRoute       one real route change — the caller decides what a navigation means
 * @param {(error: unknown) => void} hooks.onRouteError  a failing detector or onRoute; must not itself throw into the host
 * @returns {() => void} restore — unpatches history and removes the listeners
 */
export function watchRoutes({ isActive, onRoute, onRouteError }) {
	let lastRoute = pageAddress(window.location.href);
	const onUrlChange = () => {
		if (!isActive()) return;
		const href = window.location.href;
		const route = pageAddress(href);
		if (route === lastRoute) return;
		lastRoute = route;
		try {
			onRoute(href);
		} catch (error) {
			onRouteError(error);
		}
	};
	// Never throw into the host's navigation — and never swallow it either. A route detector that
	// has stopped working produces a recording that looks whole: every page after the first is
	// missing from an SPA session, and only the fault says so.
	const guarded = () => {
		try {
			onUrlChange();
		} catch (error) {
			try {
				onRouteError(error);
			} catch {
				/* the host's navigation is not ours to break, whatever else is broken */
			}
		}
	};
	const originalPushState = window.history.pushState;
	const originalReplaceState = window.history.replaceState;
	const wrappedPushState = function (...args) {
		const result = originalPushState.apply(this, args);
		guarded();
		return result;
	};
	const wrappedReplaceState = function (...args) {
		const result = originalReplaceState.apply(this, args);
		guarded();
		return result;
	};
	window.history.pushState = wrappedPushState;
	window.history.replaceState = wrappedReplaceState;
	window.addEventListener("popstate", onUrlChange);
	window.addEventListener("hashchange", onUrlChange);
	return () => {
		if (window.history.pushState === wrappedPushState) {
			window.history.pushState = originalPushState;
		}
		if (window.history.replaceState === wrappedReplaceState) {
			window.history.replaceState = originalReplaceState;
		}
		window.removeEventListener("popstate", onUrlChange);
		window.removeEventListener("hashchange", onUrlChange);
	};
}
