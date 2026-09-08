/**
 * The lifecycle rule this deployment owns on its bucket: its identity, the shape R2 stores it in, and the horizon read back out of a live rule set. Declared once and imported by provision, which converges it, and doctor, which asserts it, so the converger and the asserter can never look for different rules or encode the same horizon two different ways.
 *
 * The declaration states the horizon in days (deploy.config.toml); R2 states it in seconds. That conversion lives here and nowhere else.
 */
export const RULE_ID = "expire-chunks";

const SECONDS_PER_DAY = 86_400;

/** The rule as R2 stores it, for a horizon declared in days. */
export function expiryRule(days) {
	return {
		id: RULE_ID,
		enabled: true,
		conditions: {},
		deleteObjectsTransition: {
			condition: { type: "Age", maxAge: days * SECONDS_PER_DAY },
		},
	};
}

/** The horizon in days this deployment's rule states in a live rule set, or null where the rule is absent. The rule is found by its id, never by its horizon: someone else's rule that happens to expire at the same age is not ours. */
export function ruleDays(rules) {
	const maxAge = rules.find((rule) => rule.id === RULE_ID)
		?.deleteObjectsTransition?.condition?.maxAge;
	return maxAge == null ? null : maxAge / SECONDS_PER_DAY;
}
