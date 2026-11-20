// Single source of truth for the toolkit version.
//
// It lives in its own module so api.mjs can stamp exports with it without
// importing the package entry point (which imports api.mjs in turn).
export const VERSION = '0.1.0';
