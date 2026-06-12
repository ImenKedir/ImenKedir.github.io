// Single-page app: no SSR, everything renders client-side.
// Routes are still prerendered as empty shells so they get real 200 responses.
export const ssr = false;
export const prerender = true;
