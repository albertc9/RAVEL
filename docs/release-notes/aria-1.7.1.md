# Aria v1.7.1

Prepared release notes; this development work does not publish a release.

## What's New

Aria 1.7.1 adds automatic analytical optimization search for two-block streaming
models while preserving the selected P/D configuration.

- Derives window schedules and position parallelism from model geometry, with
  calibrated selection and optional per-core resource limits.
- Records candidates, estimates and selection reasons, and preserves recorded
  architectures during parameter refresh.
- Reduces the qualified F12 target's II from 844 to 124 at 5 ns, using additional
  LUT and FF. Existing single-block performance remains unchanged.

Normal conversion performs software search. Vendor synthesis and routed timing
remain separate, explicit qualification steps. See the
[scope](../design/aria-1.7.1-optimization-search.md) and
[measured qualification](../../references/qualification/aria_1_7_1_search/README.md)
for supported coverage and resource tradeoffs.
