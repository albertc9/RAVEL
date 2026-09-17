# Aria v1.7.2

Prepared release notes; this development work does not publish a release.

## What's New

Aria 1.7.2 extends automatic optimization across both convolution stages,
reusing proven constant-arithmetic optimizations to improve throughput and LUT use.

- Adds an optional `TargetII` setting that prefers lower LUT use once the
  requested interval is met, with explicit reporting when it is not.
- Shares constant arithmetic and DSP/LUT realization capabilities across stages,
  while preserving fixed-point results and existing P/D interfaces.
- Removes expensive remainder operations from downstream window scheduling and
  searches finite arithmetic reuse without changing stream interfaces.
- Rebuilds coefficient-specific arithmetic during parameter refresh while
  preserving the recorded architecture and legacy single-convolution behavior.

See the [design and qualification boundaries](../design/aria-1.7.2-optimization.md).


The qualified F12 reference with `TargetII=85` achieves II 63 and 12.89% LUT,
with all core resource shares below 20% and routed timing passing at 200 MHz.
C verification, RTL CoSim and protocol checks pass; legacy firmware is preserved.
