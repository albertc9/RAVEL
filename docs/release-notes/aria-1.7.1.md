# Aria v1.7.1

Draft for the planned release. The wording below describes intended scope;
implementation and qualification are pending.

## What's New

Aria 1.7.1 introduces automatic optimization search for multi-convolution
streaming designs, exploring schedules and parallelism within the selected
P/D configuration.

- Derives convolution schedules and parallelism candidates from model geometry
  and data dependencies.
- Optimizes stream packing and stage throughput together to reduce whole-model
  bottlenecks.
- Records the selected implementation, search decisions, and validation evidence
  for reproducible conversion.

Performance comparisons will be added after paired qualification.
