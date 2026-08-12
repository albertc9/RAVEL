# RAVEL project format

A RAVEL project is an hls4ml-style project directory containing a complete Aria
specialization. Vendor tools may or may not have run.

## Root records

- `hls4ml_config.yml` is the portable hls4ml configuration. `OutputDir` is `.`
  and the project-local Keras snapshot uses a relative path.
- `ravel_config.yml` is the normalized RAVEL configuration, including the
  resolved `Optimization` section. Its
  published output directory is also `.`.
- `ravel_manifest.json` is the immutable schema-v4 generation record.
- `ravel_qualification.json` is optional schema-v2 measured Vitis evidence.
- `build_opt.tcl` contains explicit Vitis stage booleans.

Published records contain no original source filename, username, hostname, or
generation-machine directory. Moving the complete project preserves source
integrity and the ability to open, refresh, link, and build it.

## Generation identities

- `source_artifact_sha256` identifies the project-local serialized model as a
  representation-provenance fact.
- `semantic_model_sha256` identifies canonical topology, parameters, learned
  quantizer state, and static quantization semantics.
- `configuration_sha256` identifies normalized generation-affecting settings.
- `implementation_sha256` identifies the resolved plan, passes, templates, and
  compatibility profile.
- `generation_fingerprint` combines the semantic, configuration, and
  implementation identities.
- `source_closure_sha256` identifies the bounded list of published source,
  parameter, model, configuration, simulation, and vendor-script files.

Output directory, verification selection, Vitis invocation, timestamps, and
qualification results do not participate in generation identity.
The versioned default policy and final temporal-packing/Dense-parallelism values
do participate. An architecture-preserving refresh reuses those recorded values.

## Status and qualification

Generation, dependency qualification, correctness verification, model fidelity,
source integrity, and performance qualification are independent. Full opening
or inspection recomputes the source closure; fast inspection skips payload
hashing and never claims the source is clean.

Qualification is separate so vendor execution cannot rewrite generation facts.
It binds the manifest hash, generation fingerprint, source-closure hash, top,
tool version, part, target clock, measured timing/performance/resources, RTL
ports, requested RTL CoSim status, and report-file hashes. When CoSim is
selected, a qualification record is not written unless the top-level Verilog
report says `Pass`; that report is included in the evidence hash closure.
Foreign or edited evidence is `stale`. Recorded measurements have no universal
performance pass/fail threshold.

## Parameter package schema

`.ravelparams` is a deterministic ZIP container with
`parameter_package.json` and ordered `arrays/*.npy` payloads. Schema version 2
defines:

- generation, model family, frontend provenance, and ModelGraph model facts;
- a canonical model-structure and numeric-contract identity;
- entries with operation/role ID, shape, dtype, numeric type, storage, and hash;
- `compatibility_sha256`, `parameter_state_sha256`, and
  `package_content_sha256` as separate identities.

Packages include only generation-relevant hardware state. They exclude layer
names, framework variable order, optimizer and training state, data, labels,
absolute paths, generated sources, reports, and application thresholds.
Applying a package replaces the clean ModelGraph payload in staging and runs the
normal generation pipeline; it never patches a published weight header in
place.

The JSON Schemas for configuration, manifest, qualification, and parameter
packages ship under `ravel_hls/schemas` in the installed distribution.
