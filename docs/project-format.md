# RAVEL project format

A RAVEL project is an hls4ml-style project directory whose generated source and
configuration already contain the selected RAVEL specialization. Vendor tools
have not necessarily run.

## Configuration and evidence

Three root files have separate ownership:

- `hls4ml_config.yml` contains hls4ml-owned baseline and downstream-tool
  configuration.
- `ravel_config.yml` contains only RAVEL-owned user intent plus hashes linking
  it to the source model and hls4ml configuration.
- `ravel_manifest.json` is the immutable generation record.

The schema-version-1 manifest has stable sections for RAVEL identity, source
model, dependencies, normalized configuration, profile, implementation plan,
pipeline, interfaces, verification, managed files, and generation identity.
Paths are relative POSIX paths; local usernames, hostnames, absolute input
paths, arbitrary environment variables, and complete shell commands are not
recorded. JSON Schema documents for the RAVEL configuration, generation
manifest, and qualification record ship in the installed `ravel_hls/schemas`
package data.

## Fingerprints

SHA-256 identities remain distinct:

- `source_artifact_sha256` identifies the exact project-local serialized model.
- `semantic_model_sha256` identifies canonical topology, parameters, and
  quantization without paths, archive metadata, or non-semantic names.
- `configuration_sha256` identifies normalized generation-affecting settings.
- `implementation_sha256` identifies the resolved plan, pass versions,
  template profile, and compatibility profile.
- managed-file hashes identify each RAVEL-owned generated file.
- `generation_fingerprint` combines normalized inputs that affect generated
  results.

Timestamps and qualification results do not participate in generation
identity. A separate qualification record links to the SHA-256 of the complete
immutable manifest.

## Independent status axes

RAVEL reports generation, dependency qualification, correctness verification,
model fidelity, source integrity, and performance qualification independently.
A staged failure is not published as a failed project.

Opening a project recomputes managed-file integrity without modifying the
manifest. Manual changes produce an in-memory `modified` status and make prior
correctness evidence stale. Vendor measurements are stored separately in
`ravel_qualification.json` and never rewrite generation-time facts.

## Interface contracts

The manifest distinguishes:

- `logical_model_interface`: tensor shapes and numeric semantics;
- `hls_stream_interface`: packing, ordering, word counts, handshake, and
  source-level control;
- `rtl_interface.expected`: only rules derived from a qualified tool profile;
- `rtl_interface.measured`: only facts read from an actual vendor result.

For the canonical reference, the HLS stream carries two chronological rows and
four lanes per input word, producing 128 input words per inference. The current
wrapper expects eight 9-bit values in eight 16-bit RTL slots for 128-bit input
TDATA, and one 22-bit result in 32-bit output TDATA. Those RTL widths are a
reference-configuration contract, not a universal inference from C++ payload
widths.

## Replacement behavior

Generated projects are reproducible derived artifacts. Reusing the path of a
recognized RAVEL project fully regenerates, validates, and atomically replaces
it even when its fingerprint is unchanged. Corrupt or non-RAVEL targets require
an explicit force-replacement operation. Failure during staging preserves the
previous published directory.
