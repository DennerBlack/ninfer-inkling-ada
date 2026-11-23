# Contributing to Inkling Ada

This fork focuses on Inkling 2 Mini inference on NVIDIA Ada GPUs. The [README](README.md)
covers the build and launch settings. The [laptop benchmark](docs/inkling-ada-laptop.md)
shows how to reproduce the long-context tests.

## Bug reports and benchmarks

For a bug, include the commit or release, GPU, driver, CUDA version, model artifact and
runtime flags. Add the smallest request or command that reproduces it and describe what
you expected to happen.

For a performance change, include a baseline and candidate measured on the same machine,
weights and prompts. Report prefill and generation separately. For long-context runs,
include the actual input length, KV format and time to first token. Record peak memory or
the GPU expert cache size when the change affects it. Include a correctness check and
report regressions alongside speedups.

If the same problem reproduces on NInfer-all, include that reproduction when reporting it
upstream. [NInfer-all contribution guide](https://github.com/iamwavecut/ninfer-all/blob/master/CONTRIBUTING.md)

## Pull request scope

Target [`inkling-ada`](https://github.com/DennerBlack/ninfer-inkling-ada/tree/inkling-ada)
in this fork. Keep each pull request focused on one fix or improvement. Use the PR
description to discuss scope for larger changes and link any related discussion. Include
the tests, measurements or documentation that explain the result.

When porting work from elsewhere, preserve the original author and reference the source
commit. Existing credits are in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## Engineering rules

These rules apply to the paths affected by a change. Include the coverage you verified and any
relevant gaps in the pull request.

### Lossless by default

An optimization must not change model output. Exact transforms, codecs and expert placements stay
exact: device, host and disk residency and the CPU share of expert products give the same results.
Floating-point changes stay within the oracle criterion of each Op
([Op development](docs/maintainer/op-development.md)).

A change that trades accuracy for speed is lossy. Examples include skipping experts or tokens,
approximate attention or sampling, early exit, and reducing storage or accumulation precision.
A lossy change needs the maintainer's agreement. It ships as an opt-in mode. Report the quality
cost (perplexity, task accuracy) next to the speed. Formats that the user selects, such as
`--kv-dtype` and quantized artifacts, are options of this kind.

After a change to expert residency, slot pools, prefetch or the expert cache, `ninfer-perplexity`
gives the same value on the same text with `--expert-residency host` and with `disk`. Time a prompt
of 5K tokens or more as well as short ones. On long prompts, compare greedy token IDs and logprobs
with the target branch because perplexity covers only some prompt paths.

### Every family

A change to shared code must preserve every model family that uses it: Qwen3.5-core
(dense and MoE releases), Qwen3.8-Flash-Next, Gemma 4, Flint, DeepSeek-V4.1 and Inkling 2. Shared
mechanisms include KV formats, rotary scaling, prompt chunking, CUDA Graphs, concurrency,
speculative decoding, sampling, expert residency and the CPU share, pipeline stages, tensor
parallelism, serving features, and kernels that several families call.

- Write the change once where the families share code. A speedup that one checkpoint, shape or
  family gets while the others could use it is incomplete.
- A dense change names its MoE counterpart and covers it in the same pull request, and the reverse.
- A family that does not get the change is named in the pull request and in its guide, with the
  reason.
- A new family brings the shared features of the other families: the `--kv-dtype` formats it can
  store, concurrency above one, CUDA Graphs, 2048-token prompt chunks, grouped Q/K/V projections,
  `--tensor-parallel`, YaRN, and the serving features in [docs/serving.md](docs/serving.md).

### Every platform

Keep shared code compatible with the inherited targets `sm_70`, `sm_75`, `sm_80`, `sm_86`,
`sm_89` and `120a`, on Linux with GCC and Windows with MSVC. CUDA 12.x is required for
`sm_70`. The `120a` target requires CUDA 13.1 or newer and excludes 13.2.0 and 13.2.1.
[Build system](docs/maintainer/build-system.md) · [Windows build](docs/build-windows.md)

- A kernel that needs newer instructions or more shared memory is gated by compute capability, and
  the other targets keep a working route. State which GPUs and platforms you tested.
- Routes and tuning constants come from measured [device profiles](docs/device-profiles.md). A
  constant from upstream, another fork or another card is a hypothesis until it is measured on each
  card that it affects.
- Add a profile for a GPU model that has no built-in one: calibrate it at its default power limit
  ([Calibrating by hand](docs/device-profiles.md#calibrating-by-hand)) and add its row to
  [docs/gpus.md](docs/gpus.md).
- Memory budgets and defaults follow the free memory of the card and the host. They do not assume
  one card size: check the smallest card that the change affects.

### Best default

Users get the best configuration without flags. The default is the fastest lossless configuration
measured on the affected families, card classes and KV formats.

- A lossless change that wins on all of them becomes the default in its pull request, with the
  measurements.
- A change that wins on some and loses on others selects itself from measured conditions (device
  profile, request shape, free memory), or stays behind an option until it can.
- A lossy mode is never the default.

### Comparable measurements

- A speed claim covers every affected family and card class, and every KV format that each family
  supports, with at least one point each. One short run per configuration is enough when it
  separates the arms. Report the worst change and every regression next to the best case.
- Both arms of an A/B run on the same host with the same build options, artifact and inputs: the
  same prompt token ids, a unique nonce in each prompt so that the prefix cache cannot reuse a
  shared prefix, interleaved runs, and no other CPU or GPU load on the host.
- A comparison with another engine uses that engine's latest release with its fastest documented
  configuration, on the same host and weights. Record its version and each flag with the reason for
  it, and show its lossy modes as separate rows. A perplexity comparison with llama.cpp uses the
  same windows and KV type ([Perplexity](docs/perplexity.md)).

### Documentation describes the current state

- Documents describe what the project is and does now. They contain no dated logs, run or CI
  histories, retry stories, accounts of what was tried, author host paths, server setups or copies
  of upstream pages. Measurements are reference tables with their conditions. A rejected approach
  gets at most one sentence, where it explains a current choice. Credits go to
  [CONTRIBUTORS.md](CONTRIBUTORS.md); git history keeps the rest.
- A user-visible change updates, in the same pull request, the family guide,
  [docs/cli.md](docs/cli.md) and [docs/serving.md](docs/serving.md) where they apply, the
  executable's `--help`, the README Artifacts table for a new artifact, and What's New when the
  change is a novelty.
- README's What's New lists novelties only. A novelty gives users something they did not have
  before: a new model, family, artifact, platform, option, endpoint, format or mode, or a measured
  speedup of a path that worked. A fix gets no item, whatever its size. A fix repairs a crash, wrong
  output, a failed start, a missed case or a regression; a speedup that wins back a slowdown and an
  option or default that a fix brings along are part of the fix. Guards, refusals, diagnostics,
  internal changes, CI, build, tests, tooling and documentation get no item either. What's New is
  not a chronicle of the day's work: a date without a novelty gets no line.
- Each date with a novelty gets one short line. Each item names the novelty in a few words and
  links inline to the document with its details. Descriptions, mechanisms, measurements and
  hardware belong in the linked documents, not in What's New. A later novelty of the same date adds
  an item to that line or extends an existing item.

### Tests and tooling

- A new kernel gets an independent FP32 or FP64 oracle test at real model shapes and route
  boundaries, for every schedule it can take, and one `compute-sanitizer` racecheck and synccheck
  run ([Op development](docs/maintainer/op-development.md)).
- Tests follow [tests/README.md](tests/README.md). A test that needs a real artifact is skipped when
  the artifact is absent.
- Product modes and tuning are command-line or serve options, or device-profile entries.
  Environment variables only force routes for diagnosis and A/B runs; a mode that users need is
  never an environment variable.
- Commit subjects follow Conventional Commits with lowercase types such as `feat`, `fix`, `perf`,
  `bench`, `test`, `build`, `refactor`, `docs` or `chore`.

## Engineering and verification

Read the active documentation governing the code being changed and preserve its ownership and
semantic contracts. In particular:

- validate exact transformations and formats with exact comparison;
- validate floating-point work against an independent mathematical oracle with a justified
  numerical criterion;
- verify the complete affected transition for stateful behavior;
- exercise relevant supported shapes and execution routes for CUDA changes;
- measure performance at the level being claimed; and
- keep externally observable implementation, schema tests, and documentation consistent.

Use the existing project workflows:

- [`tests/README.md`](tests/README.md) for test organization and commands;
- [`bench/README.md`](bench/README.md) for product and operator benchmarks; and
- [`docs/maintainer/op-development.md`](docs/maintainer/op-development.md) for numerical Op
  development and qualification.

Run the focused checks needed to establish the changed behavior and its material claims. Record the
exact commands and summarize the results. If a relevant check could not be run, state that
limitation and its consequence. Do not claim validation that was not performed.

## Contributor responsibility and AI assistance

The contributor is responsible for every submitted change, regardless of which tools were used to
produce it. Tool choice neither qualifies nor disqualifies a contribution.

The contributor must understand the complete diff, explain the design and its boundary conditions,
review generated output, perform the stated verification, and respond to technical review. Code
that the contributor cannot explain, validate, or maintain is not ready for submission. Maintainer
review is not a substitute for contributor debugging or engineering supervision.

## Pull request description

Use the [PR template](.github/pull_request_template.md) to describe the problem, implementation,
and verification.

A pull request description must include:

- the scope and any related discussion;
- the concrete design and why it fits the current architecture;
- the affected behavior, ownership boundary, or public contract;
- the exact verification commands and summarized results;
- the workload, hardware, toolchain, and methodology for any performance claim; and
- every relevant check that was not run and the resulting limitation.

Choose the verification details relevant to your change:

| Change | Evidence to include |
|---|---|
| Bug fix | The trigger, expected behavior, and verification of the fix. |
| Recipe or conversion method | The source, target representation, conversion command, and which stages were actually verified: conversion, loading, or execution. |
| New Op or shape | Supported inputs, independent numerical verification, and relevant boundaries. Follow [Op development](docs/maintainer/op-development.md). |
| Performance tuning | Correctness evidence and measurements at the claimed scope on the cards it affects: an Op benchmark for an Op claim, a whole request for an end-to-end claim. A route or schedule change states which [device profiles](docs/device-profiles.md) it changes. |
| Runtime or interface change | Reproduction and verification of the affected behavior. |
| Documentation, build, or tooling | Checks relevant to the change, such as links, affected build targets, or tool invocations. |

## Review policy

Review is a limited maintainer resource and is not guaranteed. A pull request may be closed without
a line-by-line review when it:

- proposes a broad change without establishing its problem or scope;
- falls outside the supported product or expands an unapproved contract;
- combines independent changes into an unreviewable implementation;
- contains evident correctness, ownership, lifetime, state, or boundary problems;
- lacks evidence appropriate to its correctness, numerical, or performance claims;
- cannot be explained and supported by its contributor; or
- would require the maintainer to redesign, split, debug, or complete its core implementation.

Closing a pull request does not necessarily reject the underlying report or idea. It means that the
submitted implementation does not meet the project's review and maintenance threshold. The
maintainer may address a confirmed problem through a separate implementation and reference the
original report where appropriate.
