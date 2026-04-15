# Realtime Online Serving Status

## Goal

The current goal of realtime online serving is to keep the `trace_replay` control plane semantics, but execute hybrid PIM work through real Ramulator callbacks and timing.

In practical terms, the target is:

- `trace_replay` remains the analytic reference for routing, batching, admission, and prediction semantics
- `ramulator2/src/frontend/impl/serving` implements the same serving logic on top of a real memory-system execution path
- hybrid decode can be driven either by:
  - template replay
  - runtime-generated commands

---

## What Is Implemented So Far

### 1. Realtime serving frontend and runtime

The active frontend is:

- `ramulator2/src/frontend/impl/serving/realtime_serving_frontend.cpp`

This frontend now reads a full serving configuration from YAML and forwards it into:

- `ramulator2/src/frontend/impl/serving/runtime.cpp`
- `ramulator2/src/frontend/impl/serving/runtime.h`

The runtime is responsible for:

- reading request CSVs
- loading cost tables
- route selection
- batching
- local scheduling
- predictive admission / SLO-guarded admission
- decode rebind
- hybrid PIM execution
- reporting and plotting outputs

So the old “single-request PIM replay helper” has become a real serving runtime with policy and execution state.

---

### 2. Control-plane parity with `trace_replay`

The realtime frontend now supports the same main serving-policy surface as `src/trace_replay`:

- route policies:
  - `min_finish`
  - `slack_then_finish`
  - `latency_guarded_energy`
- local scheduling policies:
  - `priority`
  - `fcfs_strict`
  - `prefill_priority_fcfs_decode`
- decode prediction modes:
  - `shadow`
  - `legacy`
  - `incremental`
  - `heuristic_refresh`
- batching controls:
  - prefill batching
  - decode batching
  - batch-size caps
  - prefill guard
  - max consecutive decode batches
- admission controls:
  - predictive admission
  - SLO-guarded admission
  - retry interval
  - harmed-request budgets
  - bypass / max wait controls
  - own-SLO checking toggle
- decode rebind:
  - hybrid to `gpu_only` after prefill when allowed and beneficial

Important semantic point:

- by default, `admission_check_own_slo = false`

That means:

- the candidate request is not rejected purely because its own SLO would be missed
- existing running requests are still protected from extra harm before admitting the new request

This matches the behavior we wanted from the replay simulator.

---

### 3. Two hybrid execution modes

The serving frontend currently supports two hybrid data-plane modes:

#### `template_replay`

This is the legacy path:

- load `lin_<L>.trace`
- parse it
- replay those commands through Ramulator

This path is still kept as the conservative baseline and validation reference.

#### `runtime_generated`

This is the new path:

- do not depend on offline `lin_<L>.trace`
- generate the hybrid command stream in C++
- encode generated KV accesses as virtual addresses
- translate them to physical addresses at runtime
- send the translated commands into Ramulator

This is the main architectural shift we wanted.

---

### 4. Runtime-generated hybrid execution

The runtime-generated path is implemented mainly in:

- `ramulator2/src/frontend/impl/serving/command_generator.h`
- `ramulator2/src/frontend/impl/serving/command_generator.cpp`
- `ramulator2/src/frontend/impl/serving/layout_planner.h`
- `ramulator2/src/frontend/impl/serving/layout_planner.cpp`
- `ramulator2/src/frontend/impl/serving/pim_executor.cpp`

What this path does now:

- ports the LPDDR5-bank hybrid command schedule into C++
- generates logical commands incrementally
- distinguishes:
  - passthrough commands like barriers / softmax controls
  - KV-backed commands targeting key/value memory
- uses a dense logical KV layout instead of embedding sparse physical offsets directly into the generator

The current generator still models one representative attention layer per generated PIM job. That is deliberate for now because it preserves parity with the earlier trace-driven setup while the serving runtime is being stabilized.

---

### 5. Real virtual-address path

The current `runtime_generated` path is now a real VA path, not just a physical-address bridge.

The main pieces are:

- `VirtualAddressEncoder`
- `AttAccServingTranslation`
- `ServingAddressSpace`

The flow is:

1. the command generator emits a logical KV access
2. `VirtualAddressEncoder` converts that logical access into a structured virtual address
3. the frontend sends the request through `AttAccServingTranslation`
4. translation resolves the VA to a physical address using shared allocator state
5. the translated request is submitted to Ramulator

This means the active runtime-generated path now has:

- generator-side logical addresses
- VA encoding
- runtime translation
- allocator-backed object existence checks

That was a major cutover point.

---

### 6. Shared allocator and memory accounting

The current allocator pieces are:

- `ramulator2/src/frontend/impl/serving/allocator.h`
- `ramulator2/src/frontend/impl/serving/allocator.cpp`
- `ramulator2/src/translation/impl/attacc_serving_translation.cpp`

What they do now:

- maintain a shared PIM-visible address space
- split memory into deterministic 4KB region pools
- support regions for:
  - KV key
  - KV value
  - score
  - context
  - scratch
- reserve and release request-level KV allocations
- reserve and release decode-task transient allocations
- expose allocator stats for reporting

Current allocation policy:

- persistent KV is reserved when a request is admitted on the hybrid route
- transient task memory is reserved when a hybrid decode task starts
- if hybrid memory cannot be reserved:
  - try `gpu_only`
  - otherwise hold through the existing admission queue if enabled
  - otherwise drop

So we already have the first usable version of:

- runtime memory checking
- capacity-aware fallback
- allocator-backed translation

---

### 7. Reporting, plotting, and run isolation

The realtime serving path now writes:

- per-request CSV
- summary JSON
- YAML config used for the run

Plotting is supported with:

- `tools/plot_realtime_serving_results.py`
- `tools/plot_replay_results.py`

We also added:

- `tools/run_realtime_online_serving_case.sh`

This script:

- creates a fresh timestamped run folder for each experiment
- stores:
  - `config.yaml`
  - `summary.json`
  - `requests_out.csv`
  - `run.log`
  - `run_command.sh`
  - `run_info.txt`
  - `plots/`

So each run is now isolated instead of writing all plots into one shared directory.

---

## What Has Been Validated

The following have already been checked:

- control-plane behavior works on realtime serving runs
- template replay and `runtime_generated` match on small parity cases
- parity was checked for representative `Lin` values such as:
  - `128`
  - `512`
  - `867`
- the matched outputs include:
  - makespan
  - TTFT
  - E2E
  - TBT
  - PIM command counters
  - real PIM job counts
  - memory-system cycle totals
- allocator-pressure fallback works:
  - when hybrid KV reservation fails, the request can fall back to `gpu_only`
- predicted-vs-actual finish export is now sane
  - the remaining mismatch is only a tiny callback-backed timing residual, not a plotting bug

So the current system is not just a design sketch. It is an operational realtime serving frontend with a working runtime-generated hybrid path.

---

## Current Limits

The implementation is still intentionally conservative in a few places.

### 1. Physical placement still preserves legacy parity

Even though `runtime_generated` now uses a real VA path, translation is still designed to reproduce the same effective physical addresses as the previous legacy path for the representative-layer accesses that are issued.

That means:

- we have real VA to PA translation
- but we are still preserving legacy physical placement behavior for parity and safety

So this is not yet the final “new physical allocation policy” step.

### 2. Representative-layer semantics are still in place

The generated hybrid command stream still models one representative attention layer per decode job.

Allocator capacity, however, is already modeled at full request scale:

- `num_layers * tokens * heads_per_hbm * dhead`

So execution semantics and capacity semantics are not yet equally detailed.

### 3. `score/context/scratch` are allocator-managed, but not fully active in command addressing

These regions are tracked for capacity and task accounting, but the current generated command stream is still primarily about KV-visible accesses.

That means:

- memory accounting is broader than active translated command coverage

### 4. Allocator policy is still simple

The allocator currently uses:

- deterministic 4KB pages
- fixed region pools
- no huge pages
- no fragmentation modeling
- no migration
- no compaction
- no eviction

This is the right first implementation, but it is not yet an OS-like or THP-like allocator.

### 5. The first backend is intentionally narrow

The runtime-generated path is currently targeted at:

- `lpddr5_pim_bank`

This is deliberate. It keeps the first runtime-generated serving backend narrow enough to validate thoroughly before generalizing.

---

## Current Execution Flow

The current realtime serving flow is:

1. read request CSV and serving config
2. load cost tables for route selection
3. route each request using the same serving control-plane logic as `trace_replay`
4. if the request is `gpu_only`, run the analytic GPU path
5. if the request is hybrid:
   - reserve hybrid memory
   - build or reuse generated artifacts for the request `Lin`
   - generate logical PIM commands
   - encode KV commands as virtual addresses
   - translate them through `AttAccServingTranslation`
   - send translated requests into Ramulator
   - wait for callback-backed completion
6. collect per-request metrics and summary statistics
7. write plots and run artifacts

So we now have the control plane of the replay simulator combined with a real callback-backed PIM execution path.

---

## Relationship To `trace_replay`

It is important to be precise here:

- `trace_replay` is still the analytic reference model
- realtime serving is the Ramulator-backed execution version

Today:

- route choice logic is intended to mirror `trace_replay`
- batching / admission / scheduling semantics are intended to mirror `trace_replay`
- the realtime difference is in the data plane:
  - hybrid decode actually executes through Ramulator timing and callbacks

So realtime serving is no longer just a trace parser. It is the serving-policy runtime plus a real PIM-backed execution path.

---

## Next Step

The next step is not “make the generator exist”; that part is already done.

The next real step is:

## replace legacy-parity physical placement with a true serving physical allocation policy

Concretely, that means:

### 1. Define the target physical placement policy

We need to choose how physical pages should be assigned for:

- KV key
- KV value
- score
- context
- scratch

This should be request-aware and placement-aware, not just parity-preserving.

The placement policy should explicitly answer:

- how to stripe across channels
- how to preserve BG/BA parallelism
- how to separate persistent and transient regions
- whether different objects should use different pools or striping rules

### 2. Change translation from parity-preserving mapping to policy-driven mapping

Today translation is effectively:

- “map VA pages so the resulting PAs match the old representative-layer physical pattern”

The next step should be:

- “map VA pages according to the serving allocator’s real placement policy”

That is the point where the system becomes a true runtime allocation design rather than a compatibility-preserving bridge.

### 3. Decide whether transient regions should become actively addressed

Right now `score/context/scratch` are tracked by the allocator but not yet first-class translated command targets in the active command stream.

We should decide whether the next step includes:

- only a better KV placement policy

or

- extending generated commands to use translated transient addresses too

The cleaner design is eventually to make these regions first-class addressable objects as well.

### 4. Only after that, move to advanced page policies

THP-style behavior should be later, not next.

That means:

- huge pages
- fragmented frames
- 4KB fallback
- compaction / migration

should come after the placement-policy cutover is stable.

---

## Recommended Immediate Work Item

The next concrete engineering task should be:

1. formalize the serving physical placement policy for KV and transient pools
2. implement that policy inside `AttAccServingTranslation` and the shared allocator state
3. keep the current VA path and control plane unchanged
4. validate the new placement on:
   - single-request parity-style runs
   - synchronized batched runs
   - Azure trace runs

That is the clean next increment because it changes exactly one layer:

- the VA to PA placement policy

without reopening:

- route choice
- batching
- admission
- prediction
- plotting

---

## Short Status

So far, realtime online serving has progressed from:

- trace replay of pre-generated PIM templates

to:

- a serving runtime with `trace_replay`-style control-plane features
- real callback-backed Ramulator execution
- runtime-generated hybrid commands
- a real VA path
- allocator-backed translation and capacity checks

The next step is to make physical placement policy truly runtime-native instead of parity-preserving.
