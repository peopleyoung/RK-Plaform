# RKNN Multi-Context and Multi-Worker Pool Design

## Goal

Add end-to-end, task-configurable RKNN context and worker pools for primary and
secondary inference. Preserve existing single-context behavior by default,
share identical pools across tasks, and account for actual RKNN contexts when
checking node capacity.

## Scope

The change covers:

- frontend task configuration and task detail display;
- API contracts, persistence, desired-state payloads, and capacity validation;
- primary and secondary runtime grouping;
- generated runtime configuration;
- the YOLO and structured RKNN execution paths;
- host-side tests, image builds, documentation, and an explicitly approved
  board-side validation procedure.

The change does not add dynamic pool resizing to a running pipeline, a separate
queue-capacity setting, or per-context NPU core masks.

## Confirmed Decisions

- A primary task has independent `contextCount` and `workerCount` settings.
- Every secondary model rule has its own `contextCount` and `workerCount`.
- Both values default to `1` for backward compatibility.
- Both values must be positive integers and `workerCount` must not exceed
  `contextCount`.
- There is no additional per-task maximum. The node's `maxModelInstances`
  value limits the total number of actual RKNN contexts.
- Tasks with identical runtime parameters share one pool.
- Pool contexts inherit the task's existing `npuCoreMask` and `npuCorePolicy`.
- The first context is created with `rknn_init`; remaining contexts are created
  with `rknn_dup_context`.
- A pool-size change creates and activates a new immutable runtime revision. It
  does not resize the active process in place.

## Alternatives Considered

### Native C++ execution pool (selected)

Implement context leasing, the bounded job queue, and workers inside the C++
RKNN runtime. This supports frame-level scheduling, keeps a context exclusive to
one worker during an inference, and avoids duplicating video pipelines.

### Python-generated instance replicas

Generate several logical instances in the runtime adapter. This requires less
C++ work but only provides task-level or stream-level partitioning. It does not
provide a shared frame queue or a true worker scheduler for one logical pool.

### Multiple inference processes

Run one pipeline process per replica. This provides process isolation but
duplicates model and media resources and complicates revision activation,
health checks, and rollback.

## Configuration Contract

### Primary task

The task create, update, response, and desired-state contracts expose:

```json
{
  "contextCount": 1,
  "workerCount": 1
}
```

The persisted `inference_tasks` record stores `context_count` and
`worker_count`. Database startup adds missing columns with a default of `1`,
following the project's existing additive migration pattern.

### Secondary model

Each `analytics.secondaryModels` item exposes the same fields:

```json
{
  "releaseId": "release-id",
  "sourceClassIds": [0],
  "confidenceThreshold": 0.25,
  "contextCount": 1,
  "workerCount": 1
}
```

Old JSON without these fields normalizes to `1/1` in the frontend, API
validation, and runtime adapter.

### Generated runtime instance

The runtime adapter writes snake-case values into each generated instance:

```yaml
context_count: 2
worker_count: 2
queue_capacity: 8
```

The C++ runtime defaults missing counts to `1`. Queue capacity is derived as
`max(8, worker_count * 2)` and is not user-configurable in this change.

## Runtime Grouping and Capacity

A primary runtime-pool key contains:

- release ID;
- adapter;
- canonical thresholds;
- NPU core mask and policy;
- context count;
- worker count.

Tasks with the same key share an instance, queue, contexts, and workers. A
difference in either concurrency value creates a separate pool.

Secondary pools use the corresponding secondary release, confidence threshold,
inherited task NPU settings, context count, and worker count as their key.
Identical secondary configurations share a pool even when referenced by
different primary pipelines.

Capacity is the sum of `contextCount` once per unique primary pool plus
`contextCount` once per unique secondary pool. A deployment that exceeds the
node's `maxModelInstances` is rejected before desired state changes. The
conflict response reports required contexts and available node capacity.

Existing exclusive-core conflict checks continue to operate on logical pools.
All contexts inside a pool are owned by that pool and apply the same mask. A
pool with an exclusive mask conflicts with another overlapping pool, but its
own contexts are allowed to use the pool's assigned cores.

## C++ Execution Pool

A shared RKNN execution-pool component is used by both `RknnYoloInstance` and
`RknnStructuredInstance`. It owns:

- all context handles;
- a FIFO of available context handles;
- a bounded FIFO of inference jobs;
- `workerCount` worker threads;
- synchronization and deterministic shutdown state.

Initialization proceeds as follows:

1. Load model bytes and create the primary context with `rknn_init`.
2. Apply `rknn_set_core_mask` to the primary context.
3. Query and validate model tensors using the primary context.
4. Create the remaining contexts with `rknn_dup_context`.
5. Apply `rknn_set_core_mask` to every duplicate context.
6. Start workers only after every context is ready.

Any initialization failure destroys every context created for the new pool and
causes activation to fail. No partially initialized pool becomes visible.

Each worker removes one job from the shared queue and leases one available
context. The context remains exclusive until input submission, `rknn_run`,
output retrieval, decoding, and output release complete. An RAII lease returns
the context on every success, failure, and exception path. The worker then
fulfills the job promise and continues serving the pool.

When `contextCount` is greater than `workerCount`, the extra contexts remain in
the available set and rotate through leases. Actual simultaneous inference is
limited by `workerCount`.

Shutdown stops accepting commits, wakes workers, completes queued jobs with a
stopped error, joins all worker threads, and then destroys all contexts. No job
promise may remain unresolved.

## Frontend

The main task form adds two compact numeric stepper inputs to the existing NPU
scheduling section. Every secondary-model editor adds its own pair. Inputs show
defaults of `1`, accept integers greater than zero, and prevent submission when
workers exceed contexts.

Task details display the configured pool as `N contexts / M workers`. Helper
text explains that workers determine simultaneous inference while contexts
consume node model-instance capacity.

## Compatibility

- Existing database rows migrate to `1/1`.
- Existing API clients may omit both fields and receive `1/1` behavior.
- Existing secondary analytics JSON normalizes to `1/1`.
- New runtimes accept old generated YAML and default to one context and worker.
- Old runtimes ignore the additional desired-task fields and remain
  single-context. A new server therefore remains protocol-compatible, although
  multi-context behavior requires the new runtime image.

## Errors and Observability

- Invalid counts return HTTP 422 before persistence.
- Capacity exhaustion returns HTTP 409 with required and maximum context counts.
- A duplicate-context or core-mask failure aborts activation and preserves the
  previously active revision.
- A single inference failure fails that job but does not terminate its worker.
- Startup logs include instance name, context count, worker count, derived queue
  capacity, core mask, and core policy.
- Revision metadata records pool counts so operators can compare desired and
  active resource use.

## Test Strategy

### Backend and runtime-adapter tests

- default and explicit API serialization;
- rejection of non-positive counts and `workerCount > contextCount`;
- additive database migration with `1/1` defaults;
- primary grouping when counts match and splitting when they differ;
- secondary pool grouping;
- capacity accounting by actual primary and secondary contexts;
- exclusive-core validation at pool granularity;
- generated YAML and revision metadata.

### C++ tests

- multiple workers process all queued jobs;
- one context is never leased to two workers simultaneously;
- context leases are returned after processing failures;
- partial duplicate-context initialization is fully cleaned up;
- shutdown resolves queued jobs and joins all workers;
- both YOLO and structured instances parse defaults and explicit counts.

The scheduler tests use stub RKNN operations and do not require board hardware.

### Frontend checks

- analytics normalization supplies secondary `1/1` defaults;
- main and secondary forms serialize explicit counts;
- invalid relationships prevent submission;
- TypeScript checking and the production frontend build succeed.

### Build and board validation

Run the focused Python suites, the full relevant backend suite, the C++ stub
build/tests, the frontend build, and the RK3588 inference-image build.

Board validation requires explicit approval immediately before deployment. It
uses a new image tag and retains the previous image unless an operator explicitly
requests cleanup after final-image validation. Validate a `1/1` revision
first, then a multi-context/multi-worker revision, confirm context and worker
startup logs and concurrent progress, and verify failed activation rolls back.
Only switch the normal Compose service after these checks pass.

## Acceptance Criteria

- Omitted settings preserve current single-context behavior.
- A configured pool creates exactly N contexts and M workers.
- Identical primary and secondary configurations share their respective pools.
- Node capacity equals the sum of contexts in unique pools.
- No context is used concurrently by more than one worker.
- Invalid or partially initialized pools never replace the active revision.
- Host checks and the RK3588 image build pass.
- Board deployment is performed only after separate user approval.
