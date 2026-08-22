# RKNN Multi-Context and Multi-Worker Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add end-to-end configurable RKNN context and worker pools for primary and secondary inference.

**Architecture:** Persist pool sizes on tasks and secondary rules, group equivalent runtime pools in one canonical planner, and charge node capacity by actual contexts. Generate immutable instance YAML and execute each instance through a reusable C++ context-leasing worker pool; retain `1/1` defaults at every compatibility boundary.

**Tech Stack:** FastAPI/Pydantic, SQLAlchemy/SQLite, pytest, Python runtime adapter, C++17/RKNN Runtime, React/TypeScript/Vite, Docker Compose.

---

## File Map

- `backend/platform_api/contracts.py`: public validation and camel-case API fields.
- `backend/platform_api/db_models.py`: persisted primary task pool sizes.
- `backend/platform_api/database.py`: additive migration for existing databases.
- `backend/platform_api/inference_service.py`: round trip, desired state, pool grouping, capacity and core conflicts.
- `workers/inference_agent/runtime_adapter.py`: desired-state validation, primary/secondary grouping, generated YAML and metadata.
- `third_party/nv_video_pipeline/src/rknn_instance/RknnExecutionPool.{h,cpp}`: context ownership, leasing, workers and shutdown.
- `third_party/nv_video_pipeline/src/rknn_instance/RknnYoloInstance.{h,cpp}`: YOLO context creation and pool callback.
- `third_party/nv_video_pipeline/src/rknn_instance/RknnStructuredInstance.{h,cpp}`: structured context creation and pool callback.
- `third_party/nv_video_pipeline/tests/RknnExecutionPoolTest.cpp`: hardware-free scheduler tests with fake handles.
- `src/types.ts`, `src/api/client.ts`, `src/pages/InferencePage.tsx`: primary task contract and UI.
- `src/components/InferenceBusinessFields.tsx`: secondary rule contract, validation and controls.
- `tests/test_database_migrations.py`, `tests/test_inference_api.py`, `tests/test_inference_runtime_adapter.py`: cross-layer regressions.
- `deploy/rk3588/runtime-adapter/README.md` and offline examples: operator contract.

### Task 1: Primary API and persistence contract

- [ ] Add tests proving omitted fields return `1/1`, explicit values round-trip,
  zero values fail, and workers cannot exceed contexts in
  `tests/test_inference_api.py`.
- [ ] Add a legacy `inference_tasks` table fixture to
  `tests/test_database_migrations.py`; assert `create_schema()` adds both
  columns with value `1` for the seeded row.
- [ ] Run the focused tests and verify they fail because the fields/columns do
  not exist:

```bash
uv run pytest tests/test_database_migrations.py tests/test_inference_api.py -q
```

- [ ] Add these Pydantic fields to `InferenceTaskCreate` and
  `InferenceTaskResponse`, relying on the existing camel-case alias generator:

```python
context_count: int = Field(default=1, ge=1)
worker_count: int = Field(default=1, ge=1)
```

- [ ] Extend the task model and additive migration:

```python
context_count: Mapped[int] = mapped_column(Integer, default=1)
worker_count: Mapped[int] = mapped_column(Integer, default=1)
```

```python
"context_count": "INTEGER NOT NULL DEFAULT 1",
"worker_count": "INTEGER NOT NULL DEFAULT 1",
```

- [ ] In the model validator, reject `worker_count > context_count`; persist,
  update, respond, and include both fields in desired task descriptors.
- [ ] Re-run the focused tests and require PASS.

### Task 2: Secondary rule validation and weighted capacity

- [ ] Add API tests covering secondary defaults, explicit pool sizes, invalid
  relationships, shared-pool counting, different-size splitting, and primary
  plus secondary context totals.
- [ ] Verify RED with:

```bash
uv run pytest tests/test_inference_api.py -q -k 'context or worker or capacity or secondary'
```

- [ ] Extend the secondary allowlist with `contextCount` and `workerCount`,
  normalize missing values to `1`, and validate positive integers with workers
  not exceeding contexts.
- [ ] Include primary counts in `_runtime_instance_key`. Add one canonical
  secondary key containing release ID, confidence threshold, inherited mask and
  policy, context count, and worker count.
- [ ] Replace `len(instances) + secondary_instance_count` with a pool plan whose
  value stores `context_count`; calculate:

```python
required_contexts = sum(pool.context_count for pool in unique_pools.values())
```

- [ ] Apply exclusive-mask overlap checks to unique primary and secondary
  pools, while allowing contexts inside one pool to share its own mask.
- [ ] Preserve the `inference_node_capacity_exceeded` error code and report
  `requiredContexts` plus `maxContexts`.
- [ ] Run all inference API tests and require PASS.

### Task 3: Runtime adapter pool configuration

- [ ] Add tests in `tests/test_inference_runtime_adapter.py` that assert:

```python
assert instance["context_count"] == 3
assert instance["worker_count"] == 2
assert instance["queue_capacity"] == 8
assert metadata["contextCount"] == expected_total
```

  Also assert matching counts share an instance, differing counts split,
  matching secondary rules share one instance, and invalid counts raise
  `RuntimeAdapterError`.
- [ ] Run those tests and verify RED.
- [ ] Add `_task_pool_config()` and `_secondary_pool_config()` boundary helpers
  that accept absent values as `1/1` and reject booleans, non-integers,
  non-positive values, and workers above contexts.
- [ ] Include counts in primary and secondary runtime keys. Maintain a global
  secondary-instance map during `prepare_revision()` so equivalent secondary
  rules reuse one generated instance.
- [ ] Extend `_instance_config()` with:

```python
"context_count": context_count,
"worker_count": worker_count,
"queue_capacity": max(8, worker_count * 2),
```

- [ ] Record per-instance pool descriptors and total context count in
  `revision.json`; keep old metadata readers tolerant of absent fields.
- [ ] Run `tests/test_inference_runtime_adapter.py` and
  `tests/test_inference_agent.py`; require PASS.

### Task 4: Reusable C++ execution pool

- [ ] Add `RknnExecutionPoolTest.cpp` first. Use fake nonzero
  `rknn_context` handles and injected process/destroy callbacks to prove:
  configurable workers drain jobs, active handle IDs are unique, failures
  return leases, partial initialization cleans all handles, and `stop()`
  resolves queued promises.
- [ ] Add a `BUILD_TESTING` target to CMake and run it to verify RED because
  `RknnExecutionPool` is missing.
- [ ] Implement `RknnExecutionPool` with this public contract:

```cpp
class RknnExecutionPool {
public:
    using Process = std::function<bool(Job&, rknn_context)>;
    using Fail = std::function<void(Job&, const std::string&)>;
    using Destroy = std::function<void(rknn_context)>;
    bool init(std::vector<rknn_context> contexts, size_t worker_count,
              size_t queue_capacity, Process process, Fail fail, Destroy destroy);
    bool commit(Job& job);
    void start();
    void stop();
    size_t context_count() const;
    size_t worker_count() const;
};
```

- [ ] The implementation must use a bounded FIFO job queue, FIFO available
  context queue, RAII lease return, idempotent start/stop, promise-safe failure,
  join-before-destroy shutdown, and no detached threads.
- [ ] Build and run the scheduler test until GREEN.

### Task 5: Integrate YOLO and structured RKNN instances

- [ ] Extend the C++ tests or instance probe fixture so old YAML reads `1/1`,
  explicit values are accepted, and workers above contexts fail initialization.
- [ ] Verify RED before changing the instance implementations.
- [ ] Replace each instance's singular `context_`, `worker_`, queue, and
  condition variable with `RknnExecutionPool pool_` and count settings.
- [ ] Keep tensor metadata queried from the primary context. Change inference
  functions to receive the leased context explicitly:

```cpp
bool process(Job& job, rknn_context context);
```

- [ ] Create context zero with `rknn_init`, apply the mask, duplicate contexts
  `1..N-1` with `rknn_dup_context`, apply the same mask to each, and transfer
  the complete vector to the pool only after all operations succeed.
- [ ] On any duplicate/mask failure, destroy all handles created so far and
  return false. Log instance name, counts, queue capacity, mask, and policy.
- [ ] Make `commit/start/stop` delegate to the shared pool and keep existing
  performance counters protected by their mutex.
- [ ] Run the C++ tests and compile `rknn_pipeline` and `rknn_instance_probe`.

### Task 6: Frontend task controls

- [ ] Extend `InferenceTask` and API create/update payload types with
  `contextCount` and `workerCount`; extend `SecondaryModelRule` similarly.
- [ ] Add a normalization/build check first that fails until missing secondary
  counts become `1/1` and explicit values retain their values.
- [ ] Add main task state defaults, edit hydration, reset, submit fields, and
  form validation in `InferencePage.tsx`.
- [ ] Add two numeric steppers to the NPU scheduling section and to each
  secondary rule. Use `min=1`, integer conversion, stable form-grid dimensions,
  and an inline validation message when workers exceed contexts.
- [ ] Display `N context / M worker` in task details and keep all existing
  mobile/desktop layouts non-overlapping.
- [ ] Run `npm run build` and `npm run test:ui`; require PASS.

### Task 7: Documentation and examples

- [ ] Update the runtime-adapter README with the JSON and YAML contracts,
  sharing key, capacity accounting, duplicate-context behavior, defaults, and
  rollback semantics.
- [ ] Update online/offline environment examples to explain that
  `RKNODE_MAX_MODEL_INSTANCES` counts actual contexts, not task count or worker
  count.
- [ ] Update operator deployment documentation found by searching all existing
  `RKNODE_MAX_MODEL_INSTANCES` and inference task examples.
- [ ] Add/adjust documentation contract tests in `tests/test_offline_deploy.py`
  and `tests/test_self_contained_deploy.py`, watching them fail before updating
  examples.

### Task 8: Full verification and image build

- [ ] Run focused suites, then full static checks and tests:

```bash
uv run pytest tests/test_database_migrations.py tests/test_inference_api.py \
  tests/test_inference_runtime_adapter.py tests/test_inference_agent.py -q
uv run ruff check backend workers tests
uv run pyright
uv run pytest -q
npm run build
npm run test:ui
```

- [ ] Configure, compile, and run C++ tests in a disposable build directory.
- [ ] Build without cache mutation or deleting prior images:

```bash
docker build -f deploy/rk3588/Dockerfile.node \
  -t rknode-rk3588-node:2026.08.19-business .
docker image inspect rknode-rk3588-node:2026.08.19-business
```

- [ ] Run `trellis-check`, fix all verified issues, update the RK3588 runtime
  spec with the new executable contract, and report exact validation evidence.
- [ ] Stop before any board SSH, image transfer, container replacement, or old
  image deletion. Request explicit approval for board validation.
