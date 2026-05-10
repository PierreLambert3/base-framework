# 6. Worker instances

> Sources: [worker/worker_instance.py](../worker/worker_instance.py),
> [worker/custom_worker.py](../worker/custom_worker.py),
> [worker/global_constants.py](../worker/global_constants.py).

A **worker instance** is one OS process running one project-specific
simulation / computation. The backend may spawn arbitrarily many.

## 6.1 The base class: `WorkerInstance`

`WorkerInstance.__init__` receives:

* `instance_name` — unique, used as a routing prefix on the BE↔W channel,
* `config` — opaque project-defined dict (whatever the spawn request sent),
* the four queues (control + data stream),
* the shared dict,
* an **un-entered** `CUDAContext` (see [07-cuda-wrapper.md](07-cuda-wrapper.md)).

It builds two `Communications`:

| Attribute | Channel | ACK |
|---|---|---|
| `self.comms` | BE ↔ this worker | yes |
| `self.data_stream_comms` | this worker → frontend | no |

And registers default listeners with the `comms_prefix(name)` prefix:

| Event | Handler |
|---|---|
| `<name> set simulation chunk size` | `_set_simulation_chunk_size` |
| `<name> frontend ready` | `_handle_frontend_ready` |
| `<name> exit program` | `exit_program` |

## 6.2 The `routine()` lifecycle

```python
def routine(self):
    self.cuda_ctx.enter()                       # 1. enter CUDA context (process-local)
    assert self.cuda_ctx.check_yoself()         #    smoke-test the GPU

    self.initialise()                           # 2. project setup (alloc GPU memory, kernels…)

    self.comms.send("info for frontend",
                    self._make_info_for_frontend())   # 3. announce ourselves
    while not self.frontend_ready_for_simulation:     #    wait for the FE to confirm it built UI
        time.sleep(0.1)
        self.process_messages()

    while self.process_messages():              # 4. main loop until exit_program
        self.run_simulation_chunk(self.simulation_chunk_size,
                                  self.selected_by_frontend,
                                  self.high_speed_mode)
        # ... track sim-seconds-per-real-second and report it ...
        time.sleep(0.06)
```

Note: `process_messages()` returns `self.running`; `exit_program` sets it
to `False`, and that's how the loop terminates.

## 6.3 Hooks you override

| Hook | Default | Override to … |
|---|---|---|
| `initialise()` | no-op | allocate GPU buffers, build kernels, initial state |
| `run_simulation_chunk(chunk_size, selected, high_speed)` | sleeps | run one batch of timesteps |
| `_on_chunk_size_changed(chunk_size, high_speed)` | no-op | resize / re-tune internal buffers |
| `_on_exit(data)` | no-op | sync streams, free non-CUDA resources |
| `_make_info_for_frontend()` | `{"instance name": …}` | add metadata used by the page (`config["n_points"]`, etc.) |
| `build_listeners()` | adds the 3 framework listeners | call `super()` and add project events |

## 6.4 Worker → frontend data stream

The data stream channel **bypasses ACK**, so use `needs_ack=False`:

```python
self.data_stream_comms.send(
    "data stream: positions",
    {"positions": self._positions_host},
    needs_ack=False,
)
```

Convention: prefix all data-stream event names with `"data stream: …"` to
make them easy to spot. The frontend page subscribes via
`frontend.add_data_stream_listener(name, event, callback)`.

## 6.5 The `selected_by_frontend` flag

The frontend can mark an instance as "currently visible / focused" by
sending or omitting `"worker instance deselected"`. Heavy instances should
honor this flag — for example, copy positions back to the host only when
selected, or run a coarser simulation when not selected. The default
implementation in `CustomWorker` ignores it; it is yours to use.

## 6.6 The example: `CustomWorker`

[`worker/custom_worker.py`](../worker/custom_worker.py) implements the
demo:

* `initialise()` allocates `n_points` x 2 float32 positions and velocities
  on the GPU (`self.cuda_ctx.m(numpy_array)` allocates **and copies**),
  pre-builds the `update_positions` kernel with a 1D launch config, and
  pre-allocates a host buffer for the read-back.
* `run_simulation_chunk(chunk_size, …)`:
  1. launches `update_positions` on the compute stream for `chunk_size`
     timesteps,
  2. asynchronously copies positions back to the pinned host buffer,
  3. syncs the stream,
  4. fires `"data stream: positions"` with the latest snapshot.
* `_on_exit` syncs the stream so all GPU work is done before the context
  is torn down.

This is the **canonical template** for a GPU worker — copy and edit it.
