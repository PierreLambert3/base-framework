# 3. The backend

> Source: [GUI/backend.py](../GUI/backend.py),
> base class in [GUI/engine/backend/logic.py](../GUI/engine/backend/logic.py).

The backend is the **orchestrator**. It runs in the main process (no
separate `start()` for itself). Subclass [`Custom_Backend`](../GUI/backend.py)
to add project-specific behaviour.

## 3.1 Responsibilities

* Talk to the frontend over the FE↔BE channel.
* Spawn / kill / route messages for **worker instances**.
* Hold the **`CUDAManager`** (one per app — detects the device and provides
  per-process kernel directory configuration).
* Own the `multiprocessing.Manager` used to create the data-stream queues
  that are sent between processes.

It does **not**:

* render anything,
* run CUDA kernels itself (those run inside worker processes),
* know about graphical elements.

## 3.2 Anatomy

```python
class Custom_Backend(Back_End):
    def __init__(self, ctx, manager, q_in, q_out, shared_dict):
        super().__init__(ctx, manager, q_in, q_out, shared_dict)
        self.cuda_manager = CUDAManager(device_id=0, kernel_dir="kernels")

        # bookkeeping for spawned workers
        self.worker_instance_names = []
        self.worker_processes      = []
        self.comms_instances       = []          # one Communications per worker
        self.cuda_contexts         = []          # CUDAContext per worker
        self.data_stream_queues    = []
        self.build_listeners()

    def routine(self):                            # blocking main loop
        self.send("Q1: how many timesteps per simulation chunk", None)
        while self.running:
            time.sleep(0.25)
            self.process_messages()

    def process_messages(self):
        super().process_messages()                # FE↔BE
        for comms in self.comms_instances:        # BE↔each worker
            comms.process_messages()
```

The `routine()` is blocking and lives in `main.py`'s main thread; on
`exit program` it falls through, joins all worker processes, and `main.py`
joins the frontend process.

## 3.3 Listeners (default events the backend understands)

| Event from frontend | Handler | Role |
|---|---|---|
| `"exit program"` | `exit_program` | shutdown everything |
| `"RE1.1: how many timesteps per simulation chunk"` | `_set_simulation_chunk_size` | propagate chunk size to all workers |
| `"launch worker instance"` | `_handle_launch_worker_instance` | spawn a `CustomWorker` |
| `"frontend ready for worker instance"` | `_handle_frontend_ready_for_worker_instance` | tell that instance to start |
| `"worker instance deselected"` | `_handle_instance_deselected` | demote instance priority |
| `"info for frontend"` (from worker, relayed) | `_handle_info_from_instance` | forward as `"worker instance info"` |

To add your own event, override `build_listeners` (call `super()`) and
register more.

## 3.4 Spawning a worker instance

`launch_worker_instance(config, instance_name_hint=None)` is the canonical
entry point. It:

1. Builds a unique instance name (`"<hint> <N>"` or `"WorkerInstance <N>"`).
2. Creates **two** queue pairs (control + data stream).
3. Creates a `CUDAContext` via `self.cuda_manager.create_context(...)`.
   The context is entered **inside** the child process — CUDA contexts are
   process-local and cannot be shared across `fork`/`spawn`.
4. Instantiates the worker (factory `_make_worker_instance`, which you can
   override), calls `instance.start()` to spawn the process.
5. Pushes the current chunk size to the new instance.
6. Sends `"new worker instance created"` to the frontend with the
   data-stream queues, so the frontend can hook up its data-stream
   `Communications` and let the active page allocate visualisation.

If you want a different worker class for a different config, override
`_make_worker_instance`:

```python
def _make_worker_instance(self, name, config, qa, qb, shared, ds_a, ds_b, cuda_ctx):
    if config.get("kind") == "fluid":
        return FluidWorker(name, config, qa, qb, shared, ds_a, ds_b, cuda_ctx)
    return ParticleWorker(name, config, qa, qb, shared, ds_a, ds_b, cuda_ctx)
```

## 3.5 Sending to a specific worker

```python
self.send_to_instance(name, "set simulation chunk size", chunk, require_ack=False)
```

This prefixes the event with `comms_prefix(name)` (= `"<name> "`) so that
the worker's listener registry, which is registered with the prefix,
matches.

## 3.6 Things you typically override

| Hook | When | Why |
|---|---|---|
| `__init__` | once | configure `cuda_manager` (device id, kernel dir) |
| `build_listeners` | once | add project events from the frontend |
| `routine` | rarely | inject a periodic task or a different shutdown trigger |
| `_make_worker_instance` | when you have multiple worker types | pick the right class per config |
