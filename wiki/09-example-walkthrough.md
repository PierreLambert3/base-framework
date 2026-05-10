# 9. End-to-end walkthrough: the bouncing-points demo

This page traces the demo from the first click to a frame on screen. It is
the easiest way to understand how all the layers fit together. Reading
this together with [02-communications.md](02-communications.md) is enough
to grasp the whole framework.

## 9.1 What you see

* Run `python main.py`.
* The intro page appears with a `next page` button.
* Click it → main page with two buttons:
  * **spawn small instance (1,000 pts)** — creates a `CustomWorker`
    with `n_points=1_000`.
  * **spawn big instance (1,000,000 pts)** — same, with 1M points.
* Each click adds a scatterplot tile in a square-ish grid; each tile
  shows that worker's points bouncing inside the unit square.
* Press `f` to cycle through chunk sizes (4 / 40 / 180 / 1000 timesteps
  per chunk). Press `Escape` to quit.

## 9.2 Process diagram

```
main.py ─┬─► Custom_Frontend (process)  ── canvas, scene, pages
         │
         └─► Custom_Backend (this process)
                 │ "launch worker instance"
                 ▼
              spawns Worker process: CustomWorker
                 │ "info for frontend"        (relayed by backend)
                 │
                 │ "data stream: positions"   (direct W → FE)
                 ▼
              Scatterplot2DDynamic.update_data(...)
```

## 9.3 Step-by-step trace

### 1. Frontend boot

`Custom_Frontend.routine()` (in the FE process):

1. `initialise_scene()` builds a `pygfx.Scene`, camera, and renderer.
2. `load_intro_page()` adds `Intro_Page` (one button).
3. `build_listeners()` subscribes to:
   * `"Q1: how many timesteps per simulation chunk"`
   * `"new worker instance created"`
   * `"worker instance info"`
4. `register_user_event_listener(self.on_user_event)`.
5. `loop.run()` enters the `rendercanvas` event loop and begins calling
   `one_frame` ~26 times per second.

### 2. Backend boot

`Custom_Backend.routine()` (main process):

1. Sends `"Q1: how many timesteps per simulation chunk"` to ask the FE
   what chunk size to use.
2. Loops: sleep(0.25) → `process_messages()`.

### 3. Chunk-size handshake

* The frontend's listener for `Q1` sends back
  `"RE1.1: how many timesteps per simulation chunk"` with the current
  value (`CHUNK_SPEEDS[0] = 4`).
* The backend stores it in `self.simulation_chunk_size` and forwards
  it to every existing worker.

### 4. User clicks "next page"

* `Intro_Page.btn_a` fires `frontend.switch_to_main_page()`.
* The intro page is `destroy()`-ed; `Main_Page` is added.

### 5. User clicks "spawn small instance"

* `Main_Page._on_spawn_small_clicked` calls:
  ```python
  self.frontend.send("launch worker instance", {
      "config":             {"n_points": 1_000},
      "instance_name_hint": "small",
  })
  ```
* The backend's `_handle_launch_worker_instance` runs in the main
  process:
  1. picks name `"small 1"`,
  2. creates 4 queues (control + data stream),
  3. creates an un-entered `CUDAContext` via `cuda_manager.create_context(...)`,
  4. spawns the `CustomWorker` process,
  5. pushes the current chunk size to it,
  6. sends `"new worker instance created"` to the frontend with the
     data-stream queues attached to the payload.

### 6. Frontend allocates the visualisation

* `Custom_Frontend._handle_new_worker_instance_created` registers a
  per-instance data-stream `Communications`, then calls
  `current_page.on_new_worker_instance(name, config)`.
* `Main_Page.on_new_worker_instance`:
  1. appends the new instance to `_spawn_order`,
  2. relayouts the grid (creates a `Scatterplot2DDynamic` of the right
     capacity / colour),
  3. registers a data-stream listener for `"data stream: positions"`,
  4. **signals readiness** by sending `"frontend ready for worker instance"`.

### 7. Worker starts simulating

In the worker's process:

1. `cuda_ctx.enter()` + `check_yoself()` (round-trip sanity test).
2. `CustomWorker.initialise()` allocates GPU buffers, builds kernels:
   * Reads `n_points` from `config`.
   * `self._positions_gpu = ctx.m(positions_np)` — uploaded to device.
   * `LaunchConfig1D(props, n_workers=n_points)`.
   * `ctx.get_kernel("update_positions", "update_positions", cfg)` —
     compiles `kernels/update_positions.cu` to
     `kernels/compiled/update_positions.cubin` if needed.
3. `comms.send("info for frontend", {"instance name": …, "n_points": …})`.
   The backend relays it as `"worker instance info"`.
4. The worker spins on `frontend_ready_for_simulation`. As soon as the
   frontend's `"frontend ready for worker instance"` arrives (routed back
   by the backend via `send_to_instance`), the flag flips.

### 8. The simulation loop

Each iteration:

1. `kernels.update_positions(stream, worker, chunk_size)` — one
   `cuLaunchKernel` advancing every point by `chunk_size` timesteps.
2. `positions_gpu.to_host(out=host_buf, stream=stream)` then
   `stream.sync()`.
3. `data_stream_comms.send("data stream: positions",
   {"positions": host_buf}, needs_ack=False)` — drops the snapshot on
   the (no-ACK) data-stream queue.
4. Light sleep to avoid contention.

### 9. The frontend renders the points

Every frame (~26 Hz):

1. `Custom_Frontend.process_messages()` drains:
   * The main FE↔BE queue.
   * Every per-instance data-stream queue. Each
     `"data stream: positions"` event fires
     `Main_Page._handle_positions_data` for that instance.
2. `_handle_positions_data` calls
   `scatter.update_data(positions[:,0], positions[:,1])` which uploads
   into the pygfx GPU buffer (no realloc unless capacity grew).
3. `scene.render()` draws everything.

### 10. Shutdown

* `Escape` → `Custom_Frontend.exit_program(0)` stops the loop, sends
  `"exit program"` to the backend.
* The backend forwards `"exit program"` to every worker (no ACK).
* Each worker calls `_on_exit` (syncs streams), drains/cancels queues,
  and `cuda_ctx.exit()` releases the context, modules, allocations, and
  streams.
* `main.py` joins the frontend and worker processes and drains the
  queues.

## 9.4 Where to start your own project

Open these files in this order:

1. [main.py](../main.py) — how processes are wired.
2. [GUI/gui.py](../GUI/gui.py) and [GUI/backend.py](../GUI/backend.py) —
   the project subclasses you'll edit.
3. [GUI/pages/MainPage.py](../GUI/pages/MainPage.py) — the canonical
   page that talks to workers.
4. [worker/custom_worker.py](../worker/custom_worker.py) — the canonical
   GPU worker.
5. [kernels/update_positions.cu](../kernels/update_positions.cu) — the
   canonical kernel.

Then read [08-extending-the-framework.md](08-extending-the-framework.md)
for copy-pasteable recipes.
