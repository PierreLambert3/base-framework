# 1. Architecture overview

## Processes

The framework uses **3 kinds of OS process**, all spawned via
`multiprocessing.get_context("spawn")` (Windows-friendly):

```
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │       MAIN process       │         │     FRONTEND process     │
   │ ─ Custom_Backend         │  events │ ─ Custom_Frontend        │
   │ ─ CUDAManager            │ <─────> │ ─ pygfx scene + canvas   │
   │ ─ owns Manager().dict()  │  shared │ ─ pages / elements       │
   │ ─ orchestrates workers   │   dict  │ ─ user input             │
   └──────────────┬───────────┘         └──────────────────────────┘
                  │ events + data stream
                  ▼
   ┌──────────────────────────┐  ... 0..N worker processes
   │   WorkerInstance proc    │
   │ ─ CustomWorker           │
   │ ─ enters CUDAContext     │
   │ ─ runs simulation chunks │
   └──────────────────────────┘
```

* **Frontend process** — runs `Custom_Frontend.routine()`. Owns the OS
  window, the `pygfx.Scene`, and the page tree. Pure rendering + user input;
  it never touches CUDA.
* **Backend (= main process)** — runs `Custom_Backend.routine()`. Acts as
  the orchestrator: receives high-level requests from the frontend
  (e.g. "spawn an instance"), creates worker processes, relays messages.
  Holds the `CUDAManager` (device detection, kernel directory).
* **Worker instance processes** — one per `CustomWorker` you spawn. Each
  enters its own `CUDAContext` and runs a simulation/computation loop.
  Streams results to the frontend on a dedicated channel.

## Communication channels

For every running worker instance there are **two pairs of queues** (= 4
queues), plus one shared dict for the whole app:

| Channel | Direction | Purpose | ACK | Backed by |
|---|---|---|---|---|
| Main FE↔BE | bidirectional | UI requests, lifecycle, worker spawns | yes (auto) | `ctx.Queue()` |
| Main BE↔Worker | bidirectional | per-worker lifecycle / config | yes (auto) | `ctx.Queue()` |
| Data stream Worker→FE | one-way (FE-bound) | high-throughput results to draw | **no** | `manager.Queue()` |
| Shared dict | shared (R/W from any) | hot mutable state (mouse, etc.) | n/a | `manager.dict()` |

The split between "events with ACK" and "data stream without ACK" is
deliberate — see [02-communications.md](02-communications.md).

## Lifecycle

1. `main.py` creates the multiprocessing context, the FE↔BE queues and the
   shared dict.
2. `Custom_Frontend(...).start()` spawns the frontend process. The frontend
   renders the intro page and starts processing events.
3. `Custom_Backend(...).routine()` runs in the main process. It blocks until
   `exit program` is received, periodically draining all queues.
4. The user clicks a button on the main page → frontend sends
   `"launch worker instance"` to the backend → backend spawns a worker
   process, creates an un-entered `CUDAContext`, opens a data stream queue
   pair, and tells the frontend so it can hook up visualisation.
5. The worker enters its CUDA context, calls `initialise()`, sends
   `"info for frontend"` (relayed by the backend), and waits for
   `"frontend ready"` before entering its main simulation loop.
6. Each simulation time chunk: GPU work → host copy → push visual updates on the data stream
   queue. The frontend's page picks the data up via a per-instance listener.
7. On `Escape` or window close, the frontend sends `exit program`. The
   backend forwards it to all worker instances, joins them, and drains the
   queues.

## Where things live

| Concern | Location |
|---|---|
| Process scaffolding (entry point) | [main.py](../main.py) |
| Frontend subclass (project-specific) | [GUI/gui.py](../GUI/gui.py) |
| Backend subclass (project-specific) | [GUI/backend.py](../GUI/backend.py) |
| Engine: `Front_End`, `Back_End`, comms | [GUI/engine/](../GUI/engine/) |
| Engine: scene, pages, elements | [GUI/engine/frontend/](../GUI/engine/frontend/) |
| Project pages | [GUI/pages/](../GUI/pages/) |
| Worker scaffolding & subclass | [worker/](../worker/) |
| CUDA wrapper | [cuda_wrapper/](../cuda_wrapper/) |
| CUDA kernel sources | [kernels/](../kernels/) |
