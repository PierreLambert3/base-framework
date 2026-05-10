# Framework Wiki

This wiki documents the GUI + CUDA framework that lives in this repository.
The code is organised as a small, opinionated scaffolding: a 3-process layout
(frontend, backend, one process per worker instance), event-based message
passing with automatic flow-control, a thin `pygfx`-based GUI with pages and
graphical elements, and a CUDA wrapper for project-specific compute kernels.

The repository ships with a working example (bouncing 2D points simulated on
the GPU and rendered as a scatterplot grid) so the framework can be exercised
end-to-end before you write any project code.

## How to read this wiki

Start with the architecture overview, then dig into the layer you care about.
Every Python module in the framework starts with a banner comment pointing at
the wiki page that documents it; agents and humans should follow those links
when they touch a file.

| # | Topic | File |
|---|-------|------|
| 1 | Big-picture architecture (processes, queues, lifecycle) | [01-architecture.md](01-architecture.md) |
| 2 | Communication protocols (events, ACK, shared dict, data stream) | [02-communications.md](02-communications.md) |
| 3 | Backend process (`Custom_Backend`, worker spawning) | [03-backend.md](03-backend.md) |
| 4 | Frontend process (`Custom_Frontend`, scene, camera, input) | [04-frontend.md](04-frontend.md) |
| 5 | Pages, containers and graphical elements | [05-pages-and-elements.md](05-pages-and-elements.md) |
| 6 | Worker instances (`WorkerInstance`, `CustomWorker`) | [06-worker-instances.md](06-worker-instances.md) |
| 7 | CUDA wrapper (`CUDAManager`, `CUDAContext`, kernels) | [07-cuda-wrapper.md](07-cuda-wrapper.md) |
| 8 | Extending the framework (recipes for your own project) | [08-extending-the-framework.md](08-extending-the-framework.md) |
| 9 | End-to-end example: the bouncing-points demo | [09-example-walkthrough.md](09-example-walkthrough.md) |

## TL;DR

* `main.py` spawns three kinds of process: **frontend** (rendering + UI),
  **backend** (orchestration, runs in the main process), and zero or more
  **worker instances** (GPU compute, one OS process each).
* All inter-process communication goes through `multiprocessing.Queue`s
  wrapped by [`Communications`](../GUI/engine/comms.py). Events carry
  `(name, data)` and get an automatic ACK; the next message with the same
  name is only sent once the previous one has been ACKed (latest-wins
  override). Continuous low-importance state goes through a `Manager().dict()`.
* The backend owns a `CUDAManager` and creates one un-entered `CUDAContext`
  per worker. The worker enters its context inside its own process.
* The frontend renders a tree of `Page > Container > Element` objects on top
  of a `pygfx` scene, and routes pointer/keyboard/wheel events to elements.
* You add project behaviour by **subclassing**: `Custom_Backend`,
  `Custom_Frontend`, `WorkerInstance`, and by adding new `Page`s under
  `GUI/pages/`. The framework calls the right hooks at the right times — see
  [08-extending-the-framework.md](08-extending-the-framework.md).
