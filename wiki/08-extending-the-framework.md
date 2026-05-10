# 8. Extending the framework: where your project code goes

This page is the **recipe book**. It tells you exactly which files to add
or subclass for typical tasks.

The framework is intentionally small. The pattern is always the same:
**subclass and override hooks**, do not modify engine files
(`GUI/engine/*`, `worker/worker_instance.py`, `cuda_wrapper/*`).

## 8.1 Project entry point

[main.py](../main.py) is already wired. You typically do not edit it.

The two project classes you can edit freely are:

* `Custom_Frontend` in [GUI/gui.py](../GUI/gui.py) — your frontend subclass.
* `Custom_Backend` in [GUI/backend.py](../GUI/backend.py) — your backend
  subclass.

## 8.2 Recipe — Add a new page

1. Create a file under [GUI/pages/](../GUI/pages/), e.g. `MyPage.py`:

   ```python
   from GUI.engine.frontend.page import Page
   from GUI.engine.frontend.graphical_elements.graphical_element import Container
   from GUI.engine.frontend.graphical_elements.button import Button_2d
   from GUI.engine.frontend.theme import AMBER

   class My_Page(Page):
       def __init__(self, scene, page_name, frontend,
                    bl_xyz_px=(0,0,0), size_xyz_px=(2000,1600,0)):
           super().__init__(scene, page_name, frontend, bl_xyz_px, size_xyz_px)
           toolbar = Container(page_name+" toolbar", self, (0, 0.9), (1, 0.1))
           self.add_container(toolbar)
           Button_2d(page_name+" go", toolbar, (0.4, 0.2), (0.2, 0.6),
                     text="go", text_colour=AMBER, colour=AMBER,
                     pointer_click_callback=self._on_go_clicked)

       def _on_go_clicked(self, event, element, page_coords):
           self.frontend.send("my project event", {"foo": 42})
   ```

2. Wire it from `Custom_Frontend`:

   ```python
   def switch_to_my_page(self):
       if self.current_page is not None:
           self.current_page.destroy()
       self.add_page(My_Page(self.scene, "my page", self))
   ```

   Call `switch_to_my_page()` from a button on another page (or from
   `load_intro_page` if it should be the first page).

## 8.3 Recipe — Listen to a new event from the backend

In `Custom_Frontend.build_listeners`:

```python
def build_listeners(self):
    super().build_listeners()
    self.add_listener("my data update", self._on_my_data)

def _on_my_data(self, data):
    if self.current_page is not None and hasattr(self.current_page, "on_my_data"):
        self.current_page.on_my_data(data)
```

In `Custom_Backend.build_listeners`:

```python
def build_listeners(self):
    super().build_listeners()
    self.add_listener("my project event", self._handle_my_project_event)

def _handle_my_project_event(self, data):
    # do work, then maybe:
    self.send("my data update", {"result": ...})
```

`needs_ack=True` is the default. Two listeners for the same name is an
error.

## 8.4 Recipe — Add a new worker type

1. Subclass `WorkerInstance` (look at
   [worker/custom_worker.py](../worker/custom_worker.py)):

   ```python
   class FluidWorker(WorkerInstance):
       def initialise(self):
           # alloc GPU buffers, build kernels
           ...
       def run_simulation_chunk(self, chunk_size, selected, high_speed):
           # one batch of GPU work + push results on data stream
           self.data_stream_comms.send("data stream: density",
                                        {"density": self._density_host},
                                        needs_ack=False)
       def _on_exit(self, data):
           self.streams.sync_all()
       def _make_info_for_frontend(self):
           info = super()._make_info_for_frontend()
           info["grid_size"] = self._grid_size
           return info
   ```

2. Either replace `CustomWorker` everywhere, or override
   `Custom_Backend._make_worker_instance` to pick a class based on `config`:

   ```python
   def _make_worker_instance(self, name, config, qa, qb, sd, ds_a, ds_b, cuda_ctx):
       if config.get("kind") == "fluid":
           return FluidWorker(name, config, qa, qb, sd, ds_a, ds_b, cuda_ctx)
       return CustomWorker(name, config, qa, qb, sd, ds_a, ds_b, cuda_ctx)
   ```

3. Trigger from a button:

   ```python
   self.frontend.send("launch worker instance", {
       "config":             {"kind": "fluid", "grid": 256},
       "instance_name_hint": "fluid",
   })
   ```

## 8.5 Recipe — Subscribe a page to a worker's data stream

When a worker is spawned, the backend tells the frontend, which then asks
the active page (`on_new_worker_instance` hook):

```python
class My_Page(Page):
    def on_new_worker_instance(self, instance_name, config):
        # 1. allocate UI for this instance (a scatterplot, a plot, …)
        self._allocate_panel_for(instance_name, config)

        # 2. subscribe to its data stream
        self.frontend.add_data_stream_listener(
            instance_name,
            "data stream: positions",
            lambda data, _n=instance_name: self._on_positions(_n, data),
        )

        # 3. signal that we are ready (worker waits for this before its loop)
        self.frontend.send("frontend ready for worker instance", instance_name)

    def _on_positions(self, instance_name, data):
        scatter = self._scatter_for[instance_name]
        positions = data["positions"]
        scatter.update_data(positions[:, 0], positions[:, 1])
```

The third step (`"frontend ready for worker instance"`) is **mandatory** —
the worker's `routine()` blocks in a `while not frontend_ready_for_simulation`
spinner until it receives the matching event.

## 8.6 Recipe — Add a CUDA kernel

1. Drop a `.cu` file under [kernels/](../kernels/), e.g.
   `kernels/my_kernel.cu`:

   ```cpp
   #include <stdint.h>
   extern "C" __global__ void my_kernel(float* a, uint32_t n) {
       uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       if (i < n) a[i] *= 2.0f;
   }
   ```

2. In your worker's `initialise()`:

   ```python
   from cuda_wrapper import LaunchConfig1D, DeviceProperties
   props = DeviceProperties()
   cfg   = LaunchConfig1D(props, n_workers=n)
   self.k_my = self.cuda_ctx.get_kernel("my_kernel", "my_kernel", cfg)
   ```

3. Launch it:

   ```python
   self.k_my.launch(self.streams.compute, gpu_a, np.uint32(n))
   ```

The `nvcc` compile + cubin caching is automatic; recompiles only when the
source (or any quoted-`#include`d header) changes.

## 8.7 Recipe — Use the shared dict for live state

The shared dict is good for tiny per-frame state that the frontend wants to
expose without enqueuing an event every frame.

Frontend writes:
```python
self.update_shared_dict("camera_pos", tuple(self.scene.camera.local.position))
```

Backend / workers read:
```python
pos = self.comms.shared.get("camera_pos", default=(0,0,0))
```

## 8.8 Common pitfalls

* Forgetting to `super().build_listeners()` when overriding it → you'll
  silently miss framework events.
* Forgetting `needs_ack=False` on a high-rate data stream send → the queue
  will block on ACKs / the receiver will see only the latest value
  instead of every chunk. Use the dedicated data stream channel.
* Doing GPU work outside the worker process — the CUDA context is not
  shared.
* Modifying engine files instead of subclassing — your project will rot
  the next time the engine is updated. Keep all project code in
  `GUI/pages/`, `GUI/gui.py`, `GUI/backend.py`, `worker/custom_worker.py`,
  `kernels/`, plus any new files you add under those directories.
