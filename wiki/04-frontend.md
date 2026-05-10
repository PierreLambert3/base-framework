# 4. The frontend

> Source: [GUI/gui.py](../GUI/gui.py),
> base class in [GUI/engine/frontend/logic.py](../GUI/engine/frontend/logic.py).

The frontend renders the GUI and processes user input. It runs in its own
OS process (spawned by `Custom_Frontend.start()`). Inside that process, the
`rendercanvas` event loop drives a per-frame callback that calls
`one_frame()`.

## 4.1 Process flow

```
start()  ── multiprocessing.Process(target=routine)
   └─► routine()  in the FE process
         ├─ initialise_scene()             → pygfx scene + camera + canvas
         ├─ load_intro_page()               → first Page added
         ├─ build_listeners()               → FE↔BE event subscriptions
         ├─ register_user_event_listener()  → pointer/key/wheel handlers
         ├─ canvas.request_draw(one_frame)
         └─ loop.run()                      ← blocking
```

`one_frame()` does, in order:

1. Throttle by `should_it_render()` (target FPS).
2. `scene.render()`.
3. `canvas.request_draw(one_frame)` (re-arm next frame).
4. `process_messages()` — drains the FE↔BE queue **and** every per-instance
   data stream queue.
5. `current_page.one_frame(mouse_coords)`.

## 4.2 Pages & the page tree

A *page* is the top-level container of UI elements (see
[05-pages-and-elements.md](05-pages-and-elements.md)). The frontend keeps
all instantiated pages in `self.pages` and exposes `self.current_page`. It
calls `current_page.one_frame()` every frame and routes pointer events to
its `manage_mouse_pointer_*` and `manage_mouse_wheel` methods.

Switch pages by:

```python
def switch_to_main_page(self):
    if self.current_page is not None:
        self.current_page.destroy()
    self.add_page(Main_Page(self.scene, "main page", self))
```

`destroy()` releases the GPU resources of the old page tree.

## 4.3 Communications wiring

The frontend has:

* `self.comms` — main FE↔BE channel.
* `self.data_stream_comms_per_instance[name]` — one `Communications` per
  worker instance. Created automatically when the backend sends
  `"new worker instance created"`. Contains its own listener registry so
  pages can subscribe with `add_data_stream_listener(name, event, cb)`.

Both groups are processed every frame in `process_messages()`.

## 4.4 User input

`on_user_event(event)` is the single entry point for `pointer_*`, `wheel`
and `key_*` events. It:

* Maps screen pixels to page-local pixels by ray-casting onto
  `current_page.pick_mesh` (an invisible plane mesh). See
  `Scene.xy_on_mesh` and `Scene.world_hit_on_mesh`
  ([GUI/engine/frontend/scene.py](../GUI/engine/frontend/scene.py)).
* Dispatches into `Page.manage_mouse_pointer_{move,down,up}` which then
  hit-tests the page's `clickable_elements` / `hoverable_elements`.
* Wheel events are dispatched first to `scrollable_elements` under the
  cursor; if no element consumes them they zoom the camera.
* `Escape` triggers `exit_program`. `f` cycles through `CHUNK_SPEEDS`.

## 4.5 Camera

A simple `pygfx.PerspectiveCamera`. The framework adds smooth
zoom-towards-cursor:

* Scroll up = zoom in towards the point under the cursor on the current
  page (clamped at `min_distance_to_page`).
* Scroll down = zoom out smoothly back to `default_camera_position`.

`_store_default_camera_state()` snapshots the current pose; call it after
positioning the camera in your `load_intro_page` / `switch_to_main_page`.

## 4.6 Hooks for project pages

When subclassing `Custom_Frontend` (or just adding a new `Page` subclass),
the framework looks for **optional** hooks on the active page:

| Hook on `current_page` | Triggered by |
|---|---|
| `on_new_worker_instance(name, config)` | backend sent `"new worker instance created"` |
| `on_worker_instance_info(info)` | worker's `_make_info_for_frontend()` was relayed |
| `manage_mouse_pointer_move/down/up` | pointer events (defined on `Page`, override if needed) |
| `manage_mouse_wheel` | wheel events (default dispatches to scrollable elements) |
| `one_frame(mouse_coords)` | every frame |
| `destroy()` | when leaving the page (release GPU resources) |

## 4.7 Things you typically override

| Hook | Why |
|---|---|
| `__init__` | tune FPS, camera defaults, custom state |
| `build_listeners` | add project FE-bound events from the backend |
| `load_intro_page`, `switch_to_*` | navigation between pages |
| `_handle_*` | project-specific events from backend / workers |
