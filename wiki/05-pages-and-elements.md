# 5. Pages, containers and graphical elements

> Sources: [GUI/engine/frontend/page.py](../GUI/engine/frontend/page.py),
> [GUI/engine/frontend/graphical_elements/](../GUI/engine/frontend/graphical_elements/).

The GUI tree has three levels:

```
Page                        ← top-level rectangle in world space (3D position, fixed size)
 └── Container              ← rectangle with relative coords inside parent (recursive)
      └── Element_2d        ← leaf widget (Button, Scatterplot, Text, …)
```

All three derive from `_GraphicalElement`, which holds:

* world-space position / size (`pos_xyz`, `size_xyz`),
* page-local position cache (`pagewise_xy`),
* registered `pygfx` objects (`_gfx_objects`),
* hit-testing helpers, hide/show, rotate, die.

## 5.1 Page

A `Page` is a `_GraphicalElement` with:

* a **border** (a `pygfx.Line` rectangle),
* an invisible **pick mesh** (`pygfx.Mesh` with `pick_write=True`,
  `opacity=0`) used to ray-cast pointer events into page coordinates,
* lists of interactive elements: `clickable_elements`, `hoverable_elements`,
  `scrollable_elements`,
* a list of top-level `containers`,
* lifecycle hooks: `on_show`, `on_hide`, `one_frame`, `destroy`.

A page is **added to the frontend** with `self.add_page(page)`. The first
added page becomes `current_page`. Switch pages by destroying the old one
and adding a new one (see [04-frontend.md](04-frontend.md)).

### Per-frame contract

The frontend calls `current_page.one_frame(mouse_coords)` after rendering.
You should:

* call `super().one_frame(mouse_coords)` (it ticks overlay particles, etc.),
* update any per-frame state that pulls fresh data from your worker
  instances (e.g. fade animations, FPS readouts).

Do **not** do GPU compute here — that lives in worker instances.

## 5.2 Container

`Container(name, parent, bl_xy_rel, size_xy_rel, borders=(t,r,b,l))` is a
relative-coordinate rectangle:

* `bl_xy_rel` and `size_xy_rel` are in `[0,1]` of the parent's size;
* `borders` enables the four sides individually;
* containers can nest containers and elements arbitrarily;
* `add(child)` is called automatically in `_GraphicalElement.__init__` when
  the parent is a container, so you usually just instantiate the child and
  it lands in the right place.

Lookup: `page.get(name)` and `container.get(name)` recurse through the
tree.

## 5.3 Element_2d

A leaf widget anchored in a parent container with relative bottom-left and
relative size. Concrete classes shipped:

| File | Class | Purpose |
|---|---|---|
| `button.py` | `Button_2d` | clickable rectangle with text |
| `text.py` | `Text` | static or dynamic text |
| `scatterplot_2d.py` | `Scatterplot2D` | static 2D points |
| `scatterplot_2d_dynamic.py` | `Scatterplot2DDynamic` | growing buffer, used by the demo |
| `scatterplot_3d_dynamic.py` | `Scatterplot3DDynamic` | 3D variant |
| `lines_dynamic.py` | dynamic line strips | |
| `rectangles_2d_dynamic.py` | many small rectangles | |
| `parallelepiped.py` | 3D box | |
| `overlay_particles.py` | screen-space particle FX | |

### Making an element interactive

To get callbacks, the element must be added to one (or more) of the page's
interactive lists. The base helpers are:

```python
self.register_hoverable()    # adds self to page.hoverable_elements
self.register_clickable()    # adds self to page.clickable_elements
self.register_scrollable()   # adds self to page.scrollable_elements
```

and the element must define the right callbacks
(`on_pointer_down_inside`, `on_pointer_up_inside`,
`on_pointer_move_inside`, `on_pointer_move_outside`, `on_wheel`).
`Button_2d` already does this internally.

### Hit-testing

`hit_by_page_coords(x, y)` does an axis-aligned bounding-box check in page
pixel coordinates. The pixel coordinates are derived from the pointer →
pick-mesh ray cast in `Scene.xy_on_mesh`.

## 5.4 Buffer-backed elements (efficient updates)

`Scatterplot2DDynamic` (and its 3D / lines / rectangles siblings) maintain
**pre-allocated GPU buffers** that grow geometrically when you stream more
points than the current capacity:

```python
scatter = Scatterplot2DDynamic(name, container, bl_rel, size_rel,
                               initial_capacity=1_000_000,
                               point_size=1, default_colour=PINK_ELECTRIC)
# every chunk:
scatter.update_data(xs, ys)         # zero allocations once capacity is reached
```

Internally:

* CPU-side double buffer, GPU `pygfx.Buffer` with `usage=COPY_DST|STORAGE`,
* `draw_range` is updated to render only the active subset,
* on capacity overflow, both CPU and GPU buffers are reallocated and the
  pygfx `Points` object is replaced.

Use these elements whenever you stream fresh data every frame from a
worker. Static one-shot variants (e.g. `Scatterplot2D`) avoid the buffer
machinery and are fine for one-time displays.

## 5.5 Theming

Common colours and helpers live in
[GUI/engine/frontend/theme.py](../GUI/engine/frontend/theme.py)
(`AMBER`, `ORANGE_YELLOW`, `PINK_ELECTRIC`, `interpolate_color`,
`brighten`, `to_rgba`, …). Post-processing (bloom + noise) is enabled by
the `POSTPROCESSING_BLOOM` / `POSTPROCESSING_NOISE` flags at the top of the
file.

## 5.6 Points-mode (swarm-rendered lines)

Every `_GraphicalElement` (page, container, leaf) accepts two optional
constructor parameters that control whether its `add_lines(...)` calls are
rendered as plain `pygfx.Line` segments or as CUDA-driven point swarms:

```python
super().__init__(..., point_mode=None, point_mode_params=None)
```

* **`point_mode`** — `1` to render lines as swarms via
  [`PointsModeManager`](../GUI/engine/frontend/points_mode.py), `0` to
  render as plain lines, `None` (default) to inherit from the closest
  ancestor that set it explicitly. If nothing in the chain sets it, the
  effective value is `0`.
* **`point_mode_params`** — optional `dict` of per-element overrides for
  the swarm behaviour (only relevant when the resolved `point_mode` is
  `1`). Recognised keys:
  `n_points_mul`, `spring_strength`, `jitter_strength`, `dt`, `damping`,
  `line_upwards_interaction`, `colour_range`. Each entry except
  `colour_range` is a `(mean, std)` tuple multiplied into the kernel
  defaults. Keys are resolved per-key by walking the parent chain (an
  element's dict only needs to contain the keys it wants to override).

`add_lines(...)` itself also accepts a per-call `point_mode_params=` dict
that takes precedence over the element's own dict and the inherited
chain.

### `colour_range` special case

`colour_range` is **not** inherited from ancestors. By default it is
derived from the element's own `colour` as
`(theme.darken(colour, 0.4), colour)`. Only the element's own
`point_mode_params={"colour_range": (...)}` (or a per-call
`add_lines(..., point_mode_params={"colour_range": (...)})`) overrides
this default.

### Typical recipes

Opt the whole page in to swarm rendering — every child element with
`point_mode=None` (the default) inherits this:

```python
class MyPage(Page):
    def __init__(self, scene, name, frontend, ...):
        super().__init__(scene, name, frontend, ..., point_mode=1)
```

Opt a single button out, even though the page enables it:

```python
Button_2d(..., point_mode=0)
```

Tweak the swarm look on one element only:

```python
Button_2d(..., point_mode_params={"n_points_mul": 4.0,
                                   "jitter_strength": (0.8, 0.1)})
```

The first time any page actually requests a `PointsModeManager` (via the
`add_lines` path), the frontend lazily creates a single shared CUDA
context — see [04-frontend.md](04-frontend.md#cuda-context).

## 5.7 Destroy / GPU resource lifecycle

`die()` removes a single element's `pygfx` objects from the scene. Pages
recursively call `die()` on every container and element when `destroy()`
is called. Always call `destroy()` on the previous page before adding the
next one — otherwise GPU buffers leak for the lifetime of the frontend
process.
