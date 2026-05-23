# Wiki: wiki/05-pages-and-elements.md, wiki/06-worker-instances.md,
#       wiki/09-example-walkthrough.md (this page is the main demo),
#       wiki/08-extending-the-framework.md (recipe: subscribe a page to a
#       worker's data stream).
# Demonstrates how a page allocates per-instance UI on `on_new_worker_instance`,
# subscribes to its data-stream channel, and signals readiness to the worker.

import math
import numpy as np

from GUI.engine.frontend.page import Page
from GUI.engine.frontend.graphical_elements.graphical_element import Container
from GUI.engine.frontend.graphical_elements.button import Button_2d
from GUI.engine.frontend.graphical_elements.scatterplot_2d_dynamic import Scatterplot2DDynamic
from GUI.engine.frontend.theme import AMBER, ORANGE_YELLOW, PINK_ELECTRIC


SMALL_N_POINTS = 1_000
BIG_N_POINTS   = 1_000_000

# Sub-rectangle of the page used to lay out scatterplots (relative coords)
GRID_BL   = (0.02, 0.02)
GRID_SIZE = (0.96, 0.78)

# Margin inside each grid cell (relative to cell)
CELL_MARGIN = 0.04


class Main_Page(Page):
    """
    Page hosting an arbitrary number of momentum-simulation scatterplots laid
    out in a square-ish grid that grows as new worker instances are spawned.

    Two buttons let the user spawn either a small (1k points) or big (1M
    points) instance. All scatterplots share the same grid; only the point
    count differs between small and big.
    """

    def __init__(self, scene, page_name, frontend, bl_xyz_px=(0, 0, 0), size_xyz_px=(2000, 1600, 0)):
        super().__init__(scene, page_name, frontend, bl_xyz_px, size_xyz_px)

        # Top toolbar: two spawn buttons
        toolbar = Container(page_name + " toolbar", self, (0.0, 0.82), (1.0, 0.18), borders=(0, 0, 1, 0))
        self.add_container(toolbar)
        self._btn_small = Button_2d(
            page_name + " btn small", toolbar, (0.10, 0.20), (0.30, 0.60),
            text=f"spawn small instance ({SMALL_N_POINTS:,} pts)",
            text_colour=AMBER, colour=AMBER,
            pointer_click_callback=self._on_spawn_small_clicked,
        )
        self._btn_big = Button_2d(
            page_name + " btn big", toolbar, (0.60, 0.20), (0.30, 0.60),
            text=f"spawn big instance ({BIG_N_POINTS:,} pts)",
            text_colour=AMBER, colour=AMBER,
            pointer_click_callback=self._on_spawn_big_clicked,
            ignore_pointmode=True
        )

        # Grid container holding the scatterplots
        self._grid_container = Container(page_name + " grid", self, GRID_BL, GRID_SIZE, borders=(0, 0, 0, 0))
        self.add_container(self._grid_container)

        # Per-instance state (preserved across re-layout)
        self._spawn_order = []                      # list[instance_name] in spawn order
        self._instance_configs = {}                 # instance_name -> config dict
        self._scatterplots = {}                     # instance_name -> Scatterplot2DDynamic
        self._scatter_counter = 0                   # for unique names across re-creates

    # ---------------------------------------------------- button callbacks
    def _on_spawn_small_clicked(self, event, element, page_coords):
        self.frontend.send("launch worker instance", {
            "config":             {"n_points": SMALL_N_POINTS},
            "instance_name_hint": "small",
        })

    def _on_spawn_big_clicked(self, event, element, page_coords):
        self.frontend.send("launch worker instance", {
            "config":             {"n_points": BIG_N_POINTS},
            "instance_name_hint": "big",
        })

    # ----------------------------------------- frontend lifecycle hook
    def on_new_worker_instance(self, instance_name, config):
        """Called by Custom_Frontend when the backend has spawned a new
        worker instance. Allocates a scatterplot, re-layouts the grid, hooks
        up the data stream listener, and sends the 'ready' signal."""
        self._spawn_order.append(instance_name)
        self._instance_configs[instance_name] = config

        self._relayout_grid()

        # Subscribe to this instance's data stream
        self.frontend.add_data_stream_listener(
            instance_name,
            "data stream: positions",
            lambda data, _name=instance_name: self._handle_positions_data(_name, data),
        )

        # Tell the backend we're ready (it routes to the worker process)
        self.frontend.send("frontend ready for worker instance", instance_name)

    # ------------------------------------------------------ data stream
    def _handle_positions_data(self, instance_name, data):
        scatter = self._scatterplots.get(instance_name)
        if scatter is None:
            return
        positions = data.get("positions")
        if positions is None:
            return
        positions = np.asarray(positions, dtype=np.float32)
        scatter.update_data(positions[:, 0], positions[:, 1])

    # ---------------------------------------------------- grid layout
    def _grid_dims(self, n):
        if n <= 0:
            return 0, 0
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        return rows, cols

    def _cell_rect(self, row, col, rows, cols):
        """Return (bl_xy_rel, size_xy_rel) for a cell, relative to grid container."""
        cell_w = 1.0 / cols
        cell_h = 1.0 / rows
        # row 0 is the TOP visually, but rel coords have y growing upward, so:
        bl_x = col * cell_w
        bl_y = (rows - 1 - row) * cell_h
        # apply inner margin
        mx = CELL_MARGIN * cell_w
        my = CELL_MARGIN * cell_h
        return (bl_x + mx, bl_y + my), (cell_w - 2 * mx, cell_h - 2 * my)

    def _relayout_grid(self):
        """Destroy existing scatterplots and re-create them at the new grid
        size. Existing instances keep their slot in the spawn order; their
        next data stream message will repopulate the new GPU buffers."""
        # 1. tear down existing scatter elements (GPU resources released)
        for name in list(self._scatterplots.keys()):
            self._grid_container.remove_by_name(self._scatterplots[name].name)
        self._scatterplots.clear()

        # 2. compute grid and recreate
        n = len(self._spawn_order)
        rows, cols = self._grid_dims(n)
        for idx, instance_name in enumerate(self._spawn_order):
            row, col = idx // cols, idx % cols
            bl_rel, size_rel = self._cell_rect(row, col, rows, cols)
            self._scatter_counter += 1
            elem_name = f"scatter[{instance_name}]#{self._scatter_counter}"
            cfg       = self._instance_configs[instance_name]
            n_points  = int(cfg.get("n_points", SMALL_N_POINTS))
            colour    = PINK_ELECTRIC if n_points >= BIG_N_POINTS else ORANGE_YELLOW
            scatter = Scatterplot2DDynamic(
                elem_name, self._grid_container, bl_rel, size_rel,
                initial_capacity=max(n_points, 1024),
                point_size=1,
                default_colour=colour,
            )
            self._scatterplots[instance_name] = scatter

    # ------------------------------------------------------ lifecycle
    def one_frame(self, mouse_coords=(0, 0)):
        super().one_frame(mouse_coords)
