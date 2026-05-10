# Wiki: wiki/05-pages-and-elements.md, wiki/06-worker-instances.md.
# Variant of `Main_Page` that uses `multiprocessing.shared_memory` to transport
# position arrays from the worker to the frontend, instead of shipping them as
# numpy payloads through the data-stream queue. The queue still carries a tiny
# "frame ready" doorbell, but the heavy data lives in a shared-memory block
# owned by each worker. See `GUI/engine/comms.py::Communications.create_shared_array`
# and `worker/custom_worker.py` (use_shared_memory branch).

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


class Shared_Memory_Page(Page):
    """
    Same UX as `Main_Page` (two spawn buttons + grid of scatterplots), but
    every worker spawned from this page is configured to publish its position
    array via `multiprocessing.shared_memory` instead of through the queue.
    """

    def __init__(self, scene, page_name, frontend, bl_xyz_px=(0, 0, 0), size_xyz_px=(2000, 1600, 0)):
        super().__init__(scene, page_name, frontend, bl_xyz_px, size_xyz_px)

        # Top toolbar: two spawn buttons
        toolbar = Container(page_name + " toolbar", self, (0.0, 0.82), (1.0, 0.18), borders=(0, 0, 1, 0))
        self.add_container(toolbar)
        self._btn_small = Button_2d(
            page_name + " btn small", toolbar, (0.10, 0.20), (0.30, 0.60),
            text=f"spawn small instance ({SMALL_N_POINTS:,} pts) [SHM]",
            text_colour=AMBER, colour=AMBER,
            pointer_click_callback=self._on_spawn_small_clicked,
        )
        self._btn_big = Button_2d(
            page_name + " btn big", toolbar, (0.60, 0.20), (0.30, 0.60),
            text=f"spawn big instance ({BIG_N_POINTS:,} pts) [SHM]",
            text_colour=PINK_ELECTRIC, colour=PINK_ELECTRIC,
            pointer_click_callback=self._on_spawn_big_clicked,
        )

        # Grid container holding the scatterplots
        self._grid_container = Container(page_name + " grid", self, GRID_BL, GRID_SIZE, borders=(0, 0, 0, 0))
        self.add_container(self._grid_container)

        # Per-instance state (preserved across re-layout)
        self._spawn_order        = []   # list[instance_name] in spawn order
        self._instance_configs   = {}   # instance_name -> config dict
        self._scatterplots       = {}   # instance_name -> Scatterplot2DDynamic
        self._positions_views    = {}   # instance_name -> ndarray view onto shared memory
        self._scatter_counter    = 0    # for unique names across re-creates

    # ---------------------------------------------------- button callbacks
    def _on_spawn_small_clicked(self, event, element, page_coords):
        self.frontend.send("launch worker instance", {
            "config": {
                "n_points":           SMALL_N_POINTS,
                "use_shared_memory":  True,
            },
            "instance_name_hint": "small-shm",
        })

    def _on_spawn_big_clicked(self, event, element, page_coords):
        self.frontend.send("launch worker instance", {
            "config": {
                "n_points":           BIG_N_POINTS,
                "use_shared_memory":  True,
            },
            "instance_name_hint": "big-shm",
        })

    # ----------------------------------------- frontend lifecycle hook
    def on_new_worker_instance(self, instance_name, config):
        """Allocates a scatterplot and subscribes to the SHM doorbell. The
        actual SHM attach happens once the worker sends its `info for frontend`
        payload (it is the worker that owns and names the block)."""
        self._spawn_order.append(instance_name)
        self._instance_configs[instance_name] = config

        self._relayout_grid()

        # Subscribe to this instance's SHM doorbell. The callback reads the
        # pre-attached shared array (populated in `on_worker_instance_info`).
        self.frontend.add_data_stream_listener(
            instance_name,
            "data stream: positions ready",
            lambda data, _name=instance_name: self._handle_positions_ready(_name, data),
        )

        # Tell the backend we're ready (it routes to the worker process).
        # NOTE: do NOT signal ready here if we still need the worker's info
        # message to arrive first -- but the existing protocol sends "info for
        # frontend" BEFORE waiting on "frontend ready", so the info will be
        # processed below in `on_worker_instance_info` by the time the doorbell
        # starts firing.
        self.frontend.send("frontend ready for worker instance", instance_name)

    def on_worker_instance_info(self, info):
        """One-time metadata from the worker. For SHM-backed instances this
        carries the descriptor needed to attach the positions block."""
        instance_name = info.get("instance name")
        if instance_name is None or instance_name not in self._instance_configs:
            return
        shm_info = info.get("shared_memory", {}).get("positions")
        if shm_info is None:
            return # this instance is not using shared memory
        comms = self.frontend.data_stream_comms_per_instance[instance_name]["comms"]
        # Use a per-instance key in case multiple SHM arrays are added later.
        arr = comms.attach_shared_array(
            "positions",
            shm_info["name"],
            tuple(shm_info["shape"]),
            np.dtype(shm_info["dtype"]),
        )
        self._positions_views[instance_name] = arr

    # ------------------------------------------------------ data stream
    def _handle_positions_ready(self, instance_name, data):
        scatter = self._scatterplots.get(instance_name)
        arr     = self._positions_views.get(instance_name)
        if scatter is None or arr is None:
            return
        # Read directly from shared memory. `update_data` memcopies into the
        # scatterplot's own swap buffer, so a torn read would at worst yield
        # one bad frame -- acceptable for v1 (no seqlock).
        scatter.update_data(arr[:, 0], arr[:, 1])

    # ---------------------------------------------------- grid layout
    def _grid_dims(self, n):
        if n <= 0:
            return 0, 0
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))
        return rows, cols

    def _cell_rect(self, row, col, rows, cols):
        cell_w = 1.0 / cols
        cell_h = 1.0 / rows
        bl_x = col * cell_w
        bl_y = (rows - 1 - row) * cell_h
        mx = CELL_MARGIN * cell_w
        my = CELL_MARGIN * cell_h
        return (bl_x + mx, bl_y + my), (cell_w - 2 * mx, cell_h - 2 * my)

    def _relayout_grid(self):
        for name in list(self._scatterplots.keys()):
            self._grid_container.remove_by_name(self._scatterplots[name].name)
        self._scatterplots.clear()

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

    def destroy(self):
        # Detach all attached shared arrays before tearing down the page so
        # the underlying handles are closed in this process. The owning worker
        # is responsible for `unlink`ing on its own exit.
        for instance_name in list(self._positions_views.keys()):
            comms_entry = self.frontend.data_stream_comms_per_instance.get(instance_name)
            if comms_entry is not None:
                try: comms_entry["comms"].release_shared_array("positions")
                except Exception: pass
        self._positions_views.clear()
        super().destroy()
