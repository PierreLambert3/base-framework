from GUI.engine.frontend.graphical_elements.graphical_element import Element_2d

import wgpu
import pygfx
import numpy as np

class Scatterplot2D(Element_2d):
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel):
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel)

        # for now: placeholder size
        self.n = 1_000_000
        self.current_positions = np.empty((self.n, 3), dtype=np.float32)
        self.next_positions    = np.empty_like(self.current_positions)
        self.rng = np.random.Generator(np.random.PCG64())
        self.current_positions[:] = self.rng.uniform(-1.0, 1.0, size=self.current_positions.shape).astype(np.float32)
        # Explicit Buffer + send_data pattern (more explicit GPU upload control)
        self.positions_buffer = pygfx.Buffer(
            nitems=self.n,
            nbytes=self.n * 3 * 4,
            format="3xf4",
            usage=wgpu.BufferUsage.COPY_DST,
        )
        self.positions_buffer.send_data(0, self.current_positions)

        self.geometry = pygfx.Geometry(positions=self.positions_buffer)
        self.material = pygfx.PointsMaterial(size=int(1), color=(1, 0, 0.1, 1))
        self.points   = pygfx.Points(self.geometry, self.material)
        self.register_gfx_object(self.points)

    def receive_data(self, positions_array):
        n_new = positions_array.shape[0]
        if n_new > self.n:
            raise Exception("Scatterplot2D Update Error: received more points than allocated in buffer: need to implement dynamic resizing.")
        elif n_new < self.n:
            raise Exception("Downsizing not implemented yet: need to set the rest of the points to some default value.")

        sz = self.sz
        bl = self.bl
        
        # Normalize to [0, 1]
        min_vals = np.min(positions_array[:, :2], axis=0)
        max_vals = np.max(positions_array[:, :2], axis=0)
        range_vals = max_vals - min_vals + 1e-8
        normalized = (positions_array[:, :2] - min_vals) / range_vals  # [0, 1] range
        # scale to element size
        self.next_positions[:n_new, 0] = normalized[:, 0] * sz[0] + bl[0]
        self.next_positions[:n_new, 1] = normalized[:, 1] * sz[1] + bl[1]
        self.next_positions[:n_new, 2] = bl[2]
        self.current_positions, self.next_positions = self.next_positions, self.current_positions
        self.positions_buffer.send_data(0, self.current_positions)
        
       