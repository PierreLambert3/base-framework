from GUI.engine.frontend.graphical_elements.graphical_element import Element_3d

import wgpu
import pygfx
import numpy as np


class Scatterplot3DDynamic(Element_3d):
    """
    A 3D scatterplot element with dynamically varying point count.
    
    This class extends Element_3d and supports full 3D rotation, translation,
    and efficient dynamic point updates. Points are rendered in 3D space and 
    projected to 2D by the engine's camera/projection system.
    
    Key efficiency features:
    - Pre-allocated GPU buffers that grow geometrically when needed
    - No re-allocation on every update: reuses existing buffers when possible
    - Efficient partial buffer updates via `update_range`
    - Uses draw_range to render only the active subset of points
    - Double-buffered CPU arrays to avoid allocation overhead
    
    The buffer capacity starts at `initial_capacity` and grows by a factor of
    `growth_factor` whenever the incoming data exceeds current capacity.
    Points outside the active range are not rendered (via geometry.positions.draw_range).
    
    Parameters
    ----------
    unique_name : str
        Unique identifier for this element.
    parent : Element
        Parent element in the GUI hierarchy.
    bl_xyz_px : tuple
        Bottom-left-back corner position in absolute pixels (x, y, z).
    size_xyz_px : tuple
        Size in absolute pixels (width, height, depth).
    initial_capacity : int
        Initial number of points the buffer can hold. Default 20000.
    growth_factor : float
        Factor by which to grow buffers when capacity is exceeded. Default 1.2.
    point_size : int
        Size of points in pixels. Default 2.
    default_colour : tuple
        Default RGBA colour for points. Default (1, 0, 0.1, 1).
    colour_mode : str
        "uniform" for single colour, "vertex" for per-point colours.
    """
    
    def __init__(self, unique_name, parent, bl_xyz_px, size_xyz_px,
                 initial_capacity=20_000, growth_factor=1.2, 
                 point_size=2, default_colour=(1, 0, 0.1, 1),
                 colour_mode="uniform"):
        super().__init__(unique_name, parent, bl_xyz_px, size_xyz_px)
        
        self.initial_capacity = initial_capacity
        self.growth_factor = growth_factor
        self.point_size = point_size
        self.default_colour = default_colour
        self.colour_mode = colour_mode  # "uniform" or "vertex"
        
        # Current number of active points (actual data)
        self._active_count = 0
        # Current buffer capacity
        self._capacity = initial_capacity
        
        # === CPU-side double buffers for positions (avoid allocations) ===
        # These are the working arrays we write into before sending to GPU
        # Points are stored centered at origin for proper rotation support
        self._positions_cpu = np.zeros((self._capacity, 3), dtype=np.float32)
        self._positions_cpu_swap = np.zeros((self._capacity, 3), dtype=np.float32)
        
        # === CPU-side buffer for colours (only if vertex colour mode) ===
        if self.colour_mode == "vertex":
            self._colours_cpu = np.zeros((self._capacity, 4), dtype=np.float32)
            self._colours_cpu_swap = np.zeros((self._capacity, 4), dtype=np.float32)
        
        # === GPU Buffers ===
        self._create_gpu_resources()
        
    def _create_gpu_resources(self):
        """Create or recreate GPU buffers with current capacity."""
        # Position buffer on GPU
        # Using COPY_DST allows us to update the buffer data
        self._positions_buffer = pygfx.Buffer(
            data=self._positions_cpu,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE,
        )
        
        # Colours buffer (only if using vertex colours)
        if self.colour_mode == "vertex":
            self._colours_buffer = pygfx.Buffer(
                data=self._colours_cpu,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE,
            )
            self._geometry = pygfx.Geometry(
                positions=self._positions_buffer,
                colors=self._colours_buffer
            )
            self._material = pygfx.PointsMaterial(
                size=self.point_size, 
                color_mode="vertex"
            )
        else:
            self._geometry = pygfx.Geometry(positions=self._positions_buffer)
            self._material = pygfx.PointsMaterial(
                size=self.point_size, 
                color=self.default_colour
            )
        
        # Set initial draw range to 0 (nothing to draw yet)
        self._positions_buffer.draw_range = (0, 0)
        
        # Create Points object
        self._points = pygfx.Points(self._geometry, self._material)
        
        # Position the points object at the element's center (for rotation support)
        self._points.local.position = self.center
        
        self.register_gfx_object(self._points)
    
    def _grow_buffers(self, required_capacity):
        """
        Grow all buffers to accommodate at least `required_capacity` points.
        Uses geometric growth to amortize reallocation cost.
        """
        new_capacity = self._capacity
        while new_capacity < required_capacity:
            new_capacity = max(int(new_capacity * self.growth_factor), new_capacity + 1)
        
        # Allocate new CPU arrays
        new_positions_cpu = np.zeros((new_capacity, 3), dtype=np.float32)
        new_positions_cpu_swap = np.zeros((new_capacity, 3), dtype=np.float32)
        
        # Copy existing data
        if self._active_count > 0:
            new_positions_cpu[:self._active_count] = self._positions_cpu[:self._active_count]
        
        self._positions_cpu = new_positions_cpu
        self._positions_cpu_swap = new_positions_cpu_swap
        
        if self.colour_mode == "vertex":
            new_colours_cpu = np.zeros((new_capacity, 4), dtype=np.float32)
            new_colours_cpu_swap = np.zeros((new_capacity, 4), dtype=np.float32)
            if self._active_count > 0:
                new_colours_cpu[:self._active_count] = self._colours_cpu[:self._active_count]
            self._colours_cpu = new_colours_cpu
            self._colours_cpu_swap = new_colours_cpu_swap
        
        self._capacity = new_capacity
        
        # Recreate GPU resources with new capacity
        # First, remove old points object from scene
        self.unregister_gfx_object(self._points)
        
        # Create new GPU resources
        self._create_gpu_resources()
        
        # If we had active points, update the GPU with their data
        if self._active_count > 0:
            self._positions_buffer.update_range(0, self._active_count)
            self._positions_buffer.draw_range = (0, self._active_count)
            if self.colour_mode == "vertex":
                self._colours_buffer.update_range(0, self._active_count)
    
    def update_data(self, points_x, points_y, points_z, points_colours=None):
        """
        Update the scatterplot with new data.
        
        This is the main method for updating the visualization efficiently.
        It reuses existing buffers when possible and only grows them when needed.
        
        Coordinates are expected to be in [0, 1] range and will be scaled to 
        fit within the element's bounding box, centered at origin for rotation.
        
        Parameters
        ----------
        points_x : array-like
            X coordinates of points (1D array of length N), in [0, 1] range.
        points_y : array-like
            Y coordinates of points (1D array of length N), in [0, 1] range.
        points_z : array-like
            Z coordinates of points (1D array of length N), in [0, 1] range.
        points_colours : array-like, optional
            Per-point colours as (N, 4) RGBA array or (N, 3) RGB array.
            Only used if colour_mode="vertex". If None and colour_mode="vertex",
            uses the default colour for all points.
            
        Notes
        -----
        - If N > current capacity, buffers will be grown (geometric growth).
        - If N < current capacity, extra buffer space is simply not rendered.
        - Coordinates are mapped from [0, 1] to element bounds, centered at origin.
        """
        points_x = np.asarray(points_x, dtype=np.float32).ravel()
        points_y = np.asarray(points_y, dtype=np.float32).ravel()
        points_z = np.asarray(points_z, dtype=np.float32).ravel()
        
        n_points = len(points_x)
        if len(points_y) != n_points or len(points_z) != n_points:
            raise ValueError(
                f"points_x, points_y, and points_z must have same length, "
                f"got {n_points} vs {len(points_y)} vs {len(points_z)}"
            )
        
        # Check if we need to grow buffers
        if n_points > self._capacity:
            self._grow_buffers(n_points)
        
        # Get element dimensions (half sizes for centering around origin)
        hw, hh, hd = self.size[0] / 2, self.size[1] / 2, self.size[2] / 2
        
        # === Write into swap buffer (coordinates centered at origin for rotation) ===
        # Map [0, 1] coordinates to [-half_size, +half_size]
        self._positions_cpu_swap[:n_points, 0] = (points_x * 2.0 - 1.0) * hw
        self._positions_cpu_swap[:n_points, 1] = (points_y * 2.0 - 1.0) * hh
        self._positions_cpu_swap[:n_points, 2] = (points_z * 2.0 - 1.0) * hd
        
        # Swap buffers
        self._positions_cpu, self._positions_cpu_swap = self._positions_cpu_swap, self._positions_cpu
        
        # Handle colours if in vertex mode
        if self.colour_mode == "vertex":
            if points_colours is not None:
                points_colours = np.asarray(points_colours, dtype=np.float32)
                if points_colours.ndim == 1:
                    # Single colour for all points
                    self._colours_cpu_swap[:n_points] = points_colours
                elif points_colours.shape[0] != n_points:
                    raise ValueError(
                        f"points_colours must have {n_points} rows, "
                        f"got {points_colours.shape[0]}"
                    )
                elif points_colours.shape[1] == 3:
                    # RGB -> RGBA (add alpha = 1)
                    self._colours_cpu_swap[:n_points, :3] = points_colours
                    self._colours_cpu_swap[:n_points, 3] = 1.0
                elif points_colours.shape[1] == 4:
                    self._colours_cpu_swap[:n_points] = points_colours
                else:
                    raise ValueError(
                        f"points_colours must have 3 or 4 columns, "
                        f"got {points_colours.shape[1]}"
                    )
            else:
                # Default colour
                self._colours_cpu_swap[:n_points] = self.default_colour
            
            # Swap colour buffers
            self._colours_cpu, self._colours_cpu_swap = self._colours_cpu_swap, self._colours_cpu
            
            # Update colour buffer on GPU
            self._colours_buffer.data[:n_points] = self._colours_cpu[:n_points]
            self._colours_buffer.update_range(0, n_points)
        
        # Update GPU buffer with new position data
        # We update the underlying data array and mark the range as dirty
        self._positions_buffer.data[:n_points] = self._positions_cpu[:n_points]
        self._positions_buffer.update_range(0, n_points)
        
        # Update draw range to render only active points
        self._positions_buffer.draw_range = (0, n_points)
        
        # Track active count
        self._active_count = n_points
    
    def update_data_absolute(self, points_x, points_y, points_z, points_colours=None):
        """
        Update the scatterplot with new data using absolute coordinates.
        
        Unlike `update_data`, this method takes coordinates in the element's
        local coordinate system (centered at origin, ranging from 
        -half_size to +half_size for each axis).
        
        Parameters
        ----------
        points_x : array-like
            X coordinates centered at origin.
        points_y : array-like
            Y coordinates centered at origin.
        points_z : array-like
            Z coordinates centered at origin.
        points_colours : array-like, optional
            Per-point colours as (N, 4) RGBA or (N, 3) RGB array.
        """
        points_x = np.asarray(points_x, dtype=np.float32).ravel()
        points_y = np.asarray(points_y, dtype=np.float32).ravel()
        points_z = np.asarray(points_z, dtype=np.float32).ravel()
        
        n_points = len(points_x)
        if len(points_y) != n_points or len(points_z) != n_points:
            raise ValueError(
                f"points_x, points_y, and points_z must have same length, "
                f"got {n_points} vs {len(points_y)} vs {len(points_z)}"
            )
        
        # Check if we need to grow buffers
        if n_points > self._capacity:
            self._grow_buffers(n_points)
        
        # === Write directly into swap buffer (already in local coordinates) ===
        self._positions_cpu_swap[:n_points, 0] = points_x
        self._positions_cpu_swap[:n_points, 1] = points_y
        self._positions_cpu_swap[:n_points, 2] = points_z
        
        # Swap buffers
        self._positions_cpu, self._positions_cpu_swap = self._positions_cpu_swap, self._positions_cpu
        
        # Handle colours if in vertex mode
        if self.colour_mode == "vertex":
            if points_colours is not None:
                points_colours = np.asarray(points_colours, dtype=np.float32)
                if points_colours.ndim == 1:
                    self._colours_cpu_swap[:n_points] = points_colours
                elif points_colours.shape[0] != n_points:
                    raise ValueError(
                        f"points_colours must have {n_points} rows, "
                        f"got {points_colours.shape[0]}"
                    )
                elif points_colours.shape[1] == 3:
                    self._colours_cpu_swap[:n_points, :3] = points_colours
                    self._colours_cpu_swap[:n_points, 3] = 1.0
                elif points_colours.shape[1] == 4:
                    self._colours_cpu_swap[:n_points] = points_colours
                else:
                    raise ValueError(
                        f"points_colours must have 3 or 4 columns, "
                        f"got {points_colours.shape[1]}"
                    )
            else:
                self._colours_cpu_swap[:n_points] = self.default_colour
            
            self._colours_cpu, self._colours_cpu_swap = self._colours_cpu_swap, self._colours_cpu
            self._colours_buffer.data[:n_points] = self._colours_cpu[:n_points]
            self._colours_buffer.update_range(0, n_points)
        
        # Update GPU buffer with new position data
        self._positions_buffer.data[:n_points] = self._positions_cpu[:n_points]
        self._positions_buffer.update_range(0, n_points)
        self._positions_buffer.draw_range = (0, n_points)
        self._active_count = n_points
    
    def clear(self):
        """Clear all points from the scatterplot."""
        self._positions_buffer.draw_range = (0, 0)
        self._active_count = 0
    
    def set_point_size(self, size):
        """Update the point size."""
        self.point_size = size
        self._material.size = size
    
    def set_colour(self, colour):
        """
        Set uniform colour for all points (only works in uniform colour mode).
        
        Parameters
        ----------
        colour : tuple
            RGBA colour tuple.
        """
        if self.colour_mode != "uniform":
            raise ValueError(
                "set_colour only works in uniform colour mode. "
                "Use update_data with points_colours for vertex mode."
            )
        self.default_colour = colour
        self._material.color = colour
    
    @property
    def active_count(self):
        """Number of currently active (visible) points."""
        return self._active_count
    
    @property
    def capacity(self):
        """Current buffer capacity."""
        return self._capacity
    
    def move_to(self, new_center_xyz_px):
        """Move the scatterplot to a new center position."""
        super().move_to(new_center_xyz_px)
        # Also update the points object's position
        self._points.local.position = self.center
    
    def die(self):
        """Clean up GPU resources."""
        if hasattr(self, '_points'):
            self.unregister_gfx_object(self._points)
        super().die()


class Scatterplot3DDynamicMultiSeries(Element_3d):
    """
    A 3D scatterplot that can display multiple data series with different colours.
    
    Each series can be updated independently, and all series share the same
    coordinate space. The entire scatterplot can be rotated as a single unit.
    
    Parameters
    ----------
    unique_name : str
        Unique identifier for this element.
    parent : Element
        Parent element in the GUI hierarchy.
    bl_xyz_px : tuple
        Bottom-left-back corner position in absolute pixels.
    size_xyz_px : tuple
        Size in absolute pixels (width, height, depth).
    series_colours : list of tuple
        List of RGBA colours, one per series.
    initial_capacity_per_series : int
        Initial point capacity per series. Default 5000.
    growth_factor : float
        Growth factor when capacity is exceeded. Default 2.0.
    point_size : int
        Point size in pixels. Default 2.
    """
    
    def __init__(self, unique_name, parent, bl_xyz_px, size_xyz_px,
                 series_colours, initial_capacity_per_series=5_000,
                 growth_factor=2.0, point_size=2):
        super().__init__(unique_name, parent, bl_xyz_px, size_xyz_px)
        
        self.series_colours = series_colours
        self.n_series = len(series_colours)
        self.point_size = point_size
        self.growth_factor = growth_factor
        
        # Create one internal series per colour
        self._series = []
        for i, colour in enumerate(series_colours):
            series = _InternalScatterSeries3D(
                capacity=initial_capacity_per_series,
                growth_factor=growth_factor,
                colour=colour,
                point_size=point_size,
                parent_element=self
            )
            self._series.append(series)
    
    def update_series(self, series_index, points_x, points_y, points_z):
        """
        Update a specific series with new data.
        
        Coordinates are expected in [0, 1] range and will be mapped to 
        the element's local coordinate system.
        
        Parameters
        ----------
        series_index : int
            Index of series to update (0 to n_series-1).
        points_x : array-like
            X coordinates in [0, 1] range.
        points_y : array-like
            Y coordinates in [0, 1] range.
        points_z : array-like
            Z coordinates in [0, 1] range.
        """
        if series_index < 0 or series_index >= self.n_series:
            raise ValueError(
                f"series_index must be 0-{self.n_series-1}, got {series_index}"
            )
        self._series[series_index].update_data(points_x, points_y, points_z)
    
    def clear_series(self, series_index):
        """Clear a specific series."""
        if series_index < 0 or series_index >= self.n_series:
            raise ValueError(
                f"series_index must be 0-{self.n_series-1}, got {series_index}"
            )
        self._series[series_index].clear()
    
    def clear_all(self):
        """Clear all series."""
        for series in self._series:
            series.clear()
    
    def die(self):
        """Clean up all series."""
        for series in self._series:
            series.die()
        self._series.clear()
        super().die()


class _InternalScatterSeries3D:
    """
    Internal helper class for a single series within Scatterplot3DDynamicMultiSeries.
    Not intended for direct use.
    """
    
    def __init__(self, capacity, growth_factor, colour, point_size, parent_element):
        self._capacity = capacity
        self._growth_factor = growth_factor
        self._colour = colour
        self._point_size = point_size
        self._parent = parent_element
        self._active_count = 0
        
        # CPU buffers (double-buffered)
        self._positions_cpu = np.zeros((capacity, 3), dtype=np.float32)
        self._positions_cpu_swap = np.zeros((capacity, 3), dtype=np.float32)
        
        # Create GPU resources
        self._create_gpu_resources()
    
    def _create_gpu_resources(self):
        self._positions_buffer = pygfx.Buffer(
            data=self._positions_cpu,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE,
        )
        self._geometry = pygfx.Geometry(positions=self._positions_buffer)
        self._material = pygfx.PointsMaterial(
            size=self._point_size,
            color=self._colour
        )
        self._positions_buffer.draw_range = (0, 0)
        self._points = pygfx.Points(self._geometry, self._material)
        
        # Position at parent's center for rotation support
        self._points.local.position = self._parent.center
        
        self._parent.register_gfx_object(self._points)
    
    def _grow_buffers(self, required_capacity):
        new_capacity = self._capacity
        while new_capacity < required_capacity:
            new_capacity = max(int(new_capacity * self._growth_factor), new_capacity + 1)
        
        new_positions_cpu = np.zeros((new_capacity, 3), dtype=np.float32)
        new_positions_cpu_swap = np.zeros((new_capacity, 3), dtype=np.float32)
        
        if self._active_count > 0:
            new_positions_cpu[:self._active_count] = self._positions_cpu[:self._active_count]
        
        self._positions_cpu = new_positions_cpu
        self._positions_cpu_swap = new_positions_cpu_swap
        self._capacity = new_capacity
        
        self._parent.unregister_gfx_object(self._points)
        self._create_gpu_resources()
        
        if self._active_count > 0:
            self._positions_buffer.update_range(0, self._active_count)
            self._positions_buffer.draw_range = (0, self._active_count)
    
    def update_data(self, points_x, points_y, points_z):
        """Update with [0,1] normalized coordinates, mapped to local space."""
        points_x = np.asarray(points_x, dtype=np.float32).ravel()
        points_y = np.asarray(points_y, dtype=np.float32).ravel()
        points_z = np.asarray(points_z, dtype=np.float32).ravel()
        n_points = len(points_x)
        
        if n_points > self._capacity:
            self._grow_buffers(n_points)
        
        # Get parent dimensions (half sizes for centering)
        hw = self._parent.size[0] / 2
        hh = self._parent.size[1] / 2
        hd = self._parent.size[2] / 2
        
        # Map [0, 1] to [-half_size, +half_size] (centered at origin)
        self._positions_cpu_swap[:n_points, 0] = (points_x * 2.0 - 1.0) * hw
        self._positions_cpu_swap[:n_points, 1] = (points_y * 2.0 - 1.0) * hh
        self._positions_cpu_swap[:n_points, 2] = (points_z * 2.0 - 1.0) * hd
        
        self._positions_cpu, self._positions_cpu_swap = self._positions_cpu_swap, self._positions_cpu
        self._positions_buffer.data[:n_points] = self._positions_cpu[:n_points]
        self._positions_buffer.update_range(0, n_points)
        self._positions_buffer.draw_range = (0, n_points)
        self._active_count = n_points
    
    def clear(self):
        self._positions_buffer.draw_range = (0, 0)
        self._active_count = 0
    
    def die(self):
        self._parent.unregister_gfx_object(self._points)
