"""Dynamic line segments collection with efficient GPU buffer management.

This module provides a collection of line segments that can dynamically grow,
following the same efficient buffer management pattern as Scatterplot2DDynamic.
"""

from GUI.engine.frontend.graphical_elements.graphical_element import Element_2d

import wgpu
import pygfx
import numpy as np


class LinesDynamic(Element_2d):
    """
    A 2D element displaying a dynamic collection of line segments.
    
    Each line segment is defined by two endpoints (start and end).
    Lines can be added, modified, and the collection can grow dynamically.
    
    Key efficiency features:
    - Pre-allocated GPU buffers that grow geometrically when needed
    - No re-allocation on every update: reuses existing buffers when possible
    - Efficient partial buffer updates via `update_range`
    - Uses draw_range to render only the active subset of line segments
    - Double-buffered CPU arrays to avoid allocation overhead
    
    Buffer layout:
    - Each line segment requires 2 consecutive vertices in the buffer
    - Line i uses positions[2*i] (start) and positions[2*i+1] (end)
    - Total vertex count = 2 * number of lines
    
    Parameters
    ----------
    unique_name : str
        Unique identifier for this element.
    parent : Element
        Parent element in the GUI hierarchy.
    bl_xy_rel : tuple
        Bottom-left corner position relative to parent (0-1 range).
    size_xy_rel : tuple
        Size relative to parent (0-1 range).
    initial_capacity : int
        Initial number of line segments the buffer can hold. Default 10000.
        (Buffer will hold 2 * initial_capacity vertices)
    growth_factor : float
        Factor by which to grow buffers when capacity is exceeded. Default 1.5.
    thickness : float
        Line thickness in pixels. Default 1.0.
    default_colour : tuple
        Default RGBA colour for lines. Default (1, 0.5, 0.1, 1) (orange).
    normalize_coords : bool
        If True, normalize input coordinates to fit within element bounds.
        If False, use coordinates as-is (useful for pre-normalized data).
    colour_mode : str
        "uniform" for single colour for all lines, "segment" for per-segment colours.
    """
    
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel,
                 initial_capacity=10_000, growth_factor=1.5,
                 thickness=1.0, default_colour=(1, 0.5, 0.1, 1),
                 normalize_coords=False, colour_mode="uniform", ignore_pointmode=False):
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, ignore_pointmode=ignore_pointmode)
        
        self.initial_capacity = initial_capacity
        self.growth_factor = growth_factor
        self.thickness = thickness
        self.default_colour = default_colour
        self.normalize_coords = normalize_coords
        self.colour_mode = colour_mode  # "uniform" or "segment"
        
        # Current number of active line segments
        self._active_count = 0
        # Current buffer capacity (in number of line segments)
        self._capacity = initial_capacity
        
        # === CPU-side double buffers for positions (avoid allocations) ===
        # Each line segment needs 2 vertices, so buffer size = 2 * capacity
        self._positions_cpu = np.zeros((self._capacity * 2, 3), dtype=np.float32)
        self._positions_cpu_swap = np.zeros((self._capacity * 2, 3), dtype=np.float32)
        
        # === CPU-side buffer for colours (only if segment colour mode) ===
        # In segment mode, each vertex can have its own colour (2 colours per segment)
        if self.colour_mode == "segment":
            self._colours_cpu = np.zeros((self._capacity * 2, 4), dtype=np.float32)
            self._colours_cpu_swap = np.zeros((self._capacity * 2, 4), dtype=np.float32)
        
        # === GPU Buffers ===
        self._create_gpu_resources()
    
    def _create_gpu_resources(self):
        """Create or recreate GPU buffers with current capacity."""
        # Position buffer on GPU
        self._positions_buffer = pygfx.Buffer(
            data=self._positions_cpu,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE,
        )
        
        # Colours buffer (only if using segment colours)
        if self.colour_mode == "segment":
            self._colours_buffer = pygfx.Buffer(
                data=self._colours_cpu,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE,
            )
            self._geometry = pygfx.Geometry(
                positions=self._positions_buffer,
                colors=self._colours_buffer
            )
            self._material = pygfx.LineSegmentMaterial(
                thickness=self.thickness,
                color_mode="vertex",
                aa=True
            )
        else:
            self._geometry = pygfx.Geometry(positions=self._positions_buffer)
            self._material = pygfx.LineSegmentMaterial(
                thickness=self.thickness,
                color=self.default_colour,
                aa=True
            )
        
        # Set initial draw range to 0 (nothing to draw yet)
        # draw_range is in vertices, so 0 lines = 0 vertices
        self._positions_buffer.draw_range = (0, 0)
        
        # Create Line object with segment material (pairs of vertices form segments)
        self._lines = pygfx.Line(self._geometry, self._material)
        self.register_gfx_object(self._lines)
    
    def _grow_buffers(self, required_capacity):
        """
        Grow all buffers to accommodate at least `required_capacity` line segments.
        Uses geometric growth to amortize reallocation cost.
        """
        new_capacity = self._capacity
        while new_capacity < required_capacity:
            new_capacity = max(int(new_capacity * self.growth_factor), new_capacity + 1)
        
        # Allocate new CPU arrays (2 vertices per line)
        new_positions_cpu = np.zeros((new_capacity * 2, 3), dtype=np.float32)
        new_positions_cpu_swap = np.zeros((new_capacity * 2, 3), dtype=np.float32)
        
        # Copy existing data
        if self._active_count > 0:
            n_vertices = self._active_count * 2
            new_positions_cpu[:n_vertices] = self._positions_cpu[:n_vertices]
        
        self._positions_cpu = new_positions_cpu
        self._positions_cpu_swap = new_positions_cpu_swap
        
        if self.colour_mode == "segment":
            new_colours_cpu = np.zeros((new_capacity * 2, 4), dtype=np.float32)
            new_colours_cpu_swap = np.zeros((new_capacity * 2, 4), dtype=np.float32)
            if self._active_count > 0:
                n_vertices = self._active_count * 2
                new_colours_cpu[:n_vertices] = self._colours_cpu[:n_vertices]
            self._colours_cpu = new_colours_cpu
            self._colours_cpu_swap = new_colours_cpu_swap
        
        self._capacity = new_capacity
        
        # Recreate GPU resources with new capacity
        self.unregister_gfx_object(self._lines)
        self._create_gpu_resources()
        
        # If we had active lines, update the GPU with their data
        if self._active_count > 0:
            n_vertices = self._active_count * 2
            self._positions_buffer.update_range(0, n_vertices)
            self._positions_buffer.draw_range = (0, n_vertices)
            if self.colour_mode == "segment":
                self._colours_buffer.update_range(0, n_vertices)
    
    def update_data(self, starts_x, starts_y, ends_x, ends_y, colours=None):
        """
        Update the line collection with new data (replaces all existing lines).
        
        This is the main method for updating the visualization efficiently.
        It reuses existing buffers when possible and only grows them when needed.
        
        Parameters
        ----------
        starts_x : array-like
            X coordinates of line start points (1D array of length N).
        starts_y : array-like
            Y coordinates of line start points (1D array of length N).
        ends_x : array-like
            X coordinates of line end points (1D array of length N).
        ends_y : array-like
            Y coordinates of line end points (1D array of length N).
        colours : array-like, optional
            Per-segment colours as (N, 4) RGBA array or (N, 3) RGB array.
            Only used if colour_mode="segment". If None and colour_mode="segment",
            uses the default colour for all segments.
            
        Notes
        -----
        - If N > current capacity, buffers will be grown (geometric growth).
        - If N < current capacity, extra buffer space is simply not rendered.
        - Coordinates are normalized to fit within element bounds if normalize_coords=True.
        """
        starts_x = np.asarray(starts_x, dtype=np.float32).ravel()
        starts_y = np.asarray(starts_y, dtype=np.float32).ravel()
        ends_x = np.asarray(ends_x, dtype=np.float32).ravel()
        ends_y = np.asarray(ends_y, dtype=np.float32).ravel()
        
        n_lines = len(starts_x)
        if len(starts_y) != n_lines or len(ends_x) != n_lines or len(ends_y) != n_lines:
            raise ValueError(f"All coordinate arrays must have the same length, got {n_lines}, {len(starts_y)}, {len(ends_x)}, {len(ends_y)}")
        
        # Check if we need to grow buffers
        if n_lines > self._capacity:
            self._grow_buffers(n_lines)
        
        # Get element bounds for coordinate transformation
        sz = self.sz
        bl = self.bl
        n_vertices = n_lines * 2
        
        # === Write into swap buffer, then swap (avoids in-place issues) ===
        if self.normalize_coords and n_lines > 0:
            # Find global min/max across all coordinates
            all_x = np.concatenate([starts_x, ends_x])
            all_y = np.concatenate([starts_y, ends_y])
            x_min, x_max = all_x.min(), all_x.max()
            y_min, y_max = all_y.min(), all_y.max()
            
            # Avoid division by zero
            x_range = x_max - x_min
            y_range = y_max - y_min
            if x_range < 1e-8:
                x_range = 1.0
            if y_range < 1e-8:
                y_range = 1.0
            
            # Normalize and scale start points (even indices)
            self._positions_cpu_swap[0:n_vertices:2, 0] = ((starts_x - x_min) / x_range) * sz[0] + bl[0]
            self._positions_cpu_swap[0:n_vertices:2, 1] = ((starts_y - y_min) / y_range) * sz[1] + bl[1]
            
            # Normalize and scale end points (odd indices)
            self._positions_cpu_swap[1:n_vertices:2, 0] = ((ends_x - x_min) / x_range) * sz[0] + bl[0]
            self._positions_cpu_swap[1:n_vertices:2, 1] = ((ends_y - y_min) / y_range) * sz[1] + bl[1]
        else:
            # Use coordinates as-is (assume pre-normalized or absolute)
            self._positions_cpu_swap[0:n_vertices:2, 0] = starts_x * sz[0] + bl[0]
            self._positions_cpu_swap[0:n_vertices:2, 1] = starts_y * sz[1] + bl[1]
            self._positions_cpu_swap[1:n_vertices:2, 0] = ends_x * sz[0] + bl[0]
            self._positions_cpu_swap[1:n_vertices:2, 1] = ends_y * sz[1] + bl[1]
        
        # Z coordinate (depth) from element's bottom-left z
        self._positions_cpu_swap[:n_vertices, 2] = bl[2]
        
        # Swap buffers
        self._positions_cpu, self._positions_cpu_swap = self._positions_cpu_swap, self._positions_cpu
        
        # Handle colours if in segment mode
        if self.colour_mode == "segment":
            if colours is not None:
                colours = np.asarray(colours, dtype=np.float32)
                if colours.ndim == 1 and len(colours) in (3, 4):
                    # Single colour for all segments - broadcast to both vertices
                    if len(colours) == 3:
                        self._colours_cpu_swap[0:n_vertices:2, :3] = colours
                        self._colours_cpu_swap[0:n_vertices:2, 3] = 1.0
                        self._colours_cpu_swap[1:n_vertices:2, :3] = colours
                        self._colours_cpu_swap[1:n_vertices:2, 3] = 1.0
                    else:
                        self._colours_cpu_swap[0:n_vertices:2] = colours
                        self._colours_cpu_swap[1:n_vertices:2] = colours
                elif colours.shape[0] != n_lines:
                    raise ValueError(f"colours must have {n_lines} rows, got {colours.shape[0]}")
                elif colours.shape[1] == 3:
                    # RGB -> RGBA (add alpha = 1), broadcast to both vertices
                    self._colours_cpu_swap[0:n_vertices:2, :3] = colours
                    self._colours_cpu_swap[0:n_vertices:2, 3] = 1.0
                    self._colours_cpu_swap[1:n_vertices:2, :3] = colours
                    self._colours_cpu_swap[1:n_vertices:2, 3] = 1.0
                elif colours.shape[1] == 4:
                    # RGBA, broadcast to both vertices
                    self._colours_cpu_swap[0:n_vertices:2] = colours
                    self._colours_cpu_swap[1:n_vertices:2] = colours
                else:
                    raise ValueError(f"colours must have 3 or 4 columns, got {colours.shape[1]}")
            else:
                # Default colour for both vertices of each segment
                self._colours_cpu_swap[0:n_vertices:2] = self.default_colour
                self._colours_cpu_swap[1:n_vertices:2] = self.default_colour
            
            # Swap colour buffers
            self._colours_cpu, self._colours_cpu_swap = self._colours_cpu_swap, self._colours_cpu
            
            # Update colour buffer on GPU
            self._colours_buffer.data[:n_vertices] = self._colours_cpu[:n_vertices]
            self._colours_buffer.update_range(0, n_vertices)
        
        # Update GPU buffer with new position data
        self._positions_buffer.data[:n_vertices] = self._positions_cpu[:n_vertices]
        self._positions_buffer.update_range(0, n_vertices)
        
        # Update draw range to render only active vertices
        self._positions_buffer.draw_range = (0, n_vertices)
        
        # Track active count (in lines, not vertices)
        self._active_count = n_lines
    
    def set_line(self, index, start_x, start_y, end_x, end_y, colour=None):
        """
        Set or update a single line segment at the given index.
        
        Parameters
        ----------
        index : int
            Line segment index (0-based). If >= current active count, 
            the active count is extended.
        start_x, start_y : float
            Start point coordinates (normalized 0-1 or absolute based on normalize_coords).
        end_x, end_y : float
            End point coordinates.
        colour : tuple, optional
            RGBA colour for this segment (only used in segment colour mode).
        """
        # Grow if needed
        if index >= self._capacity:
            self._grow_buffers(index + 1)
        
        sz = self.sz
        bl = self.bl
        
        # Calculate vertex indices
        v0 = index * 2      # start vertex
        v1 = index * 2 + 1  # end vertex
        
        # Set positions (no normalization for single-line updates)
        self._positions_cpu[v0, 0] = start_x * sz[0] + bl[0]
        self._positions_cpu[v0, 1] = start_y * sz[1] + bl[1]
        self._positions_cpu[v0, 2] = bl[2]
        
        self._positions_cpu[v1, 0] = end_x * sz[0] + bl[0]
        self._positions_cpu[v1, 1] = end_y * sz[1] + bl[1]
        self._positions_cpu[v1, 2] = bl[2]
        
        # Update GPU
        self._positions_buffer.data[v0:v1+1] = self._positions_cpu[v0:v1+1]
        self._positions_buffer.update_range(v0, 2)
        
        # Handle colour
        if self.colour_mode == "segment" and colour is not None:
            colour = np.asarray(colour, dtype=np.float32)
            if len(colour) == 3:
                self._colours_cpu[v0, :3] = colour
                self._colours_cpu[v0, 3] = 1.0
                self._colours_cpu[v1, :3] = colour
                self._colours_cpu[v1, 3] = 1.0
            else:
                self._colours_cpu[v0] = colour
                self._colours_cpu[v1] = colour
            self._colours_buffer.data[v0:v1+1] = self._colours_cpu[v0:v1+1]
            self._colours_buffer.update_range(v0, 2)
        
        # Extend active count if needed
        if index >= self._active_count:
            self._active_count = index + 1
            self._positions_buffer.draw_range = (0, self._active_count * 2)
    
    def add_line(self, start_x, start_y, end_x, end_y, colour=None):
        """
        Add a new line segment at the end of the collection.
        
        Parameters
        ----------
        start_x, start_y : float
            Start point coordinates.
        end_x, end_y : float
            End point coordinates.
        colour : tuple, optional
            RGBA colour for this segment (only used in segment colour mode).
            
        Returns
        -------
        int
            Index of the newly added line segment.
        """
        new_index = self._active_count
        self.set_line(new_index, start_x, start_y, end_x, end_y, colour)
        return new_index
    
    def update_line_positions(self, indices, starts_x, starts_y, ends_x, ends_y):
        """
        Update positions of multiple existing line segments efficiently.
        
        Parameters
        ----------
        indices : array-like
            Indices of line segments to update.
        starts_x, starts_y : array-like
            New start point coordinates.
        ends_x, ends_y : array-like
            New end point coordinates.
        """
        indices = np.asarray(indices, dtype=np.int32).ravel()
        starts_x = np.asarray(starts_x, dtype=np.float32).ravel()
        starts_y = np.asarray(starts_y, dtype=np.float32).ravel()
        ends_x = np.asarray(ends_x, dtype=np.float32).ravel()
        ends_y = np.asarray(ends_y, dtype=np.float32).ravel()
        
        n = len(indices)
        if not (len(starts_x) == len(starts_y) == len(ends_x) == len(ends_y) == n):
            raise ValueError("All arrays must have the same length")
        
        if n == 0:
            return
        
        # Check capacity
        max_index = indices.max()
        if max_index >= self._capacity:
            self._grow_buffers(max_index + 1)
        
        sz = self.sz
        bl = self.bl
        
        # Calculate vertex indices
        v0_indices = indices * 2
        v1_indices = indices * 2 + 1
        
        # Update positions
        self._positions_cpu[v0_indices, 0] = starts_x * sz[0] + bl[0]
        self._positions_cpu[v0_indices, 1] = starts_y * sz[1] + bl[1]
        self._positions_cpu[v0_indices, 2] = bl[2]
        
        self._positions_cpu[v1_indices, 0] = ends_x * sz[0] + bl[0]
        self._positions_cpu[v1_indices, 1] = ends_y * sz[1] + bl[1]
        self._positions_cpu[v1_indices, 2] = bl[2]
        
        # Update GPU - find the range of affected vertices
        min_vertex = v0_indices.min()
        max_vertex = v1_indices.max()
        n_vertices_to_update = max_vertex - min_vertex + 1
        
        self._positions_buffer.data[min_vertex:max_vertex+1] = self._positions_cpu[min_vertex:max_vertex+1]
        self._positions_buffer.update_range(min_vertex, n_vertices_to_update)
        
        # Extend active count if needed
        if max_index >= self._active_count:
            self._active_count = max_index + 1
            self._positions_buffer.draw_range = (0, self._active_count * 2)
    
    def clear(self):
        """Clear all lines from the collection."""
        self._positions_buffer.draw_range = (0, 0)
        self._active_count = 0
    
    def set_thickness(self, thickness):
        """Update the line thickness."""
        self.thickness = thickness
        self._material.thickness = thickness
    
    def set_colour(self, colour):
        """
        Set uniform colour for all lines (only works in uniform colour mode).
        
        Parameters
        ----------
        colour : tuple
            RGBA colour tuple.
        """
        if self.colour_mode != "uniform":
            raise ValueError("set_colour only works in uniform colour mode. Use update_data with colours for segment mode.")
        self.default_colour = colour
        self._material.color = colour
    
    @property
    def active_count(self):
        """Number of currently active (visible) line segments."""
        return self._active_count
    
    @property
    def capacity(self):
        """Current buffer capacity (in line segments)."""
        return self._capacity
    
    def die(self):
        """Clean up GPU resources."""
        if hasattr(self, '_lines'):
            self.unregister_gfx_object(self._lines)
        super().die()


class LinesDynamicMultiSeries(Element_2d):
    """
    A 2D element for displaying multiple series of line segments with different colours.
    
    Each series can be updated independently, and all series share the same
    coordinate space.
    
    Parameters
    ----------
    unique_name : str
        Unique identifier for this element.
    parent : Element
        Parent element in the GUI hierarchy.
    bl_xy_rel : tuple
        Bottom-left corner position relative to parent (0-1 range).
    size_xy_rel : tuple
        Size relative to parent (0-1 range).
    series_colours : list of tuple
        List of RGBA colours, one per series.
    initial_capacity_per_series : int
        Initial line segment capacity per series. Default 5000.
    growth_factor : float
        Growth factor when capacity is exceeded. Default 1.5.
    thickness : float
        Line thickness in pixels. Default 1.0.
    """
    
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel,
                 series_colours, initial_capacity_per_series=5_000,
                 growth_factor=1.5, thickness=1.0, ignore_pointmode=False):
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, ignore_pointmode=ignore_pointmode)
        
        self.series_colours = series_colours
        self.n_series = len(series_colours)
        self.thickness = thickness
        self.growth_factor = growth_factor
        
        # Create one lines collection per series
        self._series = []
        for i, colour in enumerate(series_colours):
            series = _InternalLinesSeries(
                capacity=initial_capacity_per_series,
                growth_factor=growth_factor,
                colour=colour,
                thickness=thickness,
                parent_element=self
            )
            self._series.append(series)
    
    def update_series(self, series_index, starts_x, starts_y, ends_x, ends_y,
                      x_bounds=None, y_bounds=None):
        """
        Update a specific series with new data.
        
        Parameters
        ----------
        series_index : int
            Index of series to update (0 to n_series-1).
        starts_x, starts_y : array-like
            Start point coordinates.
        ends_x, ends_y : array-like
            End point coordinates.
        x_bounds : tuple, optional
            (x_min, x_max) for normalization. If None, uses data bounds.
        y_bounds : tuple, optional
            (y_min, y_max) for normalization. If None, uses data bounds.
        """
        if series_index < 0 or series_index >= self.n_series:
            raise ValueError(f"series_index must be 0-{self.n_series-1}, got {series_index}")
        
        self._series[series_index].update_data(
            starts_x, starts_y, ends_x, ends_y, x_bounds, y_bounds
        )
    
    def clear_series(self, series_index):
        """Clear a specific series."""
        if series_index < 0 or series_index >= self.n_series:
            raise ValueError(f"series_index must be 0-{self.n_series-1}, got {series_index}")
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


class _InternalLinesSeries:
    """
    Internal helper class for a single series within LinesDynamicMultiSeries.
    Not intended for direct use.
    """
    
    def __init__(self, capacity, growth_factor, colour, thickness, parent_element):
        self._capacity = capacity
        self._growth_factor = growth_factor
        self._colour = colour
        self._thickness = thickness
        self._parent = parent_element
        self._active_count = 0
        
        # CPU buffers (2 vertices per line segment)
        self._positions_cpu = np.zeros((capacity * 2, 3), dtype=np.float32)
        self._positions_cpu_swap = np.zeros((capacity * 2, 3), dtype=np.float32)
        
        # Create GPU resources
        self._create_gpu_resources()
    
    def _create_gpu_resources(self):
        self._positions_buffer = pygfx.Buffer(
            data=self._positions_cpu,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE,
        )
        self._geometry = pygfx.Geometry(positions=self._positions_buffer)
        self._material = pygfx.LineSegmentMaterial(
            thickness=self._thickness,
            color=self._colour,
            aa=True
        )
        self._positions_buffer.draw_range = (0, 0)
        self._lines = pygfx.Line(self._geometry, self._material)
        self._parent.register_gfx_object(self._lines)
    
    def _grow_buffers(self, required_capacity):
        new_capacity = self._capacity
        while new_capacity < required_capacity:
            new_capacity = max(int(new_capacity * self._growth_factor), new_capacity + 1)
        
        new_positions_cpu = np.zeros((new_capacity * 2, 3), dtype=np.float32)
        new_positions_cpu_swap = np.zeros((new_capacity * 2, 3), dtype=np.float32)
        
        if self._active_count > 0:
            n_vertices = self._active_count * 2
            new_positions_cpu[:n_vertices] = self._positions_cpu[:n_vertices]
        
        self._positions_cpu = new_positions_cpu
        self._positions_cpu_swap = new_positions_cpu_swap
        self._capacity = new_capacity
        
        self._parent.unregister_gfx_object(self._lines)
        self._create_gpu_resources()
        
        if self._active_count > 0:
            n_vertices = self._active_count * 2
            self._positions_buffer.update_range(0, n_vertices)
            self._positions_buffer.draw_range = (0, n_vertices)
    
    def update_data(self, starts_x, starts_y, ends_x, ends_y, x_bounds=None, y_bounds=None):
        starts_x = np.asarray(starts_x, dtype=np.float32).ravel()
        starts_y = np.asarray(starts_y, dtype=np.float32).ravel()
        ends_x = np.asarray(ends_x, dtype=np.float32).ravel()
        ends_y = np.asarray(ends_y, dtype=np.float32).ravel()
        n_lines = len(starts_x)
        n_vertices = n_lines * 2
        
        if n_lines > self._capacity:
            self._grow_buffers(n_lines)
        
        sz = self._parent.sz
        bl = self._parent.bl
        
        if n_lines > 0:
            # Gather all coords for normalization
            all_x = np.concatenate([starts_x, ends_x])
            all_y = np.concatenate([starts_y, ends_y])
            
            if x_bounds is not None:
                x_min, x_max = x_bounds
            else:
                x_min, x_max = all_x.min(), all_x.max()
            
            if y_bounds is not None:
                y_min, y_max = y_bounds
            else:
                y_min, y_max = all_y.min(), all_y.max()
            
            x_range = max(x_max - x_min, 1e-8)
            y_range = max(y_max - y_min, 1e-8)
            
            # Start vertices (even indices)
            self._positions_cpu_swap[0:n_vertices:2, 0] = ((starts_x - x_min) / x_range) * sz[0] + bl[0]
            self._positions_cpu_swap[0:n_vertices:2, 1] = ((starts_y - y_min) / y_range) * sz[1] + bl[1]
            self._positions_cpu_swap[0:n_vertices:2, 2] = bl[2]
            
            # End vertices (odd indices)
            self._positions_cpu_swap[1:n_vertices:2, 0] = ((ends_x - x_min) / x_range) * sz[0] + bl[0]
            self._positions_cpu_swap[1:n_vertices:2, 1] = ((ends_y - y_min) / y_range) * sz[1] + bl[1]
            self._positions_cpu_swap[1:n_vertices:2, 2] = bl[2]
        
        self._positions_cpu, self._positions_cpu_swap = self._positions_cpu_swap, self._positions_cpu
        self._positions_buffer.data[:n_vertices] = self._positions_cpu[:n_vertices]
        self._positions_buffer.update_range(0, n_vertices)
        self._positions_buffer.draw_range = (0, n_vertices)
        self._active_count = n_lines
    
    def clear(self):
        self._positions_buffer.draw_range = (0, 0)
        self._active_count = 0
    
    def die(self):
        self._parent.unregister_gfx_object(self._lines)
