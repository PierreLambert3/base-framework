"""
Points-mode renderer.

When `theme.POINTS_MODE` is set, GUI line allocations are redirected here instead of being drawn as `pygfx.Line` objects.
Each registered line owns a slice of a single per-page `Scatterplot3DDynamic`. 
A CUDA kernel (`kernels/points_mode_motion.cu`) advances the points each frame so they drift around their assigned line.

Positions and line endpoints are stored in page-local frame (scene − center).
`_pos_host` and `_colour_host` alias the scatter's staging buffers directly;
`to_host` writes positions straight into `_pos_host` with no intermediate copy.
Rebind both after any `_grow_points()` call (scatter recreates its buffer arrays).

`LinesDynamic` is not routed through this manager.
"""

from __future__ import annotations

import math

import numpy as np

from GUI.engine.frontend.graphical_elements.scatterplot_3d_dynamic import Scatterplot3DDynamic
from GUI.engine.frontend.theme import (
    ORANGE_YELLOW, ORANGE_DARK, ORANGE_WHITE, PURPLE_DARK,
    interpolate_color, to_rgba,
)

from cuda_wrapper.helpers import generate_uint32_seed, next_seed_uint32


POINTS_PER_0_1_OF_DIAGONAL = 28
MIN_POINTS_PER_LINE        = 1

POINT_SIZE_PX              = 1

GROWTH_FACTOR              = 1.2

# Column indices into the packed (N, 5) per-point modulation array.
MOD_SPRING  = 0
MOD_JITTER  = 1
MOD_DT      = 2
MOD_DAMPING = 3
MOD_UPWARDS = 4
N_MOD_COLS  = 5

# Default `point_mods` bundle. Reproduces the legacy (unmodulated) behaviour
# when no `point_mods` argument is supplied to `register_lines`.
DEFAULT_POINT_MODS = {
    "n_points_mul":              1.0,
    "colour_range":              None,       # None -> use the explicit `colour` argument uniformly
    "spring_strength":           (1.0, 0.0), # (mu, std)
    "jitter_strength":           (1.0, 0.0),
    "dt":                        (1.0, 0.3),
    "damping":                   (1.0, 0.0),
    "line_upwards_interaction":  (0.0, 0.0), # 0 -> disabled; + attracts from normal side, - from opposite
}

_rng = np.random.default_rng()


def _resolve_point_mods(point_mods):
    """Merge a user-supplied `point_mods` dict on top of defaults."""
    if point_mods is None:
        return dict(DEFAULT_POINT_MODS)
    merged = dict(DEFAULT_POINT_MODS)
    merged.update(point_mods)
    return merged


def _sample_mod_column(spec, n: int) -> np.ndarray:
    """Sample n floats from N(mu, std) where spec = (mu, std)."""
    mu, std = float(spec[0]), float(spec[1])
    if std == 0.0:
        return np.full(n, mu, dtype=np.float32)
    return np.clip(mu + std * _rng.standard_normal(n), 0.5 * mu, 1.5 * mu).astype(np.float32)


def _sample_colours_range(colour_range, n: int) -> np.ndarray:
    """Sample n RGBA rows via c1 + (c2 - c1) * U[0,1]**2."""
    c1 = np.array(_to_rgba(colour_range[0]), dtype=np.float32)
    c2 = np.array(_to_rgba(colour_range[1]), dtype=np.float32)
    t  = (_rng.random(n).astype(np.float32)) ** 2
    return c1[None, :] + (c2 - c1)[None, :] * t[:, None]


def _to_rgba(c) -> tuple:
    if c is None:
        return to_rgba(interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5))
    if isinstance(c, str):
        return to_rgba(c)
    return (float(c[0]), float(c[1]), float(c[2]), 1.0 if len(c) == 3 else float(c[3]))


class LinesHandle:
    """Opaque handle returned by `PointsModeManager.register_lines`.

    Each entry is [line_index, slot_start, slot_count]. Lists are mutated in
    place by the manager during removal (defragmentation), so callers must not
    cache `entries` snapshots.
    """

    __slots__ = ("_manager", "entries", "_alive", "_base_mods_spec")

    def __init__(self, manager: "PointsModeManager"):
        self._manager       = manager
        self.entries        = []
        self._alive         = True
        self._base_mods_spec = None

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def n_segments(self) -> int:
        return len(self.entries)

    def change_line(self, i: int, new_start, new_end):
        if not self._alive:
            return
        self._manager._update_line_endpoints(self.entries[i][0], new_start, new_end)

    def change_lines(self, pairs):
        for i, (s, e) in enumerate(pairs):
            self.change_line(i, s, e)

    def set_colour(self, colour):
        # Uniform overwrite: this discards any per-point randomization that
        # may have been written by a `pointMode_colour_range` argument.
        if not self._alive:
            return
        rgba = _to_rgba(colour)
        for line_index, slot_start, slot_count in self.entries:
            self._manager._update_line_colour(line_index, slot_start, slot_count, rgba)

    def set_point_mods(self, spring_strength=None, jitter_strength=None,
                       dt=None, damping=None, colour_range=None,
                       line_upwards_interaction=None):
        """Re-sample per-point modulation values for this handle's points.

        Any argument left as `None` leaves the corresponding channel
        untouched. `spring_strength` / `jitter_strength` / `dt` / `damping` /
        `line_upwards_interaction` each take a `(mu, std)` tuple. `colour_range`
        takes a pair of colours and regenerates per-point colours via
        `c1 + (c2 - c1) * U**2`.
        A single GPU upload is performed at the end regardless of how many
        segments the handle owns.
        """
        if not self._alive:
            return
        self._manager._update_point_mods(
            self,
            spring_strength=spring_strength,
            jitter_strength=jitter_strength,
            dt=dt,
            damping=damping,
            colour_range=colour_range,
            line_upwards_interaction=line_upwards_interaction,
        )

    def restore_mod(self, *col_names: str):
        """Restore one or more per-point modulation columns to their base spec.

        Each `col_name` must be a keyword accepted by `set_point_mods`
        (e.g. ``"line_upwards_interaction"``, ``"jitter_strength"``,
        ``"colour_range"``).  All columns are restored in a single GPU upload.
        Values are re-sampled from the (mu, std) spec recorded at registration,
        so for std=0 (typical button case) the restore is exact.
        """
        if not self._alive or self._base_mods_spec is None:
            return
        kwargs = {name: self._base_mods_spec[name] for name in col_names}
        self.set_point_mods(**kwargs)

    def change_up_vector(self, i: int, new_looking_at):
        """Update the up-normal vector for segment i.

        `new_looking_at` is a world-space 3D location. The up vector is
        computed as `new_looking_at − midpoint(segment i)`, matching the
        convention used in `add_lines(looking_at_locations=...)`.
        """
        if not self._alive:
            return
        line_index = self.entries[i][0]
        # Reconstruct midpoint in world space from page-local stored endpoints
        row = self._manager._lines_xyzxyz[line_index]
        cx, cy, cz = (float(self._manager._center[0]),
                      float(self._manager._center[1]),
                      float(self._manager._center[2]))
        mid_world = np.array([
            (row[0] + row[3]) * 0.5 + cx,
            (row[1] + row[4]) * 0.5 + cy,
            (row[2] + row[5]) * 0.5 + cz,
        ], dtype=np.float32)
        la = np.array(new_looking_at, dtype=np.float32)
        up_vec = la - mid_world
        self._manager._update_line_normal(line_index, up_vec)

    def remove(self):
        if not self._alive:
            return
        self._manager._remove_handle(self)
        self._alive = False

class PointsModeManager:
    """One per Page. Owns a Scatterplot3DDynamic, parallel CUDA arrays for
    positions / velocities / line_idx / lines, and a registry of live handles.
    """

    def __init__(self, page):
        self.page     = page
        self.frontend = page.frontend

        sx, sy, sz = page.size[0], page.size[1], page.size[2]
        self._page_diagonal = math.sqrt(sx * sx + sy * sy + max(sz, 1.0) ** 2)
        self._center = np.array([page.center[0], page.center[1], page.center[2]], dtype=np.float32)

        page_bl = page.bottom_left
        self._scatter = Scatterplot3DDynamic(
            unique_name      = f"{page.name}::points_mode_scatter",
            parent           = page,
            bl_xyz_px        = (page_bl[0], page_bl[1], page_bl[2]),
            size_xyz_px      = (sx, sy, max(sz, 1.0)),
            initial_capacity = 1023,
            growth_factor    = GROWTH_FACTOR,
            point_size       = POINT_SIZE_PX,
            colour_mode      = "vertex",
        )

        self._lines_capacity = 64
        # Per-line data: [ax, ay, az, bx, by, bz, nx, ny, nz]  (stride 9; cols 6-8 = up-normal)
        self._lines_xyzxyz   = np.zeros((self._lines_capacity, 9), dtype=np.float32)
        self._line_colour     = np.zeros((self._lines_capacity, 4), dtype=np.float32)
        self._n_lines         = 0

        self._point_vel      = np.zeros((self._scatter._capacity, 3), dtype=np.float32)
        self._point_line_idx = np.zeros((self._scatter._capacity,),   dtype=np.int32)
        # Packed per-point modulation: columns [spring, jitter, dt, damping].
        # Initialized to 1.0 so unmodulated points pass through unchanged.
        self._point_mods     = np.ones((self._scatter._capacity, N_MOD_COLS), dtype=np.float32)
        self._n_points       = 0

        # _pos_host / _colour_host alias scatter's staging buffers; must be
        # rebound after any _grow_points() call.
        self._pos_host    = self._scatter._positions_buffer.data
        self._colour_host = self._scatter._colours_buffer.data
        self._colours_dirty = False

        self._handles: list[LinesHandle] = []

        self._stream1         = None
        self._stream2         = None
        self._positions_gpu  = None
        self._velocities_gpu = None
        self._line_idx_gpu   = None
        self._lines_gpu      = None
        self._mods_gpu       = None
        self._kernel         = None
        self._tick_phase     = False

        self._setup_cuda()
        self.cuda_seed = generate_uint32_seed()

    def register_lines(self, pairs, colour=None, point_mods=None, line_up_vectors=None) -> LinesHandle:
        rgba   = _to_rgba(colour)
        mods   = _resolve_point_mods(point_mods)
        handle = LinesHandle(self)
        handle._base_mods_spec = mods
        for idx, (s, e) in enumerate(pairs):
            up = line_up_vectors[idx] if (line_up_vectors is not None and idx < len(line_up_vectors)) else (0.5, 0.5, 0.0)
            self._append_segment(handle, s, e, rgba, mods, up_vec=up)
        self._handles.append(handle)
        self._sync_to_gpu()
        return handle

    def tick(self, dt: float, max_dt: float):
        if self._n_points == 0:
            return

        # 1. move the points (GPU) 
        # 2. GPU -> CPU copy of positions

        # dt = min(dt / max_dt, 1.0) * 0.1
        dt = 0.2

        self.cuda_seed = next_seed_uint32(self.cuda_seed)
        self._kernel.launch(
            self._stream1,
            self._positions_gpu,
            self._velocities_gpu,
            self._line_idx_gpu,
            self._lines_gpu,
            self._mods_gpu,
            np.uint32(self._n_points),
            np.float32(dt),
            np.uint32(self.cuda_seed),
        )
        self._positions_gpu.to_host(out=self._pos_host, stream=self._stream1)

        # update the visuals
        self._stream1.sync()
        n = self._n_points
        self._scatter._positions_buffer.update_range(0, n)
        self._scatter._positions_buffer.draw_range = (0, n)
        self._scatter._active_count = n
        if self._colours_dirty:
            self._scatter._colours_buffer.update_range(0, n)
            self._colours_dirty = False


        """ if not self._tick_phase:
            # 1. move the points (GPU) 
            # 2. GPU -> CPU copy of positions

            # dt = min(dt / max_dt, 1.0)
            dt = 1.0

            self._kernel.launch(
                self._stream1,
                self._positions_gpu,
                self._velocities_gpu,
                self._line_idx_gpu,
                self._lines_gpu,
                self._mods_gpu,
                np.uint32(self._n_points),
                np.float32(dt),
                np.float32(ATTRACTION_MULTIPLIER),
                np.float32(MOMENTUM),
                np.float32(self._noise_intensity),
                np.float32(ENDPOINTEDNESS),
            )
            self._positions_gpu.to_host(out=self._pos_host, stream=self._stream1)
        
        else:

            # update the visuals

            self._stream1.sync()
            n = self._n_points
            self._scatter._positions_buffer.update_range(0, n)
            self._scatter._positions_buffer.draw_range = (0, n)
            self._scatter._active_count = n
            if self._colours_dirty:
                self._scatter._colours_buffer.update_range(0, n)
                self._colours_dirty = False

        self._tick_phase = not self._tick_phase """

    def destroy(self):
        if self._positions_gpu is not None:
            ctx = self.frontend.cuda_ctx
            for arr in (self._positions_gpu, self._velocities_gpu,
                        self._line_idx_gpu, self._lines_gpu, self._mods_gpu):
                try:
                    ctx.free(arr)
                except Exception:
                    pass
        self._positions_gpu = self._velocities_gpu = None
        self._line_idx_gpu  = self._lines_gpu      = None
        self._mods_gpu      = None
        self._kernel        = None
        try:
            self._scatter.die()
        except Exception:
            pass
        for handle in self._handles:
            handle._alive = False
        self._handles.clear()
        self._n_lines  = 0
        self._n_points = 0

    def _append_segment(self, handle: LinesHandle, start, end, rgba: tuple, mods: dict,
                        up_vec=(0.2, 0.2, 0.0)):
        ax, ay, az = float(start[0]), float(start[1]), float(start[2])
        bx, by, bz = float(end[0]),   float(end[1]),   float(end[2])

        seg_len = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)
        rel     = seg_len / self._page_diagonal if self._page_diagonal > 0 else 0.0
        n_pts   = max(MIN_POINTS_PER_LINE, int(math.ceil(rel / 0.1)) * POINTS_PER_0_1_OF_DIAGONAL)
        # Apply per-allocation count multiplier, then re-floor.
        n_pts = max(MIN_POINTS_PER_LINE, int(round(n_pts * float(mods["n_points_mul"]))))

        if self._n_lines + 1 > self._lines_capacity:
            self._grow_lines_cpu(self._n_lines + 1)
        if self._n_points + n_pts > self._scatter._capacity:
            self._grow_points(self._n_points + n_pts)

        line_index = self._n_lines
        slot_start = self._n_points
        slot_count = n_pts

        cx, cy, cz = float(self._center[0]), float(self._center[1]), float(self._center[2])
        lax, lay, laz = ax - cx, ay - cy, az - cz
        lbx, lby, lbz = bx - cx, by - cy, bz - cz

        self._lines_xyzxyz[line_index, :6] = (lax, lay, laz, lbx, lby, lbz)
        self._lines_xyzxyz[line_index, 6:] = (float(up_vec[0]), float(up_vec[1]), float(up_vec[2]))
        self._line_colour[line_index]  = rgba

        ts = np.linspace(0.0, 1.0, n_pts, dtype=np.float32)
        sl = slice(slot_start, slot_start + slot_count)
        self._pos_host[sl, 0] = lax + ts * (lbx - lax) 
        self._pos_host[sl, 1] = lay + ts * (lby - lay) 
        self._pos_host[sl, 2] = laz + ts * (lbz - laz) 
        self._point_vel[sl]      = 0.0
        self._point_line_idx[sl] = line_index

        # Per-point colours: range overrides single colour when provided.
        colour_range = mods.get("colour_range")
        if colour_range is not None:
            self._colour_host[sl] = _sample_colours_range(colour_range, n_pts)
        else:
            self._colour_host[sl] = rgba
        self._colours_dirty = True

        # Per-point modulation columns.
        self._point_mods[sl, MOD_SPRING]  = _sample_mod_column(mods["spring_strength"], n_pts)
        self._point_mods[sl, MOD_JITTER]  = _sample_mod_column(mods["jitter_strength"], n_pts)
        self._point_mods[sl, MOD_DT]      = _sample_mod_column(mods["dt"], n_pts)
        self._point_mods[sl, MOD_DAMPING] = _sample_mod_column(mods["damping"], n_pts)
        self._point_mods[sl, MOD_UPWARDS] = _sample_mod_column(mods["line_upwards_interaction"], n_pts)

        handle.entries.append([line_index, slot_start, slot_count])
        self._n_lines  += 1
        self._n_points += slot_count

    def _update_line_endpoints(self, line_index: int, new_start, new_end):
        cx, cy, cz = float(self._center[0]), float(self._center[1]), float(self._center[2])
        self._lines_xyzxyz[line_index, :6] = (
            float(new_start[0]) - cx, float(new_start[1]) - cy, float(new_start[2]) - cz,
            float(new_end[0])   - cx, float(new_end[1])   - cy, float(new_end[2])   - cz,
        )
        if self._lines_gpu is not None:
            self._lines_gpu.copy_from(np.ascontiguousarray(self._lines_xyzxyz))

    def _update_line_normal(self, line_index: int, up_vec):
        """Normalises the up vector and Writes it (cols 6-8) for a registered line and upload to GPU."""
        up_vec = up_vec / (np.linalg.norm(up_vec) + 1e-12)
        self._lines_xyzxyz[line_index, 6:] = (float(up_vec[0]), float(up_vec[1]), float(up_vec[2]))
        if self._lines_gpu is not None:
            self._lines_gpu.copy_from(np.ascontiguousarray(self._lines_xyzxyz))

    def _update_line_colour(self, line_index: int, slot_start: int, slot_count: int, rgba: tuple):
        self._line_colour[line_index] = rgba
        self._colour_host[slot_start:slot_start + slot_count] = rgba
        self._colours_dirty = True

    def _remove_handle(self, handle: LinesHandle):
        if handle not in self._handles:
            print("--- Warning: attempt to remove LinesHandle not in manager registry ---")
            return
        entries_sorted = sorted(handle.entries, key=lambda e: e[1], reverse=True)
        for line_index, slot_start, slot_count in entries_sorted:
            self._compact_remove(line_index, slot_start, slot_count)
        self._handles.remove(handle)
        if self._positions_gpu is not None:
            self._positions_gpu.copy_from(self._pos_host)
            self._velocities_gpu.copy_from(self._point_vel)
            self._line_idx_gpu.copy_from(self._point_line_idx)
            self._lines_gpu.copy_from(np.ascontiguousarray(self._lines_xyzxyz))
            self._mods_gpu.copy_from(np.ascontiguousarray(self._point_mods))

    def _compact_remove(self, line_index: int, slot_start: int, slot_count: int):
        n_after = self._n_points - (slot_start + slot_count)
        if n_after > 0:
            sl_dst = slice(slot_start, slot_start + n_after)
            sl_src = slice(slot_start + slot_count, self._n_points)
            self._pos_host[sl_dst]       = self._pos_host[sl_src]
            self._point_vel[sl_dst]      = self._point_vel[sl_src]
            self._point_line_idx[sl_dst] = self._point_line_idx[sl_src]
            self._colour_host[sl_dst]    = self._colour_host[sl_src]
            self._point_mods[sl_dst]     = self._point_mods[sl_src]
        self._n_points -= slot_count
        self._colours_dirty = True

        n_lines_after = self._n_lines - (line_index + 1)
        if n_lines_after > 0:
            sl_dst = slice(line_index, line_index + n_lines_after)
            sl_src = slice(line_index + 1, self._n_lines)
            self._lines_xyzxyz[sl_dst] = self._lines_xyzxyz[sl_src]
            self._line_colour[sl_dst]  = self._line_colour[sl_src]
        self._n_lines -= 1

        if self._n_points > 0:
            mask = self._point_line_idx[:self._n_points] > line_index
            self._point_line_idx[:self._n_points][mask] -= 1

        for h in self._handles:
            for entry in h.entries:
                if entry[0] > line_index:
                    entry[0] -= 1
                if entry[1] > slot_start:
                    entry[1] -= slot_count

    def _grow_lines_cpu(self, required: int):
        new_capacity = self._lines_capacity
        while new_capacity < required:
            new_capacity = max(int(new_capacity * GROWTH_FACTOR), new_capacity + 1)
        new_lines  = np.zeros((new_capacity, 9), dtype=np.float32)
        new_colour = np.zeros((new_capacity, 4), dtype=np.float32)
        if self._n_lines > 0:
            new_lines[:self._n_lines]  = self._lines_xyzxyz[:self._n_lines]
            new_colour[:self._n_lines] = self._line_colour[:self._n_lines]
        self._lines_xyzxyz   = new_lines
        self._line_colour    = new_colour
        self._lines_capacity = new_capacity

    def _grow_points(self, required: int):
        self._scatter._active_count = self._n_points
        self._scatter._grow_buffers(required)
        self._pos_host    = self._scatter._positions_buffer.data
        self._colour_host = self._scatter._colours_buffer.data
        cap = self._scatter._capacity
        new_vel  = np.zeros((cap, 3),          dtype=np.float32)
        new_idx  = np.zeros((cap,),            dtype=np.int32)
        new_mods = np.ones((cap, N_MOD_COLS), dtype=np.float32)
        if self._n_points > 0:
            new_vel[:self._n_points]  = self._point_vel[:self._n_points]
            new_idx[:self._n_points]  = self._point_line_idx[:self._n_points]
            new_mods[:self._n_points] = self._point_mods[:self._n_points]
        self._point_vel      = new_vel
        self._point_line_idx = new_idx
        self._point_mods     = new_mods

    def _setup_cuda(self):
        ctx = self.frontend.cuda_ctx
        for arr in (self._positions_gpu, self._velocities_gpu,
                    self._line_idx_gpu, self._lines_gpu, self._mods_gpu):
            if arr is not None:
                ctx.free(arr)
        self._positions_gpu  = ctx.m(self._pos_host)
        self._velocities_gpu = ctx.m(self._point_vel)
        self._line_idx_gpu   = ctx.m(self._point_line_idx)
        self._lines_gpu      = ctx.m(np.ascontiguousarray(self._lines_xyzxyz))
        self._mods_gpu       = ctx.m(np.ascontiguousarray(self._point_mods))
        if self._stream1 is None:
            self._stream1 = ctx.stream()
            self._stream2 = ctx.stream()
        from cuda_wrapper import LaunchConfig1D, DeviceProperties
        cfg = LaunchConfig1D(DeviceProperties(), n_workers=max(1, self._scatter._capacity))
        if self._kernel is None:
            self._kernel = ctx.get_kernel("points_mode_motion", "points_mode_motion", cfg)
        else:
            self._kernel.new_launch_config(cfg)

    def _sync_to_gpu(self):
        if (self._positions_gpu.shape[0] != self._scatter._capacity
                or self._lines_gpu.shape[0] != self._lines_capacity):
            self._setup_cuda()
        else:
            self._positions_gpu.copy_from(self._pos_host)
            self._velocities_gpu.copy_from(self._point_vel)
            self._line_idx_gpu.copy_from(self._point_line_idx)
            self._lines_gpu.copy_from(np.ascontiguousarray(self._lines_xyzxyz))
            self._mods_gpu.copy_from(np.ascontiguousarray(self._point_mods))

    def _update_point_mods(self, handle: LinesHandle,
                           spring_strength=None, jitter_strength=None,
                           dt=None, damping=None, colour_range=None,
                           line_upwards_interaction=None):
        """Re-sample selected modulation columns and/or colours for `handle`.

        Uploads a single batched copy at the end. No-op if every argument is
        None.
        """
        col_specs = [
            (MOD_SPRING,  spring_strength),
            (MOD_JITTER,  jitter_strength),
            (MOD_DT,      dt),
            (MOD_DAMPING, damping),
            (MOD_UPWARDS, line_upwards_interaction),
        ]
        any_mod_touched    = any(spec is not None for _, spec in col_specs)
        any_colour_touched = colour_range is not None
        if not (any_mod_touched or any_colour_touched):
            return

        for _line_index, slot_start, slot_count in handle.entries:
            sl = slice(slot_start, slot_start + slot_count)
            for col, spec in col_specs:
                if spec is not None:
                    self._point_mods[sl, col] = _sample_mod_column(spec, slot_count)
            if any_colour_touched:
                self._colour_host[sl] = _sample_colours_range(colour_range, slot_count)

        if any_colour_touched:
            self._colours_dirty = True
        if any_mod_touched and self._mods_gpu is not None:
            self._mods_gpu.copy_from(np.ascontiguousarray(self._point_mods))
