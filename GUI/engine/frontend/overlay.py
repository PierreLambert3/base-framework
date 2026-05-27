import numpy as np
import pygfx
import wgpu

from GUI.engine.frontend.theme import AMBER, BONE, ORANGE_YELLOW, PINK_NEON, to_rgba, DEFAULT_FONT_SIZE
from GUI.engine.frontend.font_registry import get_family

# HUD overlay: a persistent screen-space layer that sits in front of all pages.
#
# Each element stores its position as a fraction of the current page's world-space
#  bounds (values in [0, 1]).  Positions are recomputed once when a page is set
#  (page_changed), and again whenever the gauge fill width changes (update_perf_stats).
#
# The overlay doesn't use _GraphicalElement classes but renders directly with
#  low-level pygfx calls.

W_GAUGE = 0.076
H_GAUGE = 0.005

GAUGE_FILL_COLOR = to_rgba(BONE, 0.7)

HUD_Z   = 20.0  # Z at which all HUD objects are placed.

FPS_FONT_SIZE        = 32 - 3
FPS_NUMBER_FONT_SIZE = 32 + 4

RECT_SIZE            = 55.0    # value-box side length (square, world units)

SCALAR_KEY_ALPHA     = 0.22    # key label opacity (ghost)
SCALAR_RECT_ALPHA    = 0.05    # value-box outline opacity
SCALAR_VAL_ALPHA     = 0.90    # value text opacity
GAUGE_OUTLINE_ALPHA  = SCALAR_VAL_ALPHA*0.8    # gauge outline opacity
GAUGE_FILL_ALPHA     = 0.30    # gauge fill opacity

GAUGE_X       = 0.0
CHUNK_GAUGE_X = 1.0 - H_GAUGE
FPS_X     = GAUGE_X + 0.02
SIM_X     = 1.0 - 0.04
BTM_Y_REL = 0.0

CHUNK_SIZE_MIN = 4
CHUNK_SIZE_MAX = 1000

HISTORY_SIZE = 40   # the window size for the history curve when displaying a scalar

# -- history curve --------------------------------------------------
CURVE_EMA_ALPHA = 0.06  # EMA smoothing for adaptive min/max
CURVE_THICKNESS = 1.2
CURVE_COLOR     = to_rgba(AMBER, 0.60)
CURVE_Z         = HUD_Z + 0.1   # in front of text

# -- stacked layout slot heights (world units) --------------------------------
KEY_SLOT_H   = 30.0   # key label slot
VAL_SLOT_H   = 40.0   # value text slot
CURVE_Y_SPAN = 45.0   # history curve slot
SEP_ALPHA    = 0.03   # separator line opacity (nearly invisible)
SEP_Z        = HUD_Z + 0.05   # between text and curve depth

class KeyValueHUD:
    """Three stacked elements: history curve (top), key label, value text.

    Layout ``invert=False`` (default) — top → bottom: curve / key / value
    Layout ``invert=True``            — top → bottom: curve / value / key
    """

    def __init__(
        self,
        scene,
        key_text: str,
        rel_left: float,
        rel_bottom: float,
        initial_value: str = "-.-",
        value_font_size: int = FPS_NUMBER_FONT_SIZE,
        key_font_size: int = FPS_FONT_SIZE,
        key_text_w: float = 50.0,
        offset_y: float = 0.0,   # kept for API compatibility; not used in layout
        invert: bool = False,
    ):
        self._rel_left   = rel_left
        self._rel_bottom = rel_bottom
        self._key_text_w = key_text_w
        self._invert     = invert

        # -- text objects (both centred horizontally within the widget) --------
        from GUI.engine.frontend import theme as _theme
        _family = get_family(_theme.DEFAULT_FONT_NAME[0])
        _font_kw = {"family": _family} if _family else {}

        self._key_obj = pygfx.Text(
            text=key_text,
            font_size=key_font_size,
            anchor="middle-center",
            screen_space=False,
            material=pygfx.TextMaterial(color=to_rgba(AMBER, SCALAR_KEY_ALPHA)),
            **_font_kw,
        )
        scene.add(self._key_obj)

        self._val_obj = pygfx.Text(
            text=initial_value,
            font_size=value_font_size,
            anchor="middle-center",
            screen_space=False,
            material=pygfx.TextMaterial(color=to_rgba(AMBER, SCALAR_VAL_ALPHA)),
            **_font_kw,
        )
        scene.add(self._val_obj)

        # -- separator lines (thin AMBER, nearly invisible) --------------------
        self._sep1_obj = pygfx.Line(
            pygfx.Geometry(positions=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)),
            pygfx.LineMaterial(color=to_rgba(AMBER, SEP_ALPHA), thickness=1.0, aa=True),
        )
        scene.add(self._sep1_obj)

        self._sep2_obj = pygfx.Line(
            pygfx.Geometry(positions=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)),
            pygfx.LineMaterial(color=to_rgba(AMBER, SEP_ALPHA), thickness=1.0, aa=True),
        )
        scene.add(self._sep2_obj)

        # -- history ring buffer ----------------------------------------------
        self._history       = np.full(HISTORY_SIZE, np.nan, dtype=np.float32)
        self._history_idx   = 0
        self._history_count = 0
        self._ema_min: float | None = None
        self._ema_max: float | None = None

        # cached world-space anchors (populated by reposition)
        self._base_x:       float = 0.0
        self._curve_base_y: float = 0.0

        # -- GPU curve resources (fixed size — HISTORY_SIZE is constant) ------
        n_segs = HISTORY_SIZE - 1
        self._curve_positions_cpu = np.zeros((2 * n_segs, 3), dtype=np.float32)
        self._curve_positions_cpu[:, 2] = CURVE_Z   # Z fixed; never overwritten
        self._curve_buffer = pygfx.Buffer(
            data=self._curve_positions_cpu,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.STORAGE,
        )
        self._curve_buffer.draw_range = (0, 0)
        self._curve_line = pygfx.Line(
            pygfx.Geometry(positions=self._curve_buffer),
            pygfx.LineSegmentMaterial(thickness=CURVE_THICKNESS, color=CURVE_COLOR, aa=True),
        )
        scene.add(self._curve_line)

    # ---------------------------------------------------------------- public API

    def set_value(self, text: str, value: float | None = None):
        self._val_obj.set_text(text)
        if value is not None:
            self._push_history(value)

    def reposition(self, bl: tuple, fw: float, fh: float):
        base_x = bl[0] + self._rel_left * fw
        base_y = bl[1] + self._rel_bottom * fh
        cx     = base_x + self._key_text_w * 0.5   # horizontal centre

        if not self._invert:
            # top → bottom: curve / key / value
            val_cy       = base_y + VAL_SLOT_H * 0.5
            sep1_y       = base_y + VAL_SLOT_H
            key_cy       = base_y + VAL_SLOT_H + KEY_SLOT_H * 0.5
            sep2_y       = base_y + VAL_SLOT_H + KEY_SLOT_H
            curve_base_y = base_y + VAL_SLOT_H + KEY_SLOT_H
        else:
            # top → bottom: curve / value / key
            key_cy       = base_y + KEY_SLOT_H * 0.5
            sep1_y       = base_y + KEY_SLOT_H
            val_cy       = base_y + KEY_SLOT_H + VAL_SLOT_H * 0.5
            sep2_y       = base_y + KEY_SLOT_H + VAL_SLOT_H
            curve_base_y = base_y + KEY_SLOT_H + VAL_SLOT_H

        self._key_obj.local.position = (cx, key_cy, HUD_Z)
        self._val_obj.local.position = (cx, val_cy, HUD_Z)

        self._sep1_obj.local.position = (base_x, sep1_y, SEP_Z)
        self._sep1_obj.local.scale    = (self._key_text_w, 1.0, 1.0)
        self._sep2_obj.local.position = (base_x, sep2_y, SEP_Z)
        self._sep2_obj.local.scale    = (self._key_text_w, 1.0, 1.0)

        self._base_x       = base_x
        self._curve_base_y = curve_base_y
        self._update_curve()

    def destroy(self):
        for obj in (self._key_obj, self._val_obj,
                    self._sep1_obj, self._sep2_obj, self._curve_line):
            if obj.parent is not None:
                obj.parent.remove(obj)

    # ---------------------------------------------------------------- private helpers

    def _push_history(self, value: float):
        self._history[self._history_idx] = value
        self._history_idx   = (self._history_idx + 1) % HISTORY_SIZE
        self._history_count = min(self._history_count + 1, HISTORY_SIZE)

        v_min = 0.7 * (value - 0.001)
        v_max = 1.3 * (value + 0.001)
        if self._ema_min is None:
            self._ema_min = v_min
            self._ema_max = v_max
        else:
            a = CURVE_EMA_ALPHA
            self._ema_min = a * v_min + (1.0 - a) * self._ema_min
            self._ema_max = a * v_max + (1.0 - a) * self._ema_max

        self._update_curve()

    def _update_curve(self):
        n = self._history_count
        if n < 2:
            self._curve_buffer.draw_range = (0, 0)
            return

        # Reconstruct oldest-to-newest ordering from ring buffer
        if n < HISTORY_SIZE:
            ordered = self._history[:n].copy()
        else:
            s = self._history_idx
            ordered = np.concatenate([self._history[s:], self._history[:s]])

        # World-space X: evenly span the widget width
        xs = np.linspace(self._base_x, self._base_x + self._key_text_w,
                         n, dtype=np.float32)

        # World-space Y: EMA-based min–max scaling, no clamping
        y_range = self._ema_max - self._ema_min
        if y_range < 1e-9:
            ys_norm = np.full(n, 0.5, dtype=np.float32)
        else:
            ys_norm = ((ordered - self._ema_min) / y_range).astype(np.float32)
        ys = (self._curve_base_y + ys_norm * CURVE_Y_SPAN).astype(np.float32)

        # Write interleaved segment-pair vertices into the CPU buffer
        n_verts = (n - 1) * 2
        buf = self._curve_positions_cpu
        buf[0:n_verts:2, 0] = xs[:-1]
        buf[0:n_verts:2, 1] = ys[:-1]
        buf[1:n_verts:2, 0] = xs[1:]
        buf[1:n_verts:2, 1] = ys[1:]

        self._curve_buffer.update_range(0, n_verts)
        self._curve_buffer.draw_range = (0, n_verts)


class Overlay:

    def __init__(self, scene):
        self._scene = scene.scene    # pygfx.Scene – used for add/remove
        self._current_page = None

        # Each entry has at minimum: {"type": str, "gfx": <pygfx obj>,
        #   "rel_left": float, "rel_bottom": float}.  Extra keys depend on type.
        self._elements: list[dict] = []

        self._fps_hud: KeyValueHUD | None = None
        self._sim_hud: KeyValueHUD | None = None

        self._chunk_gauge_pct: float = 0.0

        self._make_gauge()
        self._make_chunk_gauge()
        self._make_fps_display()
        self._make_sim_display()


    # ---------------------------------------------------------------- element builders

    def _make_gauge(self):
        """Thin outline rectangle (AMBER, dim) + BONE fill mesh, vertical, growing from bottom."""

        self._gauge_pct: float = 0.0
        # -- outline: 5-point closed polyline forming a unit rectangle --------
        rect = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],   # close the loop
        ], dtype=np.float32)
        bg_obj = pygfx.Line(
            pygfx.Geometry(positions=rect),
            pygfx.LineMaterial(color=to_rgba(AMBER, GAUGE_OUTLINE_ALPHA), thickness=1.2, aa=True),
        )
        self._scene.add(bg_obj)
        self._elements.append({
            "type":       "gauge_bg",
            "gfx":        bg_obj,
            "rel_left":   GAUGE_X,
            "rel_bottom": BTM_Y_REL,
            "rel_w":      H_GAUGE,   # narrow
            "rel_h":      W_GAUGE,   # tall
        })

        # -- fill scaled by current pct ----
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        fill_obj = pygfx.Mesh(
            pygfx.Geometry(positions=verts, indices=idx),
            pygfx.MeshBasicMaterial(color=GAUGE_FILL_COLOR, side="both"),
        )
        self._scene.add(fill_obj)
        self._elements.append({
            "type":       "gauge_fill",
            "gfx":        fill_obj,
            "rel_left":   GAUGE_X + H_GAUGE * 0.1,
            "rel_bottom": BTM_Y_REL + 0.001,
            "rel_w":      H_GAUGE * 0.8,
            "rel_max_h":  W_GAUGE - 0.002,
        })

    def _make_chunk_gauge(self):
        """Thin outline rectangle + BONE fill mesh at the bottom-right, mirror of the perf gauge.
        Fill level represents current chunk size linearly within [CHUNK_SIZE_MIN, CHUNK_SIZE_MAX]."""

        # -- outline -----------------------------------------------------------
        rect = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32)
        bg_obj = pygfx.Line(
            pygfx.Geometry(positions=rect),
            pygfx.LineMaterial(color=to_rgba(AMBER, GAUGE_OUTLINE_ALPHA), thickness=1.2, aa=True),
        )
        self._scene.add(bg_obj)
        self._elements.append({
            "type":       "chunk_gauge_bg",
            "gfx":        bg_obj,
            "rel_left":   CHUNK_GAUGE_X,
            "rel_bottom": BTM_Y_REL,
            "rel_w":      H_GAUGE,
            "rel_h":      W_GAUGE,
        })

        # -- fill scaled by current chunk pct ----------------------------------
        verts = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
        fill_obj = pygfx.Mesh(
            pygfx.Geometry(positions=verts, indices=idx),
            pygfx.MeshBasicMaterial(color=GAUGE_FILL_COLOR, side="both"),
        )
        self._scene.add(fill_obj)
        self._elements.append({
            "type":       "chunk_gauge_fill",
            "gfx":        fill_obj,
            "rel_left":   CHUNK_GAUGE_X + H_GAUGE * 0.1,
            "rel_bottom": BTM_Y_REL + 0.001,
            "rel_w":      H_GAUGE * 0.8,
            "rel_max_h":  W_GAUGE - 0.002,
        })

    def _make_fps_display(self):
        """Key label 'FPS' (ghost) above which sits the value — uses KeyValueHUD."""
        self._fps_hud = KeyValueHUD(
            self._scene,
            key_text="FPS",
            rel_left=FPS_X,
            rel_bottom=BTM_Y_REL,
            initial_value="-.-",
            value_font_size=FPS_NUMBER_FONT_SIZE,
            key_font_size=FPS_FONT_SIZE,
            key_text_w=50.0,
            offset_y=-0.01,
        )

    def _make_sim_display(self):
        """Key label 'SIM.' (ghost) above which sits the value — uses KeyValueHUD."""
        self._sim_hud = KeyValueHUD(
            self._scene,
            key_text="SIM.",
            rel_left=SIM_X,
            rel_bottom=BTM_Y_REL,
            initial_value="--.-",
            value_font_size=FPS_NUMBER_FONT_SIZE,
            key_font_size=FPS_FONT_SIZE,
            key_text_w=55.0,
            offset_y=-0.01,
        )



    # ---------------------------------------------------------------- page sync

    def page_changed(self, page):
        """Reposition all HUD elements based on the new page's world-space bounds."""
        self._current_page = page
        if page is None:
            return
        bl, (fw, fh) = self._get_page_frame()

        self._fps_hud.reposition(bl, fw, fh)
        self._sim_hud.reposition(bl, fw, fh)

        for elem in self._elements:
            gfx = elem["gfx"]
            t   = elem["type"]

            if t == "gauge_bg" or t == "chunk_gauge_bg":
                gfx.local.position = (
                    bl[0] + elem["rel_left"]   * fw,
                    bl[1] + elem["rel_bottom"] * fh,
                    HUD_Z,
                )
                gfx.local.scale = (elem["rel_w"] * fw, elem["rel_h"] * fh, 1.0)

            elif t == "gauge_fill":
                self._reposition_gauge_fill(elem, bl, fw, fh)

            elif t == "chunk_gauge_fill":
                self._reposition_chunk_gauge_fill(elem, bl, fw, fh)

    def destroy(self):
        """Remove all pygfx objects from the scene and free the element list."""
        for elem in self._elements:
            if elem["gfx"].parent is not None:
                elem["gfx"].parent.remove(elem["gfx"])
        self._elements.clear()
        self._fps_hud.destroy()
        self._sim_hud.destroy()

    # ---------------------------------------------------------------- public update API

    def update_perf_stats(self, pct: float, fps: float):
        """Called by the frontend after computing pct_time_active and FPS_total."""
        self._gauge_pct = pct

        fill_elem = next((e for e in self._elements if e["type"] == "gauge_fill"), None)
        if fill_elem is not None and self._current_page is not None:
            bl, (fw, fh) = self._get_page_frame()
            self._reposition_gauge_fill(fill_elem, bl, fw, fh)

        self._fps_hud.set_value(f"{fps:.1f}", fps)

    def update_chunk_gauge(self, pct: float):
        """Called when the chunk size changes. pct is in [0, 100]."""
        self._chunk_gauge_pct = pct

        fill_elem = next((e for e in self._elements if e["type"] == "chunk_gauge_fill"), None)
        if fill_elem is not None and self._current_page is not None:
            bl, (fw, fh) = self._get_page_frame()
            self._reposition_chunk_gauge_fill(fill_elem, bl, fw, fh)

    def update_sim_speed(self, rate: float):
        """Called when a worker reports its chunks-per-second rate."""
        if rate > 10.0:
            rate = int(rate)
            self._sim_hud.set_value(f"{rate}", rate)
        else:
            self._sim_hud.set_value(f"{rate:.1f}", rate)

    # ---------------------------------------------------------------- internal helpers

    def _reposition_gauge_fill(self, elem: dict, bl: tuple, fw: float, fh: float):
        fill_h   = max(self._gauge_pct / 100.0 * elem["rel_max_h"] * fh, 0.001)
        fill_w   = elem["rel_w"] * fw
        left_x   = bl[0] + elem["rel_left"]   * fw
        bottom_y = bl[1] + elem["rel_bottom"] * fh
        elem["gfx"].local.position = (left_x, bottom_y, HUD_Z)
        elem["gfx"].local.scale    = (fill_w, fill_h, 1.0)

    def _reposition_chunk_gauge_fill(self, elem: dict, bl: tuple, fw: float, fh: float):
        fill_h   = max(self._chunk_gauge_pct / 100.0 * elem["rel_max_h"] * fh, 0.001)
        fill_w   = elem["rel_w"] * fw
        left_x   = bl[0] + elem["rel_left"]   * fw
        bottom_y = bl[1] + elem["rel_bottom"] * fh
        elem["gfx"].local.position = (left_x, bottom_y, HUD_Z)
        elem["gfx"].local.scale    = (fill_w, fill_h, 1.0)

    def _get_page_frame(self):
        """Return (bl, (fw, fh)) — the page's world-space bounding rectangle,
        with the z-component pinned to HUD_Z for element placement."""
        page = self._current_page
        bl = (page.bl[0], page.bl[1], HUD_Z)
        return bl, (page.size[0], page.size[1])


