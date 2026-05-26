# Wiki: wiki/04-frontend.md (camera & scene), wiki/05-pages-and-elements.md
# (pick-mesh ray casting used by the page tree).
# Engine file -- pygfx scene + camera + ray-cast helpers used by the page
# system. Project code should not need to subclass `Scene`.

import pygfx 
from rendercanvas.auto import RenderCanvas, loop
import pylinalg as la
import numpy as np

class Scene:
    def __init__(self, canvas_title):
        window_size   = (1000, 800)
        self.canvas   = RenderCanvas(title=canvas_title, size=window_size)
        if hasattr(self.canvas, "_window"):
            import glfw
            wx, wy, ww, wh = glfw.get_monitor_workarea(glfw.get_primary_monitor())
            win_w, win_h = glfw.get_window_size(self.canvas._window)
            glfw.set_window_pos(self.canvas._window, wx + ww - win_w, wy + 1)
        self.renderer = pygfx.WgpuRenderer(self.canvas)
        self.scene    = pygfx.Scene()
        self.scene.add(pygfx.Background.from_color("#000"))

        # post processing effects
        self.bloom = None
        effect_passes = []
        from GUI.engine.frontend.theme import POSTPROCESSING_BLOOM, POSTPROCESSING_NOISE, BLOOM_TINT, BLOOM_STRENGHT
        if POSTPROCESSING_NOISE: # noise effect
            effect_passes.append(pygfx.renderers.wgpu.NoisePass(noise=0.012))
        if POSTPROCESSING_BLOOM: # bloom effect
            from GUI.engine.frontend.post_processing.bloom.bloom_pass import BloomPass
            bloom = BloomPass(
                depth=5,
                threshold=0.4,
                knee=0.2,
                strength=BLOOM_STRENGHT,
                color_select=True,
                color_center=BLOOM_TINT,  # Blue tint
                color_sigma=0.95,  # Now acts as tint strength (0-1), 0.8 = strong tint
            )
            effect_passes.append(bloom)
            self.bloom = bloom
        if effect_passes:
            self.renderer.effect_passes = tuple(list(self.renderer.effect_passes) + effect_passes)
       

        # Camera (simple perspective)
        self.camera = pygfx.PerspectiveCamera(50, aspect = window_size[0] / window_size[1])
        self.camera.local.position = (window_size[0] / 2, window_size[1] / 2, 2000)
        self.camera.look_at((window_size[0] / 2, window_size[1] / 2, 0))
        self.scene.add(self.camera)

    def setup_one_frame(self, one_frame_method):
        self.canvas.request_draw(one_frame_method)

    def render(self):
        # Guard against invalid canvas size (e.g., window minimized on Windows)
        w, h = self.renderer.logical_size
        if w <= 0 or h <= 0:
            return
        self.renderer.render(self.scene, self.camera)
    
    def clear(self):
        self.renderer.clear(all=True)

    def xy_on_mesh(self, screen_coords, pickable_mesh):
        xy = self.screen_to_mesh_xy(screen_coords[0], screen_coords[1], pickable_mesh)
        # Use abs(scale) because meshes may be scaled negatively depending on orientation.
        w = abs(pickable_mesh.local.scale[0])
        h = abs(pickable_mesh.local.scale[1])
        is_hit = (0.0 <= xy[0] <= w) and (0.0 <= xy[1] <= h) if xy is not None else False
        return xy if is_hit else None

    def _hit_on_mesh_plane(self, x_px, y_px, mesh: pygfx.Mesh):
        """Intersect the screen ray with the (infinite) plane of `mesh`.

        Returns (hit_world_xyz, hit_local_xyz, t) or (None, None, None) if no valid hit.
        - hit_world_xyz: world-space intersection point
        - hit_local_xyz: same point in mesh local space
        - t: ray parameter where hit_world = origin + t * direction
        """
        origin, direction = self._compute_world_ray(x_px, y_px)
        if origin is None:
            return None, None, None

        # Plane basis from mesh transform: origin O, local unit axes U and V
        pO = la.vec_transform((0.0, 0.0, 0.0), mesh.world.matrix)
        pU = la.vec_transform((1.0, 0.0, 0.0), mesh.world.matrix)
        pV = la.vec_transform((0.0, 1.0, 0.0), mesh.world.matrix)
        n = np.cross(pU - pO, pV - pO)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            return None, None, None
        n /= n_norm

        denom = float(np.dot(n, direction))
        if abs(denom) < 1e-12:
            return None, None, None

        t = float(np.dot(n, (pO - origin)) / denom)
        if t < 0:
            return None, None, None

        hit_world = origin + t * direction
        hit_local = la.vec_transform(hit_world, mesh.world.inverse_matrix)
        return hit_world, hit_local, t

    @staticmethod
    def _mesh_local_to_page_xy(hit_local, width_px: float, height_px: float):
        """Convert a mesh-local hit position to page pixel coordinates.

        Assumes a center-anchored plane with local x,y in [-0.5..0.5], and returns
        top-left anchored coordinates (x right, y down).
        """
        x_px_out = (hit_local[0] + 0.5) * width_px
        y_px_out = (0.5 - hit_local[1]) * height_px
        y_flip = height_px - y_px_out
        return float(x_px_out), float(y_flip)

    def _compute_world_ray(self, x_px, y_px):
        """Compute a world-space ray (origin, direction) from screen pixel coordinates.
        Returns (origin_xyz, dir_xyz) or (None, None) if renderer size is invalid.
        """
        w, h = self.renderer.logical_size
        if w <= 0 or h <= 0:
            return None, None

        # Convert to NDC (top-left origin in logical pixels)
        x_ndc = x_px / w * 2 - 1
        y_ndc = -(y_px / h * 2 - 1)

        proj = self.camera.projection_matrix
        view = self.camera.view_matrix  # == camera.world.inverse_matrix
        # tic = time.time()
        inv  = la.mat_inverse(proj @ view)
        # tic = time.time() - tic
        # print(f"mat_inverse took {tic*1000:.2f} ms", end="      \r")

        p0_ndc = np.array([x_ndc, y_ndc, 0.0, 1.0], dtype=float)  # near
        p1_ndc = np.array([x_ndc, y_ndc, 1.0, 1.0], dtype=float)  # far

        p0_world = inv @ p0_ndc; p0_world /= p0_world[3]
        p1_world = inv @ p1_ndc; p1_world /= p1_world[3]

        origin = p0_world[:3]
        direction = p1_world[:3] - p0_world[:3]
        norm = np.linalg.norm(direction)
        if norm < 1e-12:
            return None, None
        direction /= norm
        return origin, direction

    def screen_to_mesh_xy(self, x_px, y_px, mesh: pygfx.Mesh):
        """
        Map screen pixel coordinates to the local XY of a rectangular page mesh.
        Returns (page_x_px, page_y_px) or None if no valid intersection.
        """
        _, hit_local, _ = self._hit_on_mesh_plane(x_px, y_px, mesh)
        if hit_local is None:
            return None

        width_px = abs(mesh.local.scale[0])
        height_px = abs(mesh.local.scale[1])
        return self._mesh_local_to_page_xy(hit_local, width_px, height_px)
    
    def world_hit_on_mesh(self, screen_coords, pickable_mesh):
        """Get the world-space intersection point on a mesh from screen coordinates.
        Returns (hit_world_xyz, distance_to_hit) or (None, None) if no intersection."""
        hit_world, hit_local, t = self._hit_on_mesh_plane(screen_coords[0], screen_coords[1], pickable_mesh)
        if hit_world is None:
            return None, None

        # Check if hit is within mesh bounds (same convention as screen_to_mesh_xy)
        width_px = abs(pickable_mesh.local.scale[0])
        height_px = abs(pickable_mesh.local.scale[1])
        x_px_out, y_flip = self._mesh_local_to_page_xy(hit_local, width_px, height_px)
        if not ((0.0 <= x_px_out <= width_px) and (0.0 <= y_flip <= height_px)):
            return None, None

        return hit_world, t