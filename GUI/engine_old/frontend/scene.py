import pygfx 
from rendercanvas.auto import RenderCanvas, loop
import pylinalg as la
import numpy as np

class Scene:
    def __init__(self, canvas_title):
        window_size   = (1000, 800)
        self.canvas   = RenderCanvas(title=canvas_title, size=window_size)
        self.renderer = pygfx.WgpuRenderer(self.canvas)
        self.scene    = pygfx.Scene()

        # post processing effects
        self.bloom = None
        effect_passes = []
        from GUI.engine.frontend.theme import POSTPROCESSING_BLOOM, POSTPROCESSING_NOISE, BLOOM_TINT
        if POSTPROCESSING_NOISE: # noise effect
            effect_passes.append(pygfx.renderers.wgpu.NoisePass(noise=0.012))
        if POSTPROCESSING_BLOOM: # bloom effect
            from GUI.engine.frontend.post_processing.bloom.bloom_pass import BloomPass
            bloom = BloomPass(
                depth=5,
                threshold=0.4,
                knee=0.2,
                strength=20.0,
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
        self.camera.local.position = (window_size[0] / 2, window_size[1] / 2, 1000)
        self.camera.look_at((window_size[0] / 2, window_size[1] / 2, 0))
        self.scene.add(self.camera)

    def setup_one_frame(self, one_frame_method):
        self.canvas.request_draw(one_frame_method)

    def render(self):
        self.renderer.render(self.scene, self.camera)

    def xy_on_mesh(self, screen_coords, pickable_mesh):
        xy = self.screen_to_mesh_xy(screen_coords[0], screen_coords[1], pickable_mesh)
        is_hit = (0.0 <= xy[0] <= pickable_mesh.local.scale[0]) and (0.0 <= xy[1] <= pickable_mesh.local.scale[1]) if xy is not None else False
        return xy if is_hit else None

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
        origin, direction = self._compute_world_ray(x_px, y_px)
        if origin is None:
            return None

        # Define plane from mesh transform: origin O, local unit axes U and V
        pO = la.vec_transform((0.0, 0.0, 0.0), mesh.world.matrix)
        pU = la.vec_transform((1.0, 0.0, 0.0), mesh.world.matrix)
        pV = la.vec_transform((0.0, 1.0, 0.0), mesh.world.matrix)
        n  = np.cross(pU - pO, pV - pO)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            return None
        n /= n_norm

        denom = float(np.dot(n, direction))
        if abs(denom) < 1e-12:
            return None
        t = float(np.dot(n, (pO - origin)) / denom)
        if t < 0:
            return None
        hit_world = origin + t * direction

        # Transform hit to mesh local space
        hit_local = la.vec_transform(hit_world, mesh.world.inverse_matrix)

        width_px  = abs(mesh.local.scale[0])
        height_px = abs(mesh.local.scale[1])

        # Center-anchored plane: local x,y in [-0.5..0.5]
        x_px_out = (hit_local[0] + 0.5) * width_px
        y_px_out = (0.5 - hit_local[1]) * height_px

        # flip y to achor top-left to bottom-left
        y_flip = height_px - y_px_out

        return float(x_px_out), float(y_flip)