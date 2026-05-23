# Wiki: wiki/05-pages-and-elements.md (Page / Container / Element_2d hierarchy).
# Related: wiki/04-frontend.md (per-frame contract), wiki/08-extending-the-framework.md
# (recipes for adding pages and subscribing to worker data streams).

from GUI.engine.frontend.graphical_elements.graphical_element import _GraphicalElement
from GUI.engine.frontend import theme as _theme
import pygfx
import time
import numpy as np

class Page(_GraphicalElement):
    def __init__(self, scene, unique_name, frontend, bl_xyz_px, size_xyz_px, colour=None, no_border=False):
        self.frontend       = frontend
        self._scene         = scene.scene
        self._scene_wrapper = scene
        center_xyz = bl_xyz_px[0] + size_xyz_px[0]/2, bl_xyz_px[1] + size_xyz_px[1]/2, bl_xyz_px[2] + size_xyz_px[2]/2
        super().__init__(unique_name, None, center_xyz, size_xyz_px, colour=colour)
        self.is_leaf          = False
        self.containers       = []
        self._containers_dict = {}

        # Points-mode manager (lazy: only allocated on first register_line call)
        self.points_mode      = None
        self._last_tick_time  = time.time()

        if not no_border:
            self._install_border()

        # Interaction mesh (to translate pointer coordinates to page coordinates)
        self.register_gfx_object(self._generate_pickable_mesh())

        # clickable or hoverable elements
        self.clickable_elements  = []  # mouse click : elements needs to define on_pointer_down_inside and on_pointer_up_inside
        self.hoverable_elements  = []  # mouse move  : elements needs to define on_pointer_move_inside and on_pointer_move_outside
        self.scrollable_elements = []  # mouse wheel : elements needs to define on_wheel
        self.awaiting_mouse_up   = []
        self.awaiting_hover_out  = []

        # user-defined callback when page is shown (for instance to hide some elements, or for animations)
        self.on_show             = None
    
    def manage_mouse_pointer_move(self, event, page_coords):
        # elements hit by the pointer
        for element in self.hoverable_elements:
            if not element.hidden and element.hit_by_page_coords(page_coords[0], page_coords[1]):
                element.on_pointer_move_inside(event, page_coords)
                if element not in self.awaiting_hover_out:
                    self.awaiting_hover_out.append(element)
        # elements no longer hit by the pointer
        for i in range(len(self.awaiting_hover_out)-1, -1, -1):
            element = self.awaiting_hover_out[i]
            if not element.hit_by_page_coords(page_coords[0], page_coords[1]):
                element.on_pointer_move_outside(event, page_coords)
                del self.awaiting_hover_out[i]
    
    def manage_mouse_pointer_down(self, event, page_coords):
        for element in self.clickable_elements:
            if not element.hidden and element.hit_by_page_coords(page_coords[0], page_coords[1]):
                element.on_pointer_down_inside(event, page_coords)
                self.awaiting_mouse_up.append(element)
    
    def manage_mouse_pointer_up(self, event, page_coords):
        for element in self.awaiting_mouse_up:
            if not element.hidden and element.hit_by_page_coords(page_coords[0], page_coords[1]):
                element.on_pointer_up_inside(event, page_coords)
            element.stop_pointer_down_effect()
        self.awaiting_mouse_up.clear()

    def manage_mouse_wheel(self, event, page_coords):
        """Dispatch wheel events to scrollable elements under the cursor."""
        for element in self.scrollable_elements:
            if not element.hidden and element.hit_by_page_coords(page_coords[0], page_coords[1]):
                element.on_wheel(event, page_coords)
                return True  # consumed by element
        return False  # not consumed, let page/camera handle it

    def add_container(self, container):
        assert container.name not in self._containers_dict, f"Container with name '{container.name}' already exists in page '{self.name}'"
        self.containers.append(container)
        self._containers_dict[container.name] = container

    def get(self, element_name):
        if element_name in self._containers_dict:
            return self._containers_dict[element_name]
        else:
            for container in self.containers:
                result = container.get(element_name)
                if result is not None:
                    return result
        return None
    
    def remove_container(self, container_name):
        assert container_name in self._containers_dict, f"Container with name '{container_name}' does not exist in page '{self.unique_name}'"
        container = self._containers_dict[container_name]
        container.die()
        self.containers.remove(container)
        del self._containers_dict[container_name]

    def _generate_pickable_mesh(self):
        geom = pygfx.geometries.plane_geometry(width=1.0, height=1.0)
        # mat  = pygfx.MeshBasicMaterial(color="#000", opacity=0.0, pick_write=True)  # invisible but pickable
        mat  = pygfx.MeshBasicMaterial(color="#000", opacity=0.0, pick_write=True, depth_write=False)  # invisible but pickable
        pick_mesh = pygfx.Mesh(geom, mat)
        pick_mesh.local.scale    = (self.size[0], self.size[1], 0.001)
        pick_mesh.local.position = (self.center[0], self.center[1], self.center[2]-0.5)
        self.pick_mesh = pick_mesh
        return pick_mesh

    def _install_border(self):
        hw, hh = self.size[0] / 2, self.size[1] / 2
        bl = (-hw, -hh, 0)
        br = ( hw, -hh, 0)
        tr = ( hw,  hh, 0)
        tl = (-hw,  hh, 0)
        segments = [(bl, br), (br, tr), (tr, tl), (tl, bl)]
        from GUI.engine.frontend.theme import transparent, darken, PINK_ELECTRIC, BLUE_WIERDNESS, AMBER2, ORANGE_DARK, AMBER, BONE, ORANGE_YELLOW
        self.add_lines(segments, pointMode_n_points_mul = 5.0, 
                       pointMode_colour_range=(transparent(darken(ORANGE_DARK, 0.1), 0.2), transparent(darken(AMBER2, 0.1), 0.2)),
                       pointMode_spring_strength=(8.0, 0.01),
                       pointMode_jitter_strength=(0.3, 0.01),
                       pointMode_line_upwards_interaction=(2.0, 1.6),
                       pointMode_dt=(0.1, 0.05),
                       invert_lookat=True)

    def _ensure_points_mode(self):
        """Lazily create this page's PointsModeManager."""
        if self.points_mode is None:
            from GUI.engine.frontend.points_mode import PointsModeManager
            self.points_mode = PointsModeManager(self)
        return self.points_mode
    
    def hide(self):
        for child in self.containers:
            child.hide()
        super().hide()
        if self.frontend.current_page == self:
            self.frontend.current_page = None
        self.on_hide()

    def show(self):
        for child in self.containers:
            child.show()
        super().show()
        if self.on_show is not None:
            self.on_show(self)

    def on_show(self):
        pass
    
    def on_hide(self):
        pass

    def one_frame(self, mouse_coords=(0, 0)):
        if self.points_mode is not None:
            now = time.time()
            dt = 0.001 if self._last_tick_time is None else (now - self._last_tick_time)

            threshold = 0.03
            if dt < threshold:
                return

            self._last_tick_time = now

            max_dt = 2.0 * threshold
            self.points_mode.tick(dt, max_dt)

    def destroy(self):
        # Destroy all containers (which recursively destroy their children)
        for container in self.containers:
            container.die()
        self.containers.clear()
        self._containers_dict.clear()
        
        # Clear interactive element lists
        self.clickable_elements.clear()
        self.hoverable_elements.clear()
        self.scrollable_elements.clear()
        self.awaiting_mouse_up.clear()
        self.awaiting_hover_out.clear()
        
        self.die()

        # Clear points-mode (frees its CUDA buffers + scatterplot)
        if self.points_mode is not None:
            self.points_mode.destroy()
            self.points_mode = None
        
        # Clear references
        self._scene = None
        self._scene_wrapper = None

        # set frontend "current_page" to None if this page was the current page, and remove from frontend pages dict
        if self.frontend.current_page == self:
            self.frontend.current_page = None
        del self.frontend.pages[self.name]