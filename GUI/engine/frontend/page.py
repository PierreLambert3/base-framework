from GUI.engine.frontend.graphical_elements.graphical_element import _GraphicalElement
import pygfx
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

        # border
        if not no_border:
            self.register_gfx_object(self._generate_border())

        # Interaction mesh (to translate pointer coordinates to page coordinates)
        self.register_gfx_object(self._generate_pickable_mesh())

        # clickable or hoverable elements
        self.clickable_elements  = []  # mouse click : elements needs to define on_pointer_down_inside and on_pointer_up_inside
        self.hoverable_elements  = []  # mouse move  : elements needs to define on_pointer_move_inside and on_pointer_move_outside
        self.awaiting_mouse_up   = []
        self.awaiting_hover_out  = []
    
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
        mat  = pygfx.MeshBasicMaterial(color="#000", opacity=0.0, pick_write=True)  # invisible but pickable
        pick_mesh = pygfx.Mesh(geom, mat)
        pick_mesh.local.scale    = (self.size[0], self.size[1], 0.001)
        pick_mesh.local.position = (self.center[0], self.center[1], self.center[2]-0.5)
        self.pick_mesh = pick_mesh
        return pick_mesh

    def _generate_border(self):
        from GUI.engine.frontend.theme import ORANGE_YELLOW, ORANGE_DARK, interpolate_color
        # Get bottom-left corner and size
        bl = self.bottom_left
        w, h = self.size[0], self.size[1]
        z = bl[2]
        
        # Define 4 corners: bottom-left, bottom-right, top-right, top-left, back to bottom-left
        positions = np.array([
            [bl[0],     bl[1],     z],  # bottom-left
            [bl[0] + w, bl[1],     z],  # bottom-right
            [bl[0] + w, bl[1] + h, z],  # top-right
            [bl[0],     bl[1] + h, z],  # top-left
            [bl[0],     bl[1],     z],  # back to bottom-left (close the loop)
        ], dtype=np.float32)
        
        geom = pygfx.Geometry(positions=positions)
        mat  = pygfx.LineMaterial(color=self.colour, thickness=1.0, aa=True)
        self.border_line = pygfx.Line(geom, mat)
        return self.border_line
    
    def hide(self):
        for child in self.containers:
            child.hide()
        super().hide()
        if self.frontend.current_page == self:
            self.frontend.current_page = None
    
    def show(self):
        for child in self.containers:
            child.show()
        super().show()
    
    def destroy(self):
        # Destroy all containers (which recursively destroy their children)
        for container in self.containers:
            container.die()
        self.containers.clear()
        self._containers_dict.clear()
        
        # Clear interactive element lists
        self.clickable_elements.clear()
        self.hoverable_elements.clear()
        self.awaiting_mouse_up.clear()
        self.awaiting_hover_out.clear()
        
        # Destroy the page's own gfx objects (border, pick_mesh)
        self.die()
        
        # Clear references
        self._scene = None
        self._scene_wrapper = None

        # set frontend "current_page" to None if this page was the current page, and remove from frontend pages dict
        if self.frontend.current_page == self:
            self.frontend.current_page = None
        del self.frontend.pages[self.name]