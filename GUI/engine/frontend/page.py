from GUI.engine.frontend.graphical_elements.graphical_element import _GraphicalElement
import pygfx
import numpy as np

class Page(_GraphicalElement):
    def __init__(self, scene, unique_name, bl_xyz_px, size_xyz_px):
        self._scene = scene.scene
        center_xyz = bl_xyz_px[0] + size_xyz_px[0]/2, bl_xyz_px[1] + size_xyz_px[1]/2, bl_xyz_px[2] + size_xyz_px[2]/2
        super().__init__(unique_name, None, center_xyz, size_xyz_px)
        self.is_leaf          = False
        self.containers       = []
        self._containers_dict = {}

        # border
        self.register_gfx_object(self._generate_border())

        # Interaction mesh (to translate pointer coordinates to page coordinates)
        self.register_gfx_object(self._generate_pickable_mesh())

        # clickable or hoverable elements
        self.clickable_elements = []
        self.hoverable_elements = []
    
    def manage_mouse_pointer_move(self, event, page_coords):
        for element in self.hoverable_elements:
            if element.hit_by_page_coords(page_coords[0], page_coords[1]):
                print("Hovering over element:", element.name)
                if element.on_pointer_move_inside is not None:
                    element.on_pointer_move_inside(event, page_coords)
    
    def manage_mouse_pointer_down(self, event, page_coords):
        for element in self.clickable_elements:
            if element.hit_by_page_coords(page_coords[0], page_coords[1]):
                if element.on_pointer_down_inside is not None:
                    element.on_pointer_down_inside(event, page_coords)
    
    def manage_mouse_pointer_up(self, event, page_coords):
        for element in self.clickable_elements:
            if element.hit_by_page_coords(page_coords[0], page_coords[1]):
                if element.on_pointer_up_inside is not None:
                    element.on_pointer_up_inside(event, page_coords)

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
        mat  = pygfx.LineMaterial(color=interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5), thickness=2.0, aa=True)
        self.border_line = pygfx.Line(geom, mat)
        return self.border_line