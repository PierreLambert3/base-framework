from GUI.engine.frontend.graphical_elements.graphical_element import _GraphicalElement
import pygfx

class Page(_GraphicalElement):
    def __init__(self, scene, unique_name, bl_xyz_px, size_xyz_px):
        self._scene = scene.scene
        center_xyz = bl_xyz_px[0] + size_xyz_px[0]/2, bl_xyz_px[1] + size_xyz_px[1]/2, bl_xyz_px[2] + size_xyz_px[2]/2
        super().__init__(unique_name, None, center_xyz, size_xyz_px)
        self.is_leaf          = False
        self.containers       = []
        self._containers_dict = {}

        # Interaction mesh (to translate pointer coordinates to page coordinates)
        self.scene.add(self._generate_pickable_mesh())

        """
        # Interaction: create an invisible but pickable mesh representing the page rectangle.
        geom = screen.geometries.unit_rectangle_filled
        mat  = gfx.MeshBasicMaterial(color="#000", opacity=0.0, pick_write=True)
        mesh = gfx.Mesh(geom, mat)
        mesh.local.scale    = (self.W_px, self.H_px, 0.1)
        mesh.local.position = (self.W_px * 0.5, self.H_px * 0.5, self.z_px-0.2)
        self.interacterable_mesh = mesh
        """

    def draw(self, EMA_total_rendering_time):
        for container in self.containers:
            container.draw(EMA_total_rendering_time)

    def add_container(self, container):
        assert container.name not in self._containers_dict, f"Container with name '{container.name}' already exists in page '{self.unique_name}'"
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
        pick_mesh.local.scale    = (self.size[0], self.size[1], 0.01)
        pick_mesh.local.position = self.center
        self.pick_mesh = pick_mesh
        return pick_mesh