from GUI.engine.frontend.graphical_elements.graphical_element import _GraphicalElement

class Page(_GraphicalElement):
    def __init__(self, scene, unique_name, bl_xyz_px, size_xyz_px):
        self._scene = scene.scene
        center_xyz = bl_xyz_px[0] + size_xyz_px[0]/2, bl_xyz_px[1] + size_xyz_px[1]/2, bl_xyz_px[2] + size_xyz_px[2]/2
        super().__init__(unique_name, None, center_xyz, size_xyz_px)
        self.is_leaf          = False
        self.containers       = []
        self._containers_dict = {}

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