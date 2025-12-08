from GUI.engine.frontend.page import Page
from GUI.engine.frontend.graphical_elements.graphical_element import Container
from GUI.engine.frontend.graphical_elements.scatterplot_2d import Scatterplot2D
from GUI.engine.frontend.graphical_elements.parallelepiped import Parallelepiped
from GUI.engine.frontend.theme import ORANGE_YELLOW, ORANGE_DARK, interpolate_color, brighten

class Page1(Page):
    def __init__(self, scene, page_name, bl_xyz_px=(0,0,0), size_xyz_px=(1000, 800,0)):
        super().__init__(scene, page_name, bl_xyz_px, size_xyz_px)
        
        self.add_container(Container("Scatterplot Container", self, (0.1, 0.1), (0.8, 0.8), borders=(1,0,1,0)))
        self.get("Scatterplot Container").add(Scatterplot2D("My Scatterplot", self.get("Scatterplot Container"), (0.0, 0.0), (1.0, 0.5))) 

        parallelepiped_xyz_px = (self.get("Scatterplot Container").bl[0],
                                self.get("Scatterplot Container").bl[1] + 0.5 * self.get("Scatterplot Container").size[1],
                                self.get("Scatterplot Container").bl[2] + 100)
        parallelepiped_size_xyz_px = (80,80,80)
        self.get("Scatterplot Container").add(Parallelepiped("My Parallelepiped", self.get("Scatterplot Container"), parallelepiped_xyz_px, parallelepiped_size_xyz_px, colour=brighten(ORANGE_YELLOW, 0.5)))

        self.get("Scatterplot Container").add(Container("Inner Container", self.get("Scatterplot Container"), (0.5, 0.5), (0.2, 0.2), borders=(1,1,1,1)))
        self.get("Inner Container").pos_xyz = (self.get("Inner Container").pos_xyz[0], self.get("Inner Container").pos_xyz[1], self.get("Inner Container").pos_xyz[2] + 100)