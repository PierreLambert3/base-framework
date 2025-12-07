from GUI.engine.frontend.page import Page
from GUI.engine.frontend.graphical_elements.graphical_element import Container
from GUI.engine.frontend.graphical_elements.scatterplot_2d import Scatterplot2D

class Page1(Page):
    def __init__(self, scene, page_name, bl_xyz_px=(0,0,0), size_xyz_px=(800,600,0)):
        super().__init__(scene, page_name, bl_xyz_px, size_xyz_px)
        
        self.add_container(Container("Scatterplot Container", self, (0.0, 0.0), (1.0, 1.0)))
        self.get("Scatterplot Container").add(Scatterplot2D("My Scatterplot", self.get("Scatterplot Container"), (0.1, 0.1), (0.8, 0.8)))