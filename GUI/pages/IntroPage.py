# Wiki: wiki/05-pages-and-elements.md (page tree),
#       wiki/08-extending-the-framework.md (recipe: add a new page).
# Minimal example page: one button that switches to the main page.

from GUI.engine.frontend.graphical_elements.button import Button_2d
from GUI.engine.frontend.page import Page
from GUI.engine.frontend.graphical_elements.graphical_element import Container
from GUI.engine.frontend.graphical_elements.scatterplot_2d import Scatterplot2D
from GUI.engine.frontend.graphical_elements.parallelepiped import Parallelepiped
from GUI.engine.frontend.graphical_elements.button import Button_2d
from GUI.engine.frontend.theme import AMBER, ORANGE_YELLOW, ORANGE_DARK, interpolate_color, brighten, PURPLE_LIGHT, PINK_ELECTRIC
import numpy as np

class Intro_Page(Page):
    def __init__(self, scene, page_name, frontend, bl_xyz_px=(0,0,0), size_xyz_px=(2000, 1600,0)):
        super().__init__(scene, page_name, frontend, bl_xyz_px, size_xyz_px, point_mode=1)
        
        main_container = Container(page_name+"Main Container", self, (0.0, 0.0), (1.0, 1.0), borders=(0,0,0,0))
        self.add_container(main_container)
        self.btn = Button_2d(page_name+"btn b", main_container, (0.6, 0.3), (0.3, 0.4),
                          text="go to page 2", colour=AMBER, text_colour=ORANGE_DARK,
                          pointer_click_callback=self.on_button_clicked)

    def one_frame(self, mouse_coords=(0, 0)):
        super().one_frame(mouse_coords)

    def on_button_clicked(self, event, element, page_coords):
        self.frontend.switch_to_scatterplotPage()
