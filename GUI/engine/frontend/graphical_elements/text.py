from GUI.engine.frontend.graphical_elements.graphical_element import Element_2d
from GUI.engine.frontend.graphical_elements.parallelepiped import Parallelepiped
from GUI.engine.frontend.theme import ORANGE_YELLOW, ORANGE_DARK, interpolate_color, brighten, PINK_NEON, BONE, PINK_ELECTRIC, ORANGE_RED
import pygfx
import numpy as np


class Text(Element_2d):
    """A 2D button with borders and centered text."""
    
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, text, 
                 colour=None, text_colour=ORANGE_YELLOW, font_size=24,
                 pointer_move_inside_callback=None, pointer_click_callback=None,
                 toggleable=False):
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, colour=colour)
        self.text = text
        self.text_colour = text_colour
        self.font_size = font_size
        self._make_text()

    def _make_text(self):
        self.text_obj = pygfx.Text(
            text=self.text,
            font_size=self.font_size,
            anchor="middle-center",
            screen_space=False,
            material=pygfx.TextMaterial(color=self.text_colour)
        )
        self.text_obj.local.position = self.center
        self.register_gfx_object(self.text_obj)
        if self.parent.hidden:
            self.hide()
    
    def set_text(self, new_text):
        if new_text != self.text:
            self.text = new_text
            self.text_obj.set_text(new_text)