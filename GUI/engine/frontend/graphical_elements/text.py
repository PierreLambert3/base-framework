from GUI.engine.frontend.graphical_elements.graphical_element import Element_2d
from GUI.engine.frontend.graphical_elements.parallelepiped import Parallelepiped
from GUI.engine.frontend.theme import DEFAULT_FONT_SIZE, DEFAULT_FONT_NAME, ORANGE_YELLOW, ORANGE_DARK, interpolate_color, brighten, PINK_NEON, BONE, PINK_ELECTRIC, ORANGE_RED
from GUI.engine.frontend.font_registry import get_family
import pygfx
import numpy as np


class Text(Element_2d):
    """A 2D button with borders and centered text."""
    
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, text, 
                 colour=None, text_colour=ORANGE_YELLOW, font_size=DEFAULT_FONT_SIZE, font_name=DEFAULT_FONT_NAME[0],
                 pointer_move_inside_callback=None, pointer_click_callback=None,
                 toggleable=False, line_mode=0, point_mode=0, point_mode_params=None):
        
        # Text element do not support point mode
        if point_mode != 0:
            print(f"Warning: point mode for Text graphical element is not implemented yet, falling back to point_mode=0")
            point_mode = 0
        if line_mode != 0:
            print(f"Warning: line mode for Text graphical element is not implemented yet, falling back to line_mode=0")
            line_mode = 0

        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, colour=colour, line_mode=line_mode, point_mode=point_mode, point_mode_params=point_mode_params)
        self.text = text
        self.text_colour = text_colour
        self.font_size = font_size
        self.font_name = font_name
        self._make_text()

    def _make_text(self):
        family = get_family(self.font_name)
        text_kwargs = dict(
            text=self.text,
            font_size=self.font_size,
            anchor="middle-center",
            screen_space=False,
            material=pygfx.TextMaterial(color=self.text_colour),
        )
        if family is not None:
            text_kwargs['family'] = family
        self.text_obj = pygfx.Text(**text_kwargs)
        self.text_obj.local.position = self.center
        self.register_gfx_object(self.text_obj)
        if self.parent.hidden:
            self.hide()
    
    def set_text(self, new_text):
        if new_text != self.text:
            self.text = new_text
            self.text_obj.set_text(new_text)