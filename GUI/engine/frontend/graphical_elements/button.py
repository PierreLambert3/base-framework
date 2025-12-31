from GUI.engine.frontend.graphical_elements.graphical_element import Element_2d
from GUI.engine.frontend.graphical_elements.parallelepiped import Parallelepiped
from GUI.engine.frontend.theme import ORANGE_YELLOW, ORANGE_DARK, interpolate_color, brighten, PINK_NEON, BONE, PINK_ELECTRIC, ORANGE_RED
import pygfx
import numpy as np


class Button_2d(Element_2d):
    """A 2D button with borders and centered text."""
    
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, text, 
                 colour=None, text_colour=ORANGE_YELLOW, font_size=24,
                 pointer_move_inside_callback=None, pointer_click_callback=None,
                 toggleable=False):
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, colour=colour)
        self.text = text
        self.text_colour = text_colour
        self.font_size = font_size
        self.hovered = False
        self.toggleable = toggleable
        self.pressed    = False
        
        # Create borders (all 4 sides)
        self._make_button_borders()
        
        # Create centered text
        self._make_text()

        # register as hoverable and clickable
        self.register_hoverable()
        self.register_clickable()

        # Callbacks
        if pointer_move_inside_callback is not None:
            self.add_pointer_move_inside_callback(pointer_move_inside_callback)
        if pointer_click_callback is not None:
            self.add_pointer_click_callback(pointer_click_callback)
        
    def _make_button_borders(self, thickness=2.0):
        hw, hh = self.size[0] / 2, self.size[1] / 2
        # All 4 borders as a closed loop
        positions = np.array([
            [-hw, -hh, 0],
            [ hw, -hh, 0],
            [ hw,  hh, 0],
            [-hw,  hh, 0],
            [-hw, -hh, 0],  # close the loop
        ], dtype=np.float32)
        
        geom = pygfx.Geometry(positions=positions)
        mat = pygfx.LineMaterial(color=self.colour, thickness=thickness, aa=True)
        self.border_line = pygfx.Line(geom, mat)
        self.border_line.local.position = self.center
        self.register_gfx_object(self.border_line)
    
    def _make_text(self):
        self.text_obj = pygfx.Text(
            text=self.text,
            font_size=self.font_size,
            anchor="middle-center",
            screen_space=False,
            material=pygfx.TextMaterial(color=self.text_colour)
        )
        # self.text_obj.material.depth_test = True
        # self.text_obj.material.depth_write = True
        self.text_obj.local.position = self.center
        self.register_gfx_object(self.text_obj)
    
    def set_text(self, new_text):
        self.text = new_text
        self.text_obj.set_text(new_text)
    
    def die(self):
        self.scene.remove(self.border_line)
        self.scene.remove(self.text_obj)
        super().die()

    def on_pointer_move_inside(self, event, page_coords): # make brighter on hover
        self.hovered = True
        if self.toggleable and self.pressed:
            self._apply_toggledSelected_style() # if this is a toggleable pressed button (e.g. a selected tab), keep the pressed look.
        else:
            self.border_line.material.color = brighten(self.colour, 0.4)
            self.text_obj.material.color    = BONE
        super().on_pointer_move_inside(event, page_coords)

    def on_pointer_move_outside(self, event, page_coords):
        self.hovered = False
        if not self.toggleable or (self.toggleable and not self.pressed):
            self.reset_colours()
        super().on_pointer_move_outside(event, page_coords)

    def on_pointer_down_inside(self, event, page_coords):
        self.border_line.material.color = brighten(interpolate_color(self.colour, BONE, 0.8), 0.5)
        self.text_obj.material.color    = brighten(self.colour, 0.8)
        super().on_pointer_down_inside(event, page_coords)

    def on_pointer_up_inside(self, event, page_coords):
        if self.toggleable:
            self.toggle()
        super().on_pointer_up_inside(event, page_coords)
    
    def toggle(self):
        self.set_pressed(not self.pressed)
    
    def set_pressed(self, pressed_state):
        self.pressed = pressed_state
        if self.pressed:
            if not self.toggleable:
                self._apply_pressed_style()
            else:
                self._apply_toggledSelected_style()
        else:
            self.reset_colours()

    def stop_pointer_down_effect(self):
        if self.toggleable and self.pressed: # For toggleable buttons, we must not destroy the persistent pressed look.
            self._apply_toggledSelected_style()
            return

        if self.hovered:
            self.border_line.material.color = brighten(self.colour, 0.4)
            self.text_obj.material.color    = BONE
        else:
            self.reset_colours()

    def reset_colours(self):
        self.border_line.material.color = self.colour
        self.text_obj.material.color    = self.text_colour

    def hide(self):
        return super().hide()
    
    def _apply_pressed_style(self):
        self.border_line.material.color = brighten(interpolate_color(self.colour, BONE, 0.8), 0.5)
        self.text_obj.material.color    = brighten(self.colour, 0.8)
    
    def _apply_toggledSelected_style(self):
        self.border_line.material.color = brighten(self.colour, 0.29)
        self.text_obj.material.color    = interpolate_color(self.colour, BONE, 0.2)