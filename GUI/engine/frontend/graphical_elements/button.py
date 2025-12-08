from GUI.engine.frontend.graphical_elements.graphical_element import Element_2d
from GUI.engine.frontend.graphical_elements.parallelepiped import Parallelepiped
from GUI.engine.frontend.theme import ORANGE_YELLOW, ORANGE_DARK, interpolate_color
import pygfx
import numpy as np


class Button_2d(Element_2d):
    """A 2D button with borders and centered text."""
    
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, text, 
                 colour=None, text_colour="white", font_size=16):
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, colour=colour)
        self.text = text
        self.text_colour = text_colour
        self.font_size = font_size
        
        # Create borders (all 4 sides)
        self._make_button_borders()
        
        # Create centered text
        self._make_text()
        
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
            material=pygfx.TextMaterial(color=self.text_colour),
        )
        self.text_obj.local.position = self.center
        self.register_gfx_object(self.text_obj)
    
    def set_text(self, new_text):
        self.text = new_text
        self.text_obj.set_text(new_text)
    
    def die(self):
        self.scene.remove(self.border_line)
        self.scene.remove(self.text_obj)
        super().die()


class Button_3d(Parallelepiped):
    """A 3D button (parallelepiped) with edges and centered text on the front face."""
    
    def __init__(self, unique_name, parent, bl_xyz_px, size_xyz_px, text,
                 colour=None, text_colour="white", font_size=16, edge_thickness=2.0):
        super().__init__(unique_name, parent, bl_xyz_px, size_xyz_px, 
                         edge_thickness=edge_thickness, colour=colour)
        self.text = text
        self.text_colour = text_colour
        self.font_size = font_size
        
        # Create centered text on front face
        self._make_text()
    
    def _make_text(self):
        self.text_obj = pygfx.Text(
            text=self.text,
            font_size=self.font_size,
            anchor="middle-center",
            screen_space=False,
            material=pygfx.TextMaterial(color=self.text_colour),
        )
        # Position text on front face (z = center_z - depth/2)
        front_z = self.center[2] - self.size[2] / 2 + 0.1  # slightly in front
        self.text_obj.local.position = (self.center[0], self.center[1], front_z)
        self.register_gfx_object(self.text_obj)
    
    def set_text(self, new_text):
        self.text = new_text
        self.text_obj.set_text(new_text)
    
    def die(self):
        self.scene.remove(self.text_obj)
        super().die()