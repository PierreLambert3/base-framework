from GUI.engine.frontend.graphical_elements.graphical_element import Element_3d
from GUI.engine.frontend.theme import ORANGE_YELLOW, ORANGE_DARK, interpolate_color
from GUI.engine.frontend import theme as _theme
import pygfx
import numpy as np


class Parallelepiped(Element_3d):
    """
    A 3D parallelepiped (box) element with visible edges.
    The 12 edges of the box are drawn as lines.
    """
    def __init__(self, unique_name, parent, bl_xyz_px, size_xyz_px, edge_color=None, edge_thickness=1.0, colour=None, ignore_pointmode=False):
        super().__init__(unique_name, parent, bl_xyz_px, size_xyz_px, colour=colour, ignore_pointmode=ignore_pointmode)

        self.edge_color = edge_color if edge_color is not None else interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5)
        self.edge_thickness = edge_thickness
        self.colour = colour if colour is not None else interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5)

        w, h, d = self.size
        hw, hh, hd = w/2, h/2, d/2
        
        # 8 vertices in local coordinates
        V = [
            (-hw, -hh, -hd),  # 0
            ( hw, -hh, -hd),  # 1
            ( hw,  hh, -hd),  # 2
            (-hw,  hh, -hd),  # 3
            (-hw, -hh,  hd),  # 4
            ( hw, -hh,  hd),  # 5
            ( hw,  hh,  hd),  # 6
            (-hw,  hh,  hd),  # 7
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        segments = [(V[i0], V[i1]) for i0, i1 in edges]
        self.add_lines(segments, colour=self.colour, thickness=self.edge_thickness)
    
    def set_color(self, color):
        self.colour = color
        self.set_lines_colour(color)

    def set_edge_thickness(self, thickness):
        self.edge_thickness = thickness
        self.set_lines_thickness(thickness)

    def die(self):
        if not _theme.POINTS_MODE:
            self.scene.remove(self.edges_group)
        super().die()