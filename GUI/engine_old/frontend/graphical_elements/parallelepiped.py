from GUI.engine.frontend.graphical_elements.graphical_element import Element_3d
from GUI.engine.frontend.theme import ORANGE_YELLOW, ORANGE_DARK, interpolate_color
import pygfx
import numpy as np


class Parallelepiped(Element_3d):
    """
    A 3D parallelepiped (box) element with visible edges.
    The 12 edges of the box are drawn as lines.
    """
    def __init__(self, unique_name, parent, bl_xyz_px, size_xyz_px, edge_color=None, edge_thickness=2.0, colour=None):
        super().__init__(unique_name, parent, bl_xyz_px, size_xyz_px, colour=colour)

        self.edge_color = edge_color if edge_color is not None else interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5)
        self.edge_thickness = edge_thickness
        self.colour = colour if colour is not None else interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5)
        
        self.edges_group = self._generate_edges()
        self.register_gfx_object(self.edges_group)
    
    def _generate_edges(self):
        """
        Generate the 12 edges of the parallelepiped as a single Line object.
        Uses LineSegmentMaterial so each pair of points forms a separate segment.
        """
        w, h, d = self.size
        
        # 8 vertices of the box, centered at origin (for rotation support)
        # We'll position the group at center
        hw, hh, hd = w/2, h/2, d/2
        
        vertices = np.array([
            # Front face (z = -hd)
            [-hw, -hh, -hd],  # 0: front-bottom-left
            [ hw, -hh, -hd],  # 1: front-bottom-right
            [ hw,  hh, -hd],  # 2: front-top-right
            [-hw,  hh, -hd],  # 3: front-top-left
            # Back face (z = +hd)
            [-hw, -hh,  hd],  # 4: back-bottom-left
            [ hw, -hh,  hd],  # 5: back-bottom-right
            [ hw,  hh,  hd],  # 6: back-top-right
            [-hw,  hh,  hd],  # 7: back-top-left
        ], dtype=np.float32)
        
        # 12 edges as pairs of vertex indices
        edges = [
            # Front face edges
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Back face edges
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Connecting edges (front to back)
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        
        # Build positions array for line segments (2 points per edge)
        positions = []
        for i0, i1 in edges:
            positions.append(vertices[i0])
            positions.append(vertices[i1])
        positions = np.array(positions, dtype=np.float32)
        
        geom = pygfx.Geometry(positions=positions)
        mat = pygfx.LineSegmentMaterial(color=self.colour, thickness=self.edge_thickness, aa=True)
        edges_line = pygfx.Line(geom, mat)
        
        # Position at center
        edges_line.local.position = self.center
        
        return edges_line
    
    def set_color(self, color):
        """Update the edge color."""
        self.colour = color
        self.edges_group.material.color = color
    
    def set_edge_thickness(self, thickness):
        """Update the edge thickness."""
        self.edge_thickness = thickness
        self.edges_group.material.thickness = thickness
    
    def die(self):
        """Clean up pygfx objects."""
        self.scene.remove(self.edges_group)
        super().die()