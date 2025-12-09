import pygfx
import numpy as np
from pylinalg import quat_from_axis_angle, quat_from_euler, quat_mul
from GUI.engine.frontend.theme import ORANGE_YELLOW, ORANGE_DARK, interpolate_color


class _GraphicalElement:
    def __init__(self, unique_name, parent, center_xyz_px, size_xyz_px, colour=None):
        self.name         = unique_name
        self.parent       = parent
        self.pos_xyz      = center_xyz_px
        self.size_xyz     = size_xyz_px
        self.is_leaf      = True
        self.colour       = colour if colour is not None else interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5)
        self._rotation    = (0, 0, 0, 1)  # quaternion (x, y, z, w) - identity rotation
        self._gfx_objects = []  # list of pygfx objects that need rotation applied
        self.pagewise_xy  = self._make_page_coordinates()

        self._callback_on_pointer_move_inside    = None
        self._callback_on_pointer_move_outside   = None
        self._callback_pointer_click             = None

    @property
    def size(self):
        return self.size_xyz
    sz = size

    @property
    def center(self):
        return self.pos_xyz
    
    @property
    def rotation(self):
        return self._rotation
    
    @property
    def bottom_left(self):
        return (self.center[0] - self.size[0]/2, self.center[1] - self.size[1]/2, self.center[2] - self.size[2]/2)
    bl = bottom_left

    @property
    def is_page(self):
        return self.parent is None

    @property
    def scene(self):
        if not self.is_page:
            return self.parent.scene
        return self._scene
    
    @property
    def page(self):
        if self.is_page:
            return self
        return self.parent.page
    
    def register_gfx_object(self, gfx_obj):
        self.scene.add(gfx_obj)
        self._gfx_objects.append(gfx_obj)

    def rotate(self, angles_rad, order="xyz"):
        """
        Rotate the element by Euler angles (in radians) around X, Y, Z axes.
        angles_rad: tuple (rx, ry, rz) - rotation angles for each axis
        order: rotation order, e.g. "xyz", "zyx", etc.
        """
        delta_quat = quat_from_euler(angles_rad, order=order)
        self._rotation = tuple(quat_mul(delta_quat, np.array(self._rotation)))
        self._apply_rotation()
    
    def set_rotation(self, angle_rad, axis=(0, 0, 1)):
        """
        Set the element's rotation to a specific angle around the given axis (resets previous rotation).
        """
        axis = np.array(axis, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        self._rotation = tuple(quat_from_axis_angle(axis, angle_rad))
        self._apply_rotation()
    
    def _apply_rotation(self):
        """Apply the current rotation to all registered pygfx objects, rotating around center."""
        for gfx_obj in self._gfx_objects:
            gfx_obj.local.rotation = self._rotation
            # pygfx rotates around local origin, so we set position to center
            gfx_obj.local.position = self.center
            
    def _make_page_coordinates(self):
        page_bl = self.page.bl if self.parent is not None else (0, 0, 0)
        page_x = (self.bl[0] - page_bl[0]) if self.parent is not None else self.bl[0]
        page_y = (self.bl[1] - page_bl[1]) if self.parent is not None else self.bl[1]
        return (page_x, page_y)
    
    def hit_by_page_coords(self, x, y): # in pixels in the page coordinate system
        bl_x, bl_y = self.pagewise_xy
        tr_x = bl_x + self.size[0]
        tr_y = bl_y + self.size[1]
        return (bl_x <= x <= tr_x) and (bl_y <= y <= tr_y)

    def die(self):
        print("Deleting element:", self.name)
        self._gfx_objects.clear()

    def register_hoverable(self):
        self.page.hoverable_elements.append(self)

    def register_clickable(self):
        self.page.clickable_elements.append(self)

    def add_pointer_move_inside_callback(self, callback):
        self._callback_on_pointer_move_inside = callback
    
    def add_pointer_move_outside_callback(self, callback):
        self._callback_on_pointer_move_outside = callback

    def add_pointer_click_callback(self, callback):
        self._callback_pointer_click = callback

    def on_pointer_move_inside(self, event, page_coords):
        # 1. graphical things proper to this element
        ...
        # 2. user-defined callback
        if self._callback_on_pointer_move_inside is not None:
            self._callback_on_pointer_move_inside(event, self, page_coords)

    def on_pointer_move_outside(self, event, page_coords):
        # 1. graphical things proper to this element
        ...
        # 2. user-defined callback
        if self._callback_on_pointer_move_outside is not None:
            self._callback_on_pointer_move_outside(event, self, page_coords)

    def on_pointer_down_inside(self, event, page_coords):
        # graphical things proper to this element
        ...

    def on_pointer_up_inside(self, event, page_coords):
        # 1. graphical things proper to this element
        ...
        # 2. user-defined callback
        if self._callback_pointer_click is not None:
            self._callback_pointer_click(event, self, page_coords)

    def stop_pointer_down_effect(self):
        # graphical things proper to this element
        ...

class Element_2d(_GraphicalElement):
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, colour=None): # pos_xy_rel: bottom-left corner
        size_xyz_px  = (size_xy_rel[0]*parent.size[0], size_xy_rel[1]*parent.size[1], 0)
        center_px    = (parent.bl[0] + bl_xy_rel[0]*parent.size[0] + size_xyz_px[0]/2, parent.bl[1] + bl_xy_rel[1]*parent.size[1] + size_xyz_px[1]/2, parent.bl[2])
        super().__init__(unique_name, parent, center_px, size_xyz_px, colour=colour)

    def _make_borders(self, borders, thickness=2.0):
        if sum(borders) == 0:
            return  # no borders to create
        
        # Define corners relative to center (for proper rotation around center)
        hw, hh = self.size[0] / 2, self.size[1] / 2
        lines = []
        
        if borders[0] == 1:  # top
            two_points = pygfx.Geometry(positions=np.array([
                [-hw,  hh, 0],
                [ hw,  hh, 0],
            ], dtype=np.float32))
            lines.append(pygfx.Line(two_points, pygfx.LineMaterial(color=interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5), thickness=thickness, aa=True)))
        if borders[1] == 1:  # right
            two_points = pygfx.Geometry(positions=np.array([
                [ hw,  hh, 0],
                [ hw, -hh, 0],
            ], dtype=np.float32))
            lines.append(pygfx.Line(two_points, pygfx.LineMaterial(color=interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5), thickness=thickness, aa=True)))
        if borders[2] == 1:  # bottom
            two_points = pygfx.Geometry(positions=np.array([
                [ hw, -hh, 0],
                [-hw, -hh, 0],
            ], dtype=np.float32))
            lines.append(pygfx.Line(two_points, pygfx.LineMaterial(color=interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5), thickness=thickness, aa=True)))
        if borders[3] == 1:  # left
            two_points = pygfx.Geometry(positions=np.array([
                [-hw, -hh, 0],
                [-hw,  hh, 0],
            ], dtype=np.float32))
            lines.append(pygfx.Line(two_points, pygfx.LineMaterial(color=interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5), thickness=thickness, aa=True)))
        
        self.border_lines = lines
        for line in self.border_lines:
            # Position at center (rotation will work correctly around this point)
            line.local.position = self.center
            self.register_gfx_object(line)

class Element_3d(_GraphicalElement):
    def __init__(self, unique_name, parent, bl_xyz_px, size_xyz_px, colour=None):
        center_px = (bl_xyz_px[0] + size_xyz_px[0]/2, bl_xyz_px[1] + size_xyz_px[1]/2, bl_xyz_px[2] + size_xyz_px[2]/2)
        super().__init__(unique_name, parent, center_px, size_xyz_px, colour=colour)

class Container(Element_2d):
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, borders = (0, 0, 0, 0), colour=None):
        # borders: (top, right, bottom, left) "0" means no border, "1" means full border
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, colour=colour)
        self.is_leaf  = False
        self.children = []
        self.children_dict = {}
        self.border_flags = borders
        self._make_borders(borders)
    
    def add(self, child_element):
        child_element.parent = self
        self.children.append(child_element)
        self.children_dict[child_element.name] = child_element
    
    def get(self, child_name):
        if child_name in self.children_dict:
            return self.children_dict[child_name]
        else:
            for child in self.children:
                if not child.is_leaf:
                    result = child.get_child(child_name)
                    if result is not None:
                        return result
        return None

    def die(self):
        for child in self.children:
            child.die() # well, that turned out darker than I thought
        self.children.clear()
        self.children_dict.clear()
        super().die()

    