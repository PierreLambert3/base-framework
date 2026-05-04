import pygfx
import numpy as np
from pylinalg import quat_from_axis_angle, quat_from_euler, quat_mul
from GUI.engine.frontend.theme import ORANGE_YELLOW, ORANGE_DARK, interpolate_color

class _GraphicalElement:
    def __init__(self, unique_name, parent, center_xyz_px, size_xyz_px, colour=None, background_colour=None, background_opacity=1.0):
        self.name         = unique_name
        self.parent       = parent
        self.pos_xyz      = center_xyz_px
        self.size_xyz     = size_xyz_px
        self.is_leaf      = True
        self.colour       = colour if colour is not None else interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5)
        self._rotation    = (0, 0, 0, 1)  # quaternion (x, y, z, w) - identity rotation
        self._gfx_objects = []  # list of pygfx objects that need rotation applied
        self.pagewise_xy  = self._make_page_coordinates()
        self.hidden       = False

        self._background_colour = background_colour
        self._background_opacity = background_opacity
        self._background_mesh = None
        if background_colour is not None:
            self._create_background_mesh()

        self._callback_on_pointer_move_inside    = None
        self._callback_on_pointer_move_outside   = None
        self._callback_pointer_click             = None

        # Particle magnet feature
        self.is_particle_magnet = False

        # If parent is a container, register self as its child
        if self.parent is not None and hasattr(self.parent, 'is_container'):
            self.parent.add(self)

    @property
    def size(self):
        return self.size_xyz
    sz = size

    @property
    def center(self):
        return self.pos_xyz
    pos = center

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
    def scene_wrapper(self):
        if not self.is_page:
            return self.parent.scene_wrapper
        return self._scene_wrapper

    @property
    def page(self):
        if self.is_page:
            return self
        return self.parent.page
    
    def move_to(self, new_center_xyz_px):
        self.pos_xyz = new_center_xyz_px
        self.pagewise_xy = self._make_page_coordinates()
        for i, gfx_obj in enumerate(self._gfx_objects):
            # If background mesh, push slightly back in z
            if gfx_obj is self._background_mesh:
                gfx_obj.local.position = (self.center[0], self.center[1], self.center[2] - 0.01)
            else:
                gfx_obj.local.position = self.center
    
    def translate(self, delta_xyz_px):
        new_center = (self.pos_xyz[0] + delta_xyz_px[0],
                      self.pos_xyz[1] + delta_xyz_px[1],
                      self.pos_xyz[2] + delta_xyz_px[2])
        self.move_to(new_center)
    
    def register_gfx_object(self, gfx_obj):
        self.scene.add(gfx_obj)
        self._gfx_objects.append(gfx_obj)
        if self.hidden:
            gfx_obj.visible = False

    def unregister_gfx_object(self, gfx_obj):
        self.scene.remove(gfx_obj)
        self._gfx_objects.remove(gfx_obj)

    def hide(self):
        if not self.hidden:
            self.hidden = True
            for gfx_obj in self._gfx_objects:
                gfx_obj.visible = False
    
    def show(self):
        if self.hidden:
            self.hidden = False
            for gfx_obj in self._gfx_objects:
                gfx_obj.visible = True
    
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

    def _create_background_mesh(self):
        import pygfx
        # Use a plane geometry sized to the element, centered at self.center
        w, h = self.size_xyz[0], self.size_xyz[1]
        geom = pygfx.geometries.plane_geometry(width=w, height=h)
        mat = pygfx.MeshBasicMaterial(color=self._background_colour, opacity=self._background_opacity)
        mesh = pygfx.Mesh(geom, mat)
        mesh.local.position = self.center
        # Optionally, push slightly back in z to avoid z-fighting with borders/text
        mesh.local.position = (self.center[0], self.center[1], self.center[2] - 0.01)
        self._background_mesh = mesh
        # Insert as first object so it's drawn behind everything else
        self._gfx_objects.insert(0, mesh)
        self.scene.add(mesh)
    
    def hit_by_page_coords(self, x, y): # in pixels in the page coordinate system
        bl_x, bl_y = self.pagewise_xy
        tr_x = bl_x + self.size[0]
        tr_y = bl_y + self.size[1]
        return (bl_x <= x <= tr_x) and (bl_y <= y <= tr_y)

    def die(self):
        """
        Destroy this element and remove all its pygfx objects from the scene.
        This properly deallocates GPU resources by removing objects from the scene graph.
        """
        # Make a copy of the list since we'll be modifying it during iteration
        gfx_objects_copy = list(self._gfx_objects)
        
        for gfx_obj in gfx_objects_copy:
            # Remove from scene graph (only if it's actually a child of the scene)
            if gfx_obj.parent is not None:
                gfx_obj.parent.remove(gfx_obj)
        
        # Clear our internal tracking list
        self._gfx_objects.clear()
        
        # Unregister from page's interactive element lists
        page = self.page
        if page is not None:
            if self in page.hoverable_elements:
                page.hoverable_elements.remove(self)
            if self in page.clickable_elements:
                page.clickable_elements.remove(self)
            if self in page.scrollable_elements:
                page.scrollable_elements.remove(self)
            if self in page.awaiting_mouse_up:
                page.awaiting_mouse_up.remove(self)
            if self in page.awaiting_hover_out:
                page.awaiting_hover_out.remove(self)

    def register_hoverable(self):
        if self not in self.page.hoverable_elements:
            self.page.hoverable_elements.append(self)

    def register_clickable(self):
        if self not in self.page.clickable_elements:
            self.page.clickable_elements.append(self)

    def register_scrollable(self):
        if self not in self.page.scrollable_elements:
            self.page.scrollable_elements.append(self)

    def add_pointer_move_inside_callback(self, callback):
        # if not registered as hoverable yet, do it now
        if self not in self.page.hoverable_elements:
            self.register_hoverable()
        self._callback_on_pointer_move_inside = callback
    
    def add_pointer_move_outside_callback(self, callback):
        # if not registered as hoverable yet, do it now
        if self not in self.page.hoverable_elements:
            self.register_hoverable()
        self._callback_on_pointer_move_outside = callback

    def add_pointer_click_callback(self, callback):
        # if not registered as clickable yet, do it now
        if self not in self.page.clickable_elements:
            self.register_clickable()
        self._callback_pointer_click = callback

    def on_pointer_move_inside(self, event, page_coords):
        # 1. graphical things proper to this element
        ...
        # 2. particle magnet behavior
        if self.is_particle_magnet:
            self._update_particles_inside(page_coords)
        # 3. user-defined callback
        if self._callback_on_pointer_move_inside is not None:
            self._callback_on_pointer_move_inside(event, self, page_coords)

    def on_pointer_move_outside(self, event, page_coords):
        # 1. graphical things proper to this element
        ...
        # 2. particle magnet behavior (rush to cursor)
        if self.is_particle_magnet:
            self._update_particles_outside(page_coords)
        # 3. user-defined callback
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

    def on_wheel(self, event, page_coords):
        """Handle wheel scroll event. Override in subclasses to implement scroll behavior."""
        pass

    def stop_pointer_down_effect(self):
        # graphical things proper to this element
        ...

    def enable_particle_magnet(self):
        """Enable particle magnet behavior for this element."""
        self.is_particle_magnet = True
        self.register_hoverable()

    def _update_particles_inside(self, page_coords):
        """Called when pointer moves inside a particle magnet element."""
        particles = getattr(self.page, 'overlay_particles', None)
        if particles is None:
            return
        
        # Convert page coords to scene coords
        page_bl = self.page.bl
        cursor_scene_xy = (page_bl[0] + page_coords[0], page_bl[1] + page_coords[1])
        
        # First entry or update cursor
        if particles._target_element is not self:
            particles.enter_element(self, cursor_scene_xy)
        else:
            particles.update_cursor(cursor_scene_xy)

    def _update_particles_outside(self, page_coords):
        """Called when pointer leaves a particle magnet element."""
        particles = getattr(self.page, 'overlay_particles', None)
        if particles is None:
            return
        
        # Update cursor position and switch to chasing mode
        page_bl = self.page.bl
        cursor_scene_xy = (page_bl[0] + page_coords[0], page_bl[1] + page_coords[1])
        particles.update_cursor(cursor_scene_xy)
        particles.leave_element()

class Element_2d(_GraphicalElement):
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, colour=None, background_colour=None): # pos_xy_rel: bottom-left corner
        size_xyz_px  = (size_xy_rel[0]*parent.size[0], size_xy_rel[1]*parent.size[1], 0)
        center_px    = (parent.bl[0] + bl_xy_rel[0]*parent.size[0] + size_xyz_px[0]/2, parent.bl[1] + bl_xy_rel[1]*parent.size[1] + size_xyz_px[1]/2, parent.bl[2])
        super().__init__(unique_name, parent, center_px, size_xyz_px, colour=colour, background_colour=background_colour)

    def _make_borders(self, borders, thickness=1.0):
        if sum(borders) == 0:
            return  # no borders to create
        
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
            line.local.position = self.center
            self.register_gfx_object(line)

class Element_3d(_GraphicalElement):
    def __init__(self, unique_name, parent, bl_xyz_px, size_xyz_px, colour=None):
        center_px = (bl_xyz_px[0] + size_xyz_px[0]/2, bl_xyz_px[1] + size_xyz_px[1]/2, bl_xyz_px[2] + size_xyz_px[2]/2)
        super().__init__(unique_name, parent, center_px, size_xyz_px, colour=colour)

class Container(Element_2d):
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, borders = (0, 0, 0, 0), colour=None, background_colour=None):
        # borders: (top, right, bottom, left) "0" means no border, "1" means full border
        self.is_container = True
        self.is_leaf  = False
        self.children = []
        self.children_dict = {}
        self.border_flags = borders
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, colour=colour, background_colour=background_colour)
        self._make_borders(borders)
        self.on_show = None  # user-defined callback when container is shown
    
    def add(self, child_element):
        child_element.parent = self
        for child in self.children:
            if child.name == child_element.name:
                print("WARNING: attempting to add child with duplicate name to container:", self.name, "  child name:", child_element.name)
                return
        self.children.append(child_element)
        self.children_dict[child_element.name] = child_element
        if self.hidden:
            child_element.hide()
    
    def get(self, child_name):
        if child_name in self.children_dict:
            return self.children_dict[child_name]
        else:
            for child in self.children:
                if not child.is_leaf:
                    result = child.get(child_name)
                    if result is not None:
                        return result
        return None
    
    def remove(self, child_element):
        if child_element in self.children:
            self.children.remove(child_element)
            del self.children_dict[child_element.name]
            child_element.die()
            child_element.parent = None
            return True
        return False
    
    def remove_by_name(self, child_name):
        # First check direct children
        if child_name in self.children_dict:
            child_element = self.children_dict[child_name]
            return self.remove(child_element)
        # Search recursively in nested containers
        for child in self.children:
            if not child.is_leaf:
                if child.remove_by_name(child_name):
                    return True
        return False

    def die(self):
        for child in self.children:
            child.die()
        self.children.clear()
        self.children_dict.clear()
        super().die()

    def hide(self):
        for child in self.children:
            child.hide()
        super().hide()
    
    def show(self):
        for child in self.children:
            child.show()
        super().show()
        if self.on_show is not None:
            self.on_show(self)

    def move_to(self, new_center_xyz_px):
        super().move_to(new_center_xyz_px)

    def translate(self, delta_xyz_px):
        for child in self.children:
            child.translate(delta_xyz_px)
        super().translate(delta_xyz_px)