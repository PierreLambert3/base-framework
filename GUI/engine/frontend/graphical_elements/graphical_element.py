# Wiki: wiki/05-pages-and-elements.md
# Engine base classes for the GUI tree: `_GraphicalElement` (root),
# `Element_2d` (relative-coords leaf), `Element_3d`, `Container` (recursive).
# Concrete leaf widgets live as siblings in this directory (button.py,
# scatterplot_2d_dynamic.py, ...).

import pygfx
import numpy as np
from pylinalg import quat_from_axis_angle, quat_from_euler, quat_mul
from GUI.engine.frontend.theme import ORANGE_RED, ORANGE_YELLOW, ORANGE_DARK, interpolate_color

class _GraphicalElement:
    def __init__(self, unique_name, parent, center_xyz_px, size_xyz_px, colour=None, background_colour=None, background_opacity=1.0, ignore_pointmode=False):
        self.name         = unique_name
        self.parent       = parent
        self.pos_xyz      = center_xyz_px
        self.size_xyz     = size_xyz_px
        self.is_leaf      = True
        self.colour       = colour if colour is not None else interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5)
        self.ignore_pointmode = ignore_pointmode
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
        self._points_mode_handles = []
        self._gfx_lines = []

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
            for h in self._points_mode_handles:
                h.hide()
    
    def show(self):
        if self.hidden:
            self.hidden = False
            for gfx_obj in self._gfx_objects:
                gfx_obj.visible = True
            for h in self._points_mode_handles:
                h.show()
    
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

    def add_lines(self, segments,
                  colour=None,
                  thickness=1.0, # only used when not in points mode
                  pointMode_n_points_mul=1.0,
                  pointMode_colour_range=(ORANGE_RED, ORANGE_DARK),
                  pointMode_spring_strength=(1.0, 0.0),
                  pointMode_jitter_strength=(1.0, 0.0),
                  pointMode_dt=(1.0, 0.3),
                  pointMode_damping=(1.0, 0.0),
                  pointMode_line_upwards_interaction=(0.0, 0.0),
                  looking_at_locations=None,
                  invert_lookat=False):
        """
        Add line segments to the graphical element.
        `segments` should be a list of ((x1, y1, z1), (x2, y2, z2)) in local object space.

        When `theme.POINTS_MODE` is set, the `pointMode_*` arguments modulate
        the swarm of points used to render each line:
          - `pointMode_n_points_mul`: float, multiplies the auto-computed
            point count per segment.
          - `pointMode_colour_range`: pair of colours; per-point colour is
            sampled as `c1 + (c2 - c1) * U[0,1]**2`. Defaults to
            (ORANGE_WHITE, PURPLE_DARK) when not provided. Pass an explicit
            `colour` argument and leave this unset for a uniform colour.
          - `pointMode_spring_strength` / `_jitter_strength` / `_dt`
            / `_damping`: `(mu, std)` tuples; one multiplier per point is
            sampled as `mu + N(0, std)` and multiplies the corresponding
            global kernel parameter on the GPU.
          - `pointMode_line_upwards_interaction`: `(mu, std)` tuple for the
            per-point upwards-interaction scalar. Positive values attract
            points that are on the normal side of the line toward their
            projection; negative values do the same for the opposite side.
            Intensity decays as 1/r². Zero (default) disables the interaction.
          - `looking_at_locations`: list of one 3D point per segment (in local
            object space). The up-normal for each line is computed as
            `looking_at_locations[i] - midpoint(segment[i])`. When omitted
            (or an empty list), the mean of all segment endpoints is used as a
            shared look-at target (with a small per-segment noise offset), so
            e.g. all edges of a rectangle automatically look toward its center.
          - `invert_lookat`: if True, flip the computed up-normal direction.
        """
        from GUI.engine.frontend import theme as _theme
        if colour is None:
            colour = self.colour

        if _theme.POINTS_MODE and not self.ignore_pointmode:
            # Default colour range only kicks in if the caller did not
            # explicitly pass one AND did not override `colour`.
            colour_range = pointMode_colour_range
            if colour_range is None and colour is self.colour:
                colour_range = (_theme.ORANGE_WHITE, _theme.PURPLE_DARK)

            point_mods = {
                "n_points_mul":             float(pointMode_n_points_mul),
                "colour_range":             colour_range,
                "spring_strength":          pointMode_spring_strength,
                "jitter_strength":          pointMode_jitter_strength,
                "dt":                       pointMode_dt,
                "damping":                  pointMode_damping,
                "line_upwards_interaction": pointMode_line_upwards_interaction,
            }

            # Compute per-segment up-normal vectors in local object space.
            # `looking_at_locations` entries are in the same local space as `segments`.
            segs = list(segments)
            n_segs = len(segs)
            if looking_at_locations and len(looking_at_locations) >= n_segs:
                look_at_list = [np.asarray(looking_at_locations[k], dtype=np.float32)
                                for k in range(n_segs)]
            else:
                # Default: mean of all endpoints + tiny independent noise per segment.
                all_pts = np.array([pt for seg in segs for pt in seg], dtype=np.float32)
                mean_pt = all_pts.mean(axis=0)
                noise_scale = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))) * 1e-4
                noise_scale = max(noise_scale, 1e-6)
                look_at_list = [
                    mean_pt + np.random.randn(3).astype(np.float32) * noise_scale for _ in range(n_segs)
                ]

            line_up_vectors = []
            for k, (p1, p2) in enumerate(segs):
                mid = (np.asarray(p1, dtype=np.float32) + np.asarray(p2, dtype=np.float32)) * 0.5
                if invert_lookat:
                    line_up_vectors.append(mid - look_at_list[k])
                else:
                    line_up_vectors.append(look_at_list[k] - mid)

            cx, cy, cz = self.center
            abs_segments = []
            for (p1, p2) in segs:
                abs_segments.append((
                    (p1[0] + cx, p1[1] + cy, p1[2] + cz),
                    (p2[0] + cx, p2[1] + cy, p2[2] + cz),
                ))
            pm = self.page._ensure_points_mode()
            handle = pm.register_lines(abs_segments, colour=colour, point_mods=point_mods,
                                       line_up_vectors=line_up_vectors)
            self._points_mode_handles.append(handle)
        else:
            positions = []
            for p1, p2 in segments:
                positions.append(p1)
                positions.append(p2)
            positions = np.array(positions, dtype=np.float32)

            geom = pygfx.Geometry(positions=positions)
            mat = pygfx.LineSegmentMaterial(color=colour, thickness=thickness, aa=True)
            line_obj = pygfx.Line(geom, mat)
            line_obj.local.position = self.center
            self._gfx_lines.append(line_obj)
            self.register_gfx_object(line_obj)

    def set_lines_colour(self, colour):
        from GUI.engine.frontend import theme as _theme
        if _theme.POINTS_MODE:
            for handle in self._points_mode_handles:
                handle.set_colour(colour)
        else:
            for line_obj in self._gfx_lines:
                line_obj.material.color = colour

    def set_lines_thickness(self, thickness):
        from GUI.engine.frontend import theme as _theme
        if not _theme.POINTS_MODE:
            for line_obj in self._gfx_lines:
                line_obj.material.thickness = thickness

    def die(self):
        """
        Destroy this element and remove all its pygfx objects from the scene.
        This properly deallocates GPU resources by removing objects from the scene graph.
        """
        for handle in self._points_mode_handles:
            handle.remove()
        self._points_mode_handles.clear()
        self._gfx_lines.clear()

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
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, colour=None, background_colour=None, ignore_pointmode=False): # pos_xy_rel: bottom-left corner
        size_xyz_px  = (size_xy_rel[0]*parent.size[0], size_xy_rel[1]*parent.size[1], 0)
        center_px    = (parent.bl[0] + bl_xy_rel[0]*parent.size[0] + size_xyz_px[0]/2, parent.bl[1] + bl_xy_rel[1]*parent.size[1] + size_xyz_px[1]/2, parent.bl[2])
        super().__init__(unique_name, parent, center_px, size_xyz_px, colour=colour, background_colour=background_colour, ignore_pointmode=ignore_pointmode)

    def _make_borders(self, borders, thickness=1.0):
        if sum(borders) == 0:
            return  # no borders to create

        border_colour = interpolate_color(ORANGE_YELLOW, ORANGE_DARK, 0.5)

        # Element-centered corners (relative to self.center).
        hw, hh = self.size[0] / 2, self.size[1] / 2
        top_l = (-hw,  hh, 0)
        top_r = ( hw,  hh, 0)
        bot_l = (-hw, -hh, 0)
        bot_r = ( hw, -hh, 0)

        segments = []
        if borders[0] == 1: segments.append((top_l, top_r))  # top
        if borders[1] == 1: segments.append((top_r, bot_r))  # right
        if borders[2] == 1: segments.append((bot_r, bot_l))  # bottom
        if borders[3] == 1: segments.append((bot_l, top_l))  # left

        self.add_lines(segments, colour=border_colour, thickness=thickness)

class Element_3d(_GraphicalElement):
    def __init__(self, unique_name, parent, bl_xyz_px, size_xyz_px, colour=None, ignore_pointmode=False):
        center_px = (bl_xyz_px[0] + size_xyz_px[0]/2, bl_xyz_px[1] + size_xyz_px[1]/2, bl_xyz_px[2] + size_xyz_px[2]/2)
        super().__init__(unique_name, parent, center_px, size_xyz_px, colour=colour, ignore_pointmode=ignore_pointmode)

class Container(Element_2d):
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, borders = (0, 0, 0, 0), colour=None, background_colour=None, ignore_pointmode=False):
        # borders: (top, right, bottom, left) "0" means no border, "1" means full border
        self.is_container = True
        self.is_leaf  = False
        self.children = []
        self.children_dict = {}
        self.border_flags = borders
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, colour=colour, background_colour=background_colour, ignore_pointmode=ignore_pointmode)
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