class _GraphicalElement:
    def __init__(self, unique_name, parent, center_xyz_px, size_xyz_px):
        self.name      = unique_name
        self.parent    = parent
        self.pos_xyz   = center_xyz_px
        self.size_xyz  = size_xyz_px
        self.is_leaf   = True

    @property
    def size(self):
        return self.size_xyz
    sz = size

    @property
    def center(self):
        return self.pos_xyz
    
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

    def die(self):
        print("Deleting element:", self.name)

class Element_2d(_GraphicalElement):
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel): # pos_xy_rel: bottom-left corner
        size_xyz_px  = (size_xy_rel[0]*parent.size[0], size_xy_rel[1]*parent.size[1], 0)
        center_px    = (parent.bl[0] + bl_xy_rel[0]*parent.size[0] + size_xyz_px[0]/2, parent.bl[1] + bl_xy_rel[1]*parent.size[1] + size_xyz_px[1]/2, parent.bl[2])
        super().__init__(unique_name, parent, center_px, size_xyz_px)

class Element_3d(_GraphicalElement):
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, z_depth_px): # pos_xy_rel: bottom-left corner
        size_xyz_px  = (size_xy_rel[0]*parent.size[0], size_xy_rel[1]*parent.size[1], z_depth_px)
        center_px    = (parent.bl[0] + bl_xy_rel[0]*parent.size[0] + size_xyz_px[0]/2, parent.bl[1] + bl_xy_rel[1]*parent.size[1] + size_xyz_px[1]/2, parent.bl[2] + size_xyz_px[2]/2)
        super().__init__(unique_name, parent, center_px, size_xyz_px)
    
    def __init__(self, unique_name, parent, bl_xyz_px, size_xyz_px):
        center_px = (bl_xyz_px[0] + size_xyz_px[0]/2, bl_xyz_px[1] + size_xyz_px[1]/2, bl_xyz_px[2] + size_xyz_px[2]/2)
        super().__init__(unique_name, parent, center_px, size_xyz_px)

class Container(Element_2d):
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel):
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel)
        self.is_leaf  = False
        self.children = []
        self.children_dict = {}
    
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