import wgpu
import pygfx
import numpy as np


class OverlayParticles:
    """
    A fixed-size particle system for screen-space overlay rendering.
    
    Not an Element - operates in raw screen coordinates as an overlay.
    No borders, no element hierarchy, just particles above everything.
    
    Parameters
    ----------
    scene : pygfx.Scene
        The scene to add particles to.
    n_particles : int
        Number of particles (fixed at init).
    point_size : int
        Size of particles in pixels.
    colour : tuple
        RGBA colour for particles.
    """
    
    def __init__(self, scene, n_particles=50, point_size=4, colour=(1.0, 1.0, 1.0, 1.0)):
        self.scene = scene
        self.n = n_particles
        
        # Pre-allocated CPU arrays (never resized)
        self._positions = np.zeros((self.n, 3), dtype=np.float32)
        self._velocities = np.zeros((self.n, 3), dtype=np.float32)

        self._positions[:, 2] = 10.0
        
        # GPU buffer - pre-allocated, reused for all updates
        self._positions_buffer = pygfx.Buffer(
            data=self._positions,
            usage=wgpu.BufferUsage.COPY_DST,
        )
        
        # Geometry and material
        self._geometry = pygfx.Geometry(positions=self._positions_buffer)
        self._material = pygfx.PointsMaterial(size=point_size, color=colour)
        self._points = pygfx.Points(self._geometry, self._material)
        
        self.scene.add(self._points)
        
        # State for continuous updates
        self._mode = 'chasing'  # 'inside', 'chasing'
        self.is_idle = True
        self._target_element = None
        self._cursor_scene_xy = (0.0, 0.0)
        
        # Start hidden
        self.hide()
    
    def tick(self, cursor_scene_xy=None):
        """Per-frame update. Call from page.one_frame()."""
        if self.is_idle:
            return
        
        # Update cursor position if provided (for chasing mode)
        if cursor_scene_xy is not None:
            self._cursor_scene_xy = cursor_scene_xy
        
        cursor_x, cursor_y = self._cursor_scene_xy
        pos = self._positions
        vel = self._velocities
        
        if self._mode == 'inside' and self._target_element is not None:
            el = self._target_element
            el_bl = el.bl
            el_sz = el.size
            
            for i in range(self.n):
                px, py = pos[i, 0], pos[i, 1]
                inside = (el_bl[0] <= px <= el_bl[0] + el_sz[0]) and (el_bl[1] <= py <= el_bl[1] + el_sz[1])
                
                if inside:
                    vel[i, 0] += np.random.uniform(-8.0, 8.0)
                    vel[i, 1] += np.random.uniform(-8.0, 8.0)
                else:
                    el_cx = el_bl[0] + el_sz[0] / 2
                    el_cy = el_bl[1] + el_sz[1] / 2
                    dx, dy = el_cx - px, el_cy - py
                    dist = max(np.sqrt(dx*dx + dy*dy), 1.0)
                    vel[i, 0] += dx / dist * 15.0
                    vel[i, 1] += dy / dist * 15.0
                
                # Slight pull toward cursor
                dx_c, dy_c = cursor_x - px, cursor_y - py
                dist_c = max(np.sqrt(dx_c*dx_c + dy_c*dy_c), 1.0)
                vel[i, 0] += dx_c / dist_c * 1.0
                vel[i, 1] += dy_c / dist_c * 1.0
            
            self.decay_velocity(0.75)
            self.update_velocity(5.0)
        
        elif self._mode == 'chasing':
            """ for i in range(self.n):
                px, py = pos[i, 0], pos[i, 1]
                dx, dy = cursor_x - px, cursor_y - py
                dist = np.sqrt(dx*dx + dy*dy)
                vel[i, 0] += dx / max(dist, 1.0) * 25.0
                vel[i, 1] += dy / max(dist, 1.0) * 25.0
            
            self.decay_velocity(0.5)
            self.update_velocity(3.0) """
            
            
            close_count = 0
            threshold = 20.0
            for i in range(self.n):
                px, py = pos[i, 0], pos[i, 1]
                dx, dy = cursor_x - px, cursor_y - py
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < threshold:
                    close_count += 1
                else:
                    vel[i, 0] += dx / max(dist, 1.0)
                    vel[i, 1] += dy / max(dist, 1.0)
            
            self.decay_velocity(0.1)
            self.update_velocity(600.0)
            
            if close_count >= 10:
                self.hide()
                self.is_idle = True
                self._target_element = None
    
    def enter_element(self, element, cursor_scene_xy):
        """Called when cursor enters a particle magnet element."""
        self._mode = 'inside'
        self.is_idle = False
        self._target_element = element
        self._cursor_scene_xy = cursor_scene_xy
        self.show()
    
    def update_cursor(self, cursor_scene_xy):
        """Update cursor position (call on mouse move)."""
        self._cursor_scene_xy = cursor_scene_xy
    
    def leave_element(self):
        """Called when cursor leaves the particle magnet element."""
        self._mode = 'chasing'
        self.is_idle = False
        self._target_element = None
        
        self.scene.add(self._points)
    
    def update_velocity(self, dt=1.0):
        """
        Apply velocities to positions in place (Euler integration).
        
        Parameters
        ----------
        dt : float
            Time step for integration.
        """
        self._positions += self._velocities * dt
        self._sync_to_gpu()
    
    def decay_velocity(self, decay_factor=0.95):
        """
        Decay all velocities in place by a multiplicative factor.
        
        Parameters
        ----------
        decay_factor : float
            Factor to multiply velocities by (< 1.0 for decay).
        """
        self._velocities *= decay_factor
    
    def add_noise_to_trajectories(self, noise_scale=1.0):
        """
        Add Gaussian noise to particle positions in place.
        
        Parameters
        ----------
        noise_scale : float
            Standard deviation of noise to add.
        """
        self._positions += (np.random.randn(self.n, 3) * noise_scale).astype(np.float32)
        self._sync_to_gpu()
    
    def set_velocities(self, velocities):
        """
        Set velocities from external array (in place copy).
        
        Parameters
        ----------
        velocities : np.ndarray
            Shape (n, 3) array of velocities.
        """
        np.copyto(self._velocities, velocities)
    
    def set_positions(self, positions):
        """
        Set positions from external array (in place copy).
        
        Parameters
        ----------
        positions : np.ndarray
            Shape (n, 3) array of positions.
        """
        np.copyto(self._positions, positions)
        self._sync_to_gpu()
    
    @property
    def positions(self):
        """Read-only view of current positions."""
        return self._positions
    
    @property
    def velocities(self):
        """Read-only view of current velocities."""
        return self._velocities
    
    def _sync_to_gpu(self):
        """Send current positions to GPU buffer."""
        self._positions_buffer.update_range(0, self.n)
    
    def hide(self):
        self._points.visible = False
    
    def show(self):
        self._points.visible = True
    
    def destroy(self):
        """Remove from scene."""
        if self._points.parent is not None:
            self._points.parent.remove(self._points)
