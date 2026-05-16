# Wiki: wiki/04-frontend.md (project-side frontend, scene, input, camera)
# Related: wiki/01-architecture.md, wiki/02-communications.md (per-instance
#          data-stream channel), wiki/05-pages-and-elements.md.
# Subclass `Custom_Frontend` for project-specific behaviour; do not modify
# `GUI/engine/frontend/logic.py` (the engine base class).

import multiprocessing
import time
from rendercanvas.auto import loop
import numpy as np

from GUI.engine.comms import _Listeners, Communications
from GUI.engine.frontend.logic import Front_End
from GUI.pages.IntroPage import Intro_Page
from GUI.pages.MainPage  import Main_Page
from GUI.pages.SharedMemoryPage import Shared_Memory_Page
from GUI.engine.worker.global_constants import SIMULATION_CHUNK_SIZE

class Custom_Frontend(Front_End):
    def __init__(self, multiprocessing_context, queue_from_backend, queue_to_backend, shared_dict, window_name="Custom GUI Frontend Window"):
        super().__init__(multiprocessing_context, queue_from_backend, queue_to_backend, shared_dict, window_name=window_name)
        
        # Per-instance data stream communications (keyed by instance_name)
        self.data_stream_comms_per_instance = {}  # instance_name -> Communications

        self.set_fps(26)
        self.simulation_chunk_size_timesteps = SIMULATION_CHUNK_SIZE[0]
        
        # Zoom settings
        self.default_camera_position = None   # Set after scene init
        self.default_camera_look_at  = None   # Set after scene init
        self.base_zoom_speed         = 40.0   # Fraction of distance to move per scroll
        self.min_distance_to_page    = 900.0  # Minimum distance camera can be from page
        self.current_look_at_target  = None
        self.look_at_lerp_factor     = 0.23

    def build_listeners(self): # communications with the backend
        self.add_listener("Q1: how many timesteps per simulation chunk", self._handle_how_many_timesteps_per_simulation_chunk)
        self.add_listener("new worker instance created",                self._handle_new_worker_instance_created)
        self.add_listener("worker instance info",                       self._handle_worker_instance_info)

    def _handle_worker_instance_info(self, data):
        """Project-specific metadata sent by a worker instance once it has
        finished `initialise()`. Forwarded to the current page if it cares."""
        if self.current_page is not None and hasattr(self.current_page, "on_worker_instance_info"):
            self.current_page.on_worker_instance_info(data)
    
    def load_intro_page(self):
        self.set_fps(26)
        self.scene.camera.local.position = (1000, 800, 2000)
        self._set_camera_look_at((1000, 800, 0.0))
        self._store_default_camera_state()
        self.add_page(Intro_Page(self.scene, "The Main Page", self))
    
    def switch_to_main_page(self):
        self.set_fps(20)
        if self.current_page is not None:
            self.current_page.destroy()
        self.scene.camera.local.position = (1000, 800, 2000)
        self._set_camera_look_at((1000, 800, 0.0))
        self._store_default_camera_state()
        self.add_page(Main_Page(self.scene, "main page", self))

    def switch_to_shared_memory_page(self):
        self.set_fps(20)
        if self.current_page is not None:
            self.current_page.destroy()
        self.scene.camera.local.position = (1000, 800, 2000)
        self._set_camera_look_at((1000, 800, 0.0))
        self._store_default_camera_state()
        self.add_page(Shared_Memory_Page(self.scene, "shared memory page", self))
    
    # ---------------------------------------------------- data stream wiring
    def register_data_stream_comms_for_instance(self, instance_name, queue_from_backend, queue_to_backend):
        """Create dedicated data stream Communications for a worker instance."""
        listeners = _Listeners()
        comms     = Communications(queue_from_backend, queue_to_backend, self.comms.shared, listeners)
        self.data_stream_comms_per_instance[instance_name] = {"comms": comms, "listeners": listeners}

    def add_data_stream_listener(self, instance_name, event_name, callback):
        """Register a listener on a specific instance's data stream channel."""
        if instance_name not in self.data_stream_comms_per_instance:
            print(f"Warning: no data stream comms registered for instance '{instance_name}'")
            return
        self.data_stream_comms_per_instance[instance_name]["listeners"].add(event_name, callback)

    def _handle_new_worker_instance_created(self, data):
        """Backend has spawned a worker instance: hook up data stream comms,
        and let the current page allocate visualisation resources."""
        instance_name = data["instance name"]
        config        = data.get("config", {})
        q_from_back, q_to_back = data["data_stream_queues"]
        self.register_data_stream_comms_for_instance(instance_name, q_from_back, q_to_back)
        if self.current_page is not None and hasattr(self.current_page, "on_new_worker_instance"):
            self.current_page.on_new_worker_instance(instance_name, config)
    
    def _store_default_camera_state(self):
        """Store the current camera position and orientation as the default state."""
        pos = self.scene.camera.local.position
        pos_arr = np.array([pos[0], pos[1], pos[2]], dtype=float)
        self.default_camera_position = pos_arr

        # If we do not yet track a look-at target, aim at the page center beneath the camera.
        if self.current_look_at_target is None:
            inferred_target = np.array([pos[0], pos[1], 0.0], dtype=float)
            self._set_camera_look_at(inferred_target)

        # Persist the current look-at target so we can restore it on zoom-out.
        self.default_camera_look_at = np.array(self.current_look_at_target, dtype=float)

    def _set_camera_look_at(self, target):
        target_arr = np.array(target, dtype=float)
        self.current_look_at_target = target_arr
        self.scene.camera.look_at(tuple(target_arr))

    def _smooth_camera_look_at(self, target):
        if target is None:
            return
        target_arr = np.array(target, dtype=float)
        if self.current_look_at_target is None:
            self._set_camera_look_at(target_arr)
            return
        new_target = self.current_look_at_target + self.look_at_lerp_factor * (target_arr - self.current_look_at_target)
        self._set_camera_look_at(new_target)

    def exit_program(self, data):
        super().exit_program(data)
        # Clean up data stream comms for all instances
        for instance_data in self.data_stream_comms_per_instance.values():
            instance_data["comms"].empty_queues()

    def on_user_event(self, event):
        mouse_event = (event["event_type"] == "pointer_move" or event["event_type"] == "pointer_down" or event["event_type"] == "pointer_up")
        if mouse_event:
            screen_mouse_coords = (event["x"], event["y"])
            self.mouse_coords = screen_mouse_coords
            if self.current_page is not None:
                page_coords = self.scene.xy_on_mesh(screen_mouse_coords, self.current_page.pick_mesh)
                if page_coords is not None:
                    if event["event_type"] == "pointer_move":
                        self.manage_mouse_pointer_move_in_page(event, self.current_page, page_coords)
                    elif event["event_type"] == "pointer_down":
                        self.manage_mouse_pointer_down_in_page(event, self.current_page, page_coords)
                    elif event["event_type"] == "pointer_up":
                        self.manage_mouse_pointer_up_in_page(event, self.current_page, page_coords)
        elif event["event_type"] == "wheel":
            self._handle_wheel_event(event)
        else:
            key_up_event = (event["event_type"] == "key_up")
            if key_up_event:
                key = event["key"] 
                if key == "Escape":
                    self.exit_program(0)
                elif key == "f":
                    current_index = SIMULATION_CHUNK_SIZE.index(self.simulation_chunk_size_timesteps)
                    next_index = (current_index + 1) % len(SIMULATION_CHUNK_SIZE)
                    self.simulation_chunk_size_timesteps = SIMULATION_CHUNK_SIZE[next_index]
                    self._handle_how_many_timesteps_per_simulation_chunk(None)
    
    def _handle_wheel_event(self, event):
        """Handle mouse wheel: first try dispatching to scrollable elements, then camera zoom."""
        if self.current_page is not None:
            screen_coords = (event["x"], event["y"])
            page_coords = self.scene.xy_on_mesh(screen_coords, self.current_page.pick_mesh)
            if page_coords is not None:
                # Try to dispatch to scrollable element under cursor
                if self.current_page.manage_mouse_wheel(event, page_coords):
                    return  # consumed by element
        # Otherwise, handle as camera zoom
        self._handle_wheel_zoom(event)

    def _handle_wheel_zoom(self, event):
        """Handle mouse wheel scrolling for zoom in/out."""
        dy = event.get("dy", 0)
        if dy == 0:
            return
        
        screen_coords = (event["x"], event["y"])
        cam_pos = np.array(self.scene.camera.local.position, dtype=float)
        if dy < 0:  # Scroll up = zoom in towards page
            if self.current_page is not None:
                hit_world, distance = self.scene.world_hit_on_mesh(screen_coords, self.current_page.pick_mesh)
                if distance is None or distance < 1e-6:
                    return
                speed = self.base_zoom_speed
                speed *= 2.0 / (1.0 + distance * 0.3)
                if hit_world is not None:
                    if distance > self.min_distance_to_page:
                        # Move camera towards the hit point
                        direction = hit_world - cam_pos
                        move_distance = min(distance - self.min_distance_to_page, distance * speed)
                        direction_norm = np.linalg.norm(direction)
                        if direction_norm > 1e-6:
                            direction /= direction_norm
                            new_pos = cam_pos + direction * move_distance
                            self.scene.camera.local.position = tuple(new_pos)
                    # Always start orienting toward the hovered point on the page
                    self._smooth_camera_look_at(hit_world)
        
        else:  # Scroll down = zoom out towards default position
            if self.default_camera_position is not None:
                direction = self.default_camera_position - cam_pos
                distance = np.linalg.norm(direction)
                if distance is None or distance < 1e-6:
                    return
                speed = self.base_zoom_speed
                speed *= 2.4 / (1.0 + distance * 0.2)
                # Clamp so we never overshoot past the default position.
                if distance <= 1.0:
                    self.scene.camera.local.position = tuple(self.default_camera_position)
                    if self.default_camera_look_at is not None:
                        self._set_camera_look_at(self.default_camera_look_at)
                    return

                move_distance = distance * speed
                if move_distance >= distance:
                    self.scene.camera.local.position = tuple(self.default_camera_position)
                    if self.default_camera_look_at is not None:
                        self._set_camera_look_at(self.default_camera_look_at)
                    return

                unit_dir = direction / distance
                new_pos = cam_pos + unit_dir * move_distance
                self.scene.camera.local.position = tuple(new_pos)
                if self.default_camera_look_at is not None:
                    self._smooth_camera_look_at(self.default_camera_look_at)

    def process_shared_dict(self):
        pass

    def one_frame(self):
        if not self.should_it_render():
            self.scene.canvas.request_draw(self.one_frame)
            return

        # --- 1. render ---
        self.scene.render()

        # --- 2. schedule next frame ---
        self.scene.canvas.request_draw(self.one_frame)

        # --- 3. logic ---
        self.process_messages()
        if self.current_page is not None:
            self.current_page.one_frame(self.mouse_coords)

    def process_messages(self):
        self.comms.process_messages()
        # Process data stream comms for all instances (high-throughput, no ACK)
        for instance_data in self.data_stream_comms_per_instance.values():
            instance_data["comms"].process_messages()

    def routine(self):
        # 1. initialisation
        try:
            self.initialise_scene()
            self.load_intro_page()
            self.build_listeners()
            self.register_user_event_listener(self.on_user_event)
        except Exception as e:
            print(f"--- Frontend initialisation error: {e} ---")
            self.send("exit program", 1)
            return

        # 2. Kickstart the loop
        self.scene.canvas.request_draw(self.one_frame)

        # 3. Blocking event loop: handles events and calls the custom one_frame().
        loop.run()

        # 4. notify backend of exit
        self.send("exit program", 1)

    def start(self):
        process = multiprocessing.Process(
            target = self.routine,
            args   = (),
            name   = "Frontend Process",
            daemon = False,
        )
        process.start()
        return process
    
    def _handle_how_many_timesteps_per_simulation_chunk(self, _):
        self.send("RE1.1: how many timesteps per simulation chunk", self.simulation_chunk_size_timesteps)
