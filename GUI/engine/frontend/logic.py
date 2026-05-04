import numpy as np
import time
from rendercanvas.auto import loop
from GUI.engine.comms import _Listeners, Communications
from GUI.engine.frontend.scene import Scene

from GUI.engine.backend.logic import DEAD_AFTER_S
MIN_FPS    = 3.0 * 1.0 / (DEAD_AFTER_S)
PING_EVERY = DEAD_AFTER_S / 4.0

class Front_End:
    def __init__(self, multiprocessing_context, queue_from_backend, queue_to_backend, shared_dict, window_name="GUI Frontend Window"):
        self.multiprocessing_context = multiprocessing_context
        self.window_name = window_name
        self.mouse_coords = (0, 0)

        # 1. rendering
        self.scene               = None
        self.frame_time_EMA      = 1.0 / 32.0
        self.last_frame_timstamp = time.time()
        self.target_frametime    = 0.033

        # 2. communication
        self.listeners = _Listeners()
        self.comms     = Communications(queue_from_backend, queue_to_backend, shared_dict, self.listeners)
        self.add_listener("exit program", self.exit_program)

        # 3. pages and elements
        self.pages        = {}
        self.current_page = None

        # 4. keep alive ping
        self.last_ping = time.time()

    def set_fps(self, target_fps):
        self.target_frametime = 1.0 / max(MIN_FPS, target_fps)

    def should_it_render(self):
        timestamp = time.time()
        dt = timestamp - self.last_frame_timstamp
        should_render = dt >= self.target_frametime
        if should_render:
            self.last_frame_timstamp = timestamp
            self.frame_time_EMA = 0.95 * self.frame_time_EMA + (1 - 0.95) * dt
            return True
        return False

    def one_frame(self):
        raise Exception("--- Front_End.one_frame() should be implemented in the derived class! ---")

    def routine(self):
        raise Exception("--- Front_End.routine() should be implemented in the derived class! ---")

    def initialise_scene(self):
        self.scene = Scene(canvas_title=self.window_name)

    def build_pages(self):
        raise Exception("--- Front_End.build_pages() should be implemented in the derived class! ---")

    def build_listeners(self):
        raise Exception("--- Front_End.build_listeners() should be implemented in the derived class! ---")

    def on_user_event(self, event):
        raise Exception("--- Front_End.on_event() should be implemented in the derived class! ---")

    def register_user_event_listener(self, on_user_event):
        self.scene.canvas.add_event_handler(
            on_user_event,
            "pointer_down",
            "pointer_up",
            "pointer_move",
            "wheel",
            "key_down",
            "key_up"
        )

    def exit_program(self, data):
        loop.stop()
        self.scene.canvas.close()
        self.comms.send("exit program", None, needs_ack=False)
        self.comms.cancel_join_threads()

    def add_page(self, page_object):
        page_name = page_object.name
        assert page_name not in self.pages, f"Page with name '{page_name}' already exists."
        self.pages[page_name] = page_object
        if self.current_page is None:
            self.current_page = page_object
    
    def send(self, event_name, event_data=None, min_interval_s=0.0):
        self.comms.send(event_name, event_data)

    def process_messages(self):
        self.comms.process_messages()

    def update_shared_dict(self, key, value):
        self.comms.shared.set(key, value)

    def process_shared_dict(self):
        raise Exception("--- Back_End.process_shared_dict() should be implemented in the derived class! ---")

    def add_listener(self, event_name, callback):
        self.listeners.add(event_name, callback)

    def read_shared_dict(self, key, default=None):
        return self.comms.shared.get(key, default=default)
    
    def manage_mouse_pointer_move_in_page(self, event, current_page, page_coords):
        if self.current_page is not None:
            self.current_page.manage_mouse_pointer_move(event, page_coords)

    def manage_mouse_pointer_down_in_page(self, event, current_page, page_coords):
        if self.current_page is not None:
            self.current_page.manage_mouse_pointer_down(event, page_coords)

    def manage_mouse_pointer_up_in_page(self, event, current_page, page_coords):
        if self.current_page is not None:
            self.current_page.manage_mouse_pointer_up(event, page_coords)