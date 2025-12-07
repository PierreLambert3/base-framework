import numpy as np
import time
from rendercanvas.auto import loop
from GUI.engine.comms import _Receiver, _Sender, _Shared_dict, _Listeners
from GUI.engine.frontend.scene import Scene

class Front_End:
    def __init__(self, queue_from_backend, queue_to_backend, shared_dict, window_name="GUI Frontend Window"):
        self.window_name = window_name

        # 1. rendering
        self.scene               = None
        self.frame_time_EMA      = 1.0 / 32.0
        self.last_frame_timstamp = time.time()
        self.target_frametime    = 0.033

        # 2. communication
        self.listeners = _Listeners()
        self.receiver  = _Receiver(queue_from_backend, self.listeners) # from back end
        self.sender    = _Sender(queue_to_backend)     # to back end
        self.shared    = _Shared_dict(shared_dict)
        self.add_listener("exit program", self.exit_program)

        # 3. pages and elements
        self.pages        = {}
        self.current_page = None

    def set_fps(self, target_fps):
        self.target_frametime = 1.0 / max(0.1, target_fps)

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

    def add_listener(self, event_name, callback):
        self.listeners.add(event_name, callback)

    def exit_program(self, data):
        print("---  Front_End: exit_program received, this should be implemented in the derived class!  ---")

    def send(self, event_name, event_data=None, min_interval_s=0.0):
        self.sender.send(event_name, event_data, min_interval_s)

    def process_messages(self):
        self.receiver.process()

    def process_shared_dict(self):
        raise Exception("--- Front_End.process_shared_dict() should be implemented in the derived class! ---")
    
    def update_shared_dict(self, key, value):
        self.shared.set(key, value)

    def read_shared_dict(self, key, value):
        self.shared.set(key, value)