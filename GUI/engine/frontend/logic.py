import numpy as np
import time
from rendercanvas.auto import loop
from GUI.engine.comms import _Receiver, _Sender, _Listeners
from GUI.engine.frontend.scene import Scene

class Front_End:
    def __init__(self, queue_from_backend, queue_to_backend, window_name="GUI Frontend Window"):
        self.window_name = window_name
        # 1. rendering
        self.scene               = None
        self.frame_time_EMA      = 1.0 / 32.0
        self.last_frame_timstamp = time.time()

        # 2. communication
        self.receiver  = _Receiver(queue_from_backend) # from back end
        self.sender    = _Sender(queue_to_backend)     # to back end
        self.listeners = _Listeners(self.receiver)
        self.listeners.add("exit program", self.exit_program)

        # 3. pages and elements
        self.pages        = {}
        self.current_page = None

    def fps_update(self):
        timestamp = time.time()
        dt = timestamp - self.last_frame_timstamp
        self.last_frame_timstamp = timestamp
        self.frame_time_EMA = 0.95 * self.frame_time_EMA + (1 - 0.95) * dt

    def one_frame(self):
        raise Exception("--- Front_End.one_frame() should be implemented in the derived class! ---")
        """ self.fps_update()
        print("Frame Time EMA: {:.2f} ms".format(self.frame_time_EMA * 1000))

        # --- 1. logic ---
        ...

        # --- 2. render ---
        self.scene.render()

        # --- 3. schedule next frame ---
        self.scene.canvas.request_draw(self.one_frame) """

    def routine(self):
        raise Exception("--- Front_End.routine() should be implemented in the derived class! ---")
        """ # 1. initialisation
        self.initialise_scene()
        self.build_pages()
        self.build_listeners()

        # 2. Kickstart the loop
        self.scene.canvas.request_draw(self.one_frame)

        # 3. Blocking event loop: handles events and calls the custom one_frame().
        loop.run() """

    def initialise_scene(self):
        self.scene = Scene(canvas_title=self.window_name)

    def build_pages(self):
        # self.pages['main'] = MyPage(self.scene, "The Main Page")
        pass

    def build_listeners(self):
        pass

    

    def exit_program(self, data):
        print("---  Front_End: exit_program received, this should be implemented in the derived class!  ---")