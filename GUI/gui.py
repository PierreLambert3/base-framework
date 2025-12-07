import multiprocessing
import time
from rendercanvas.auto import loop
import numpy as np

from GUI.engine.frontend.logic import Front_End
from GUI.pages.example_page import Page1

class Custom_Frontend(Front_End):
    def __init__(self, multiprocessing_context, queue_from_backend, queue_to_backend, shared_dict, window_name="Custom GUI Frontend Window"):
        super().__init__(queue_from_backend, queue_to_backend, shared_dict, window_name=window_name)
        self.set_fps(24)

    def build_pages(self):
        self.pages['main'] = Page1(self.scene, "The Main Page")

    def build_listeners(self):
        self.add_listener("backend ping", self.on_backend_ping)

    def on_user_event(self, event):
        if event["event_type"] == "pointer_move":
            x, y = event["x"], event["y"]
            self.update_shared_dict("pointer position", (x, y))

    def process_shared_dict(self):
        scatterplot = self.pages['main'].get("My Scatterplot")
        scatterplot_data = self.shared.get("scatterplot data")
        scatterplot.receive_data(scatterplot_data)

    def one_frame(self):
        if not self.should_it_render():
            self.scene.canvas.request_draw(self.one_frame)
            return
        
        # --- 1. logic ---
        self.process_messages()    # from Queue (careful not to saturate the queue)
        self.process_shared_dict() # from shared dict
        self.send("frontend ping", "One frame rendered")
        # scatterplot = self.pages['main'].get("My Scatterplot")
        # scatterplot.receive_data(np.random.rand(scatterplot.n, 3).astype(np.float32) * 2 - 1)

        # --- 2. render ---
        self.scene.render()

        # --- 3. schedule next frame ---
        self.scene.canvas.request_draw(self.one_frame)

    def routine(self):
        # 1. initialisation
        self.initialise_scene()
        self.build_pages()
        self.build_listeners()
        self.register_user_event_listener(self.on_user_event)

        # 2. Kickstart the loop
        self.scene.canvas.request_draw(self.one_frame)

        # 3. Blocking event loop: handles events and calls the custom one_frame().
        loop.run()

    def start(self):
        process = multiprocessing.Process(
            target = self.routine,
            args   = (),
            name   = "Frontend Process",
            daemon = False,
        )
        process.start()
        return process
    
    def on_backend_ping(self, data):
        print(f"Frontend received backend ping with data: {data}")